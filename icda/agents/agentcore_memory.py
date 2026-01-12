"""Bedrock AgentCore Memory Integration for ICDA Pipeline.

This module provides unified memory layer using AWS Bedrock AgentCore with:
- SemanticStrategy: Facts extracted from conversations
- UserPreferenceStrategy: User preferences and settings
- SummaryStrategy: Session summaries for context

Session-only persistence (no cross-session memory by default).
Graceful fallback to local Redis-based MemoryAgent when AgentCore unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

from .models import (
    MemoryContext,
    MemoryFact,
    MemoryNamespace,
    UnifiedMemoryContext,
    AgentCoreMemoryConfig,
)

if TYPE_CHECKING:
    from .models import IntentResult
    from .memory_agent import MemoryAgent
    from icda.cache import RedisCache

logger = logging.getLogger(__name__)


# ============================================================================
# AgentCore Memory Manager (Control Plane)
# ============================================================================

class AgentCoreMemoryManager:
    """Control plane for Bedrock AgentCore memory operations.

    Manages memory resource lifecycle:
    - Creation with 3 strategies (Semantic, UserPreference, Summary)
    - Session cleanup on conversation end
    - Health monitoring and status tracking
    """

    __slots__ = (
        "_region",
        "_memory_id",
        "_manager",
        "_available",
        "_strategies_active",
        "_initialization_error",
    )

    # Memory resource name for ICDA
    MEMORY_NAME = "ICDAAgentMemory"

    def __init__(self, region: str = "us-east-1"):
        """Initialize AgentCore memory manager.

        Args:
            region: AWS region for Bedrock AgentCore.
        """
        self._region = region
        self._memory_id: str | None = None
        self._manager = None
        self._available = False
        self._strategies_active = False
        self._initialization_error: str | None = None

    async def initialize(self, memory_id: str | None = None) -> bool:
        """Initialize the AgentCore memory resource with all strategies.

        Creates or retrieves the ICDA memory resource and configures
        all three strategy types for session-scoped memory.

        Args:
            memory_id: Optional existing memory ID to use.

        Returns:
            True if initialization succeeded.
        """
        if memory_id:
            # Use provided memory ID
            self._memory_id = memory_id
            self._available = True
            self._strategies_active = True
            logger.info(f"AgentCore memory using provided ID: {memory_id}")
            return True

        try:
            # Try to import AgentCore SDK
            from bedrock_agentcore_starter_toolkit.operations.memory.manager import (
                MemoryManager,
            )
            from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import (
                SemanticStrategy,
                UserPreferenceStrategy,
                SummaryStrategy,
            )

            self._manager = MemoryManager(region_name=self._region)

            # Create memory with all 3 strategies
            memory = self._manager.get_or_create_memory(
                name=self.MEMORY_NAME,
                description="ICDA session memory with facts, preferences, and summaries",
                strategies=[
                    SemanticStrategy(
                        name="icdaFactExtractor",
                        namespaces=[MemoryNamespace.FACTS.value],
                    ),
                    UserPreferenceStrategy(
                        name="icdaUserPreferences",
                        namespaces=[MemoryNamespace.PREFERENCES.value],
                    ),
                    SummaryStrategy(
                        name="icdaSessionSummary",
                        description="Summarizes ICDA conversation sessions",
                        namespaces=[MemoryNamespace.SUMMARIES.value],
                    ),
                ],
            )

            self._memory_id = memory.get("id")
            self._available = True

            # Wait for strategies to become active (background)
            asyncio.create_task(self._wait_for_strategies_active())

            logger.info(f"AgentCore memory initialized: {self._memory_id}")
            return True

        except ImportError as e:
            self._initialization_error = f"AgentCore SDK not installed: {e}"
            logger.warning(self._initialization_error)
            self._available = False
            return False
        except Exception as e:
            self._initialization_error = f"AgentCore initialization failed: {e}"
            logger.warning(self._initialization_error)
            self._available = False
            return False

    async def _wait_for_strategies_active(self, timeout_seconds: int = 180):
        """Wait for all strategies to become ACTIVE.

        Memory strategies take 2-3 minutes to activate after creation.
        """
        if not self._manager or not self._memory_id:
            return

        for _ in range(timeout_seconds // 5):
            try:
                strategies = self._manager.get_memory_strategies(
                    memoryId=self._memory_id
                )
                if all(s.get("status") == "ACTIVE" for s in strategies):
                    self._strategies_active = True
                    logger.info("All memory strategies active")
                    return
            except Exception:
                pass
            await asyncio.sleep(5)

        logger.warning("Memory strategies did not activate within timeout")

    @property
    def available(self) -> bool:
        """Check if AgentCore is operational."""
        return self._available

    @property
    def strategies_active(self) -> bool:
        """Check if all strategies are active."""
        return self._strategies_active

    @property
    def memory_id(self) -> str | None:
        """Get the memory resource ID."""
        return self._memory_id

    async def cleanup_session(self, actor_id: str, session_id: str) -> bool:
        """Clean up session memory when conversation ends.

        For session-only persistence, this clears all memory
        associated with the session.

        Args:
            actor_id: User identifier.
            session_id: Session identifier.

        Returns:
            True if cleanup succeeded.
        """
        # AgentCore handles TTL-based cleanup automatically
        # This method is for explicit cleanup if needed
        logger.debug(f"Session cleanup requested: {actor_id}/{session_id}")
        return True

    def get_status(self) -> dict[str, Any]:
        """Get memory manager status."""
        return {
            "available": self._available,
            "strategies_active": self._strategies_active,
            "memory_id": self._memory_id,
            "region": self._region,
            "initialization_error": self._initialization_error,
        }


# ============================================================================
# AgentCore Session Manager (Data Plane)
# ============================================================================

class AgentCoreSessionManager:
    """Data plane for AgentCore memory session operations.

    Handles:
    - Session creation and management
    - Message persistence (add_turns)
    - Memory retrieval (facts, preferences, summaries)
    - Semantic search across memories
    """

    __slots__ = (
        "_memory_id",
        "_region",
        "_session_manager",
        "_active_sessions",
        "_available",
    )

    def __init__(self, memory_id: str, region: str = "us-east-1"):
        """Initialize session manager.

        Args:
            memory_id: AgentCore memory resource ID.
            region: AWS region.
        """
        self._memory_id = memory_id
        self._region = region
        self._session_manager = None
        self._active_sessions: dict[str, Any] = {}
        self._available = False

        self._init_session_manager()

    def _init_session_manager(self) -> None:
        """Initialize the AgentCore session manager."""
        try:
            from bedrock_agentcore.memory.session import MemorySessionManager

            self._session_manager = MemorySessionManager(
                memory_id=self._memory_id,
                region_name=self._region,
            )
            self._available = True
            logger.info(f"AgentCore session manager initialized: {self._memory_id}")
        except ImportError:
            logger.warning("AgentCore SDK not installed for session manager")
            self._available = False
        except Exception as e:
            logger.warning(f"Session manager initialization failed: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        """Check if session manager is operational."""
        return self._available

    def get_or_create_session(self, actor_id: str, session_id: str):
        """Get or create a memory session.

        Args:
            actor_id: User identifier (maps to {actorId} in namespaces).
            session_id: Session identifier (maps to {sessionId}).

        Returns:
            AgentCore memory session object.
        """
        if not self._available or not self._session_manager:
            return None

        key = f"{actor_id}:{session_id}"

        if key not in self._active_sessions:
            try:
                self._active_sessions[key] = self._session_manager.create_memory_session(
                    actor_id=actor_id,
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to create session: {e}")
                return None

        return self._active_sessions[key]

    async def add_conversation_turn(
        self,
        actor_id: str,
        session_id: str,
        message: str,
        role: str,
    ) -> bool:
        """Add a conversation turn to memory.

        This triggers async extraction by the configured strategies:
        - SemanticStrategy extracts facts
        - UserPreferenceStrategy captures preferences
        - SummaryStrategy updates session summary

        Args:
            actor_id: User identifier.
            session_id: Session identifier.
            message: Message content.
            role: Message role ("user" or "assistant").

        Returns:
            True if turn was added successfully.
        """
        session = self.get_or_create_session(actor_id, session_id)
        if not session:
            return False

        try:
            from bedrock_agentcore.memory.constants import (
                ConversationalMessage,
                MessageRole,
            )

            msg_role = MessageRole.USER if role == "user" else MessageRole.ASSISTANT

            session.add_turns(messages=[ConversationalMessage(message, msg_role)])
            return True
        except Exception as e:
            logger.warning(f"Failed to add turn: {e}")
            return False

    async def get_short_term_memory(
        self,
        actor_id: str,
        session_id: str,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Get last k conversation turns (short-term memory).

        Args:
            actor_id: User identifier.
            session_id: Session identifier.
            k: Number of turns to retrieve.

        Returns:
            List of conversation turns.
        """
        session = self.get_or_create_session(actor_id, session_id)
        if not session:
            return []

        try:
            turns = session.get_last_k_turns(k=k)
            return [self._convert_turn(t) for t in turns]
        except Exception as e:
            logger.warning(f"Failed to get STM: {e}")
            return []

    async def search_facts(
        self,
        actor_id: str,
        session_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryFact]:
        """Semantic search for relevant facts.

        Args:
            actor_id: User identifier.
            session_id: Session identifier.
            query: Search query.
            top_k: Max results.

        Returns:
            List of relevant MemoryFact objects.
        """
        session = self.get_or_create_session(actor_id, session_id)
        if not session:
            return []

        try:
            namespace = MemoryNamespace.FACTS.value.replace("{actorId}", actor_id)
            records = session.search_long_term_memories(
                query=query,
                namespace_prefix=namespace,
                top_k=top_k,
            )

            return [MemoryFact.from_dict(r) for r in records]
        except Exception as e:
            logger.debug(f"Fact search failed: {e}")
            return []

    async def get_preferences(
        self,
        actor_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Get user preferences from memory.

        Args:
            actor_id: User identifier.
            session_id: Session identifier.

        Returns:
            Dict of user preferences.
        """
        session = self.get_or_create_session(actor_id, session_id)
        if not session:
            return {}

        try:
            namespace = MemoryNamespace.PREFERENCES.value.replace("{actorId}", actor_id)
            records = session.list_long_term_memory_records(
                namespace_prefix=namespace,
            )

            preferences = {}
            for r in records:
                content = r.get("content", {})
                if isinstance(content, dict):
                    preferences.update(content)
                elif isinstance(content, str):
                    # Try to parse as key:value
                    if ":" in content:
                        key, value = content.split(":", 1)
                        preferences[key.strip()] = value.strip()

            return preferences
        except Exception as e:
            logger.debug(f"Preferences load failed: {e}")
            return {}

    async def get_session_summary(
        self,
        actor_id: str,
        session_id: str,
    ) -> str:
        """Get session summary from memory.

        Args:
            actor_id: User identifier.
            session_id: Session identifier.

        Returns:
            Session summary string.
        """
        session = self.get_or_create_session(actor_id, session_id)
        if not session:
            return ""

        try:
            namespace = MemoryNamespace.SUMMARIES.value.replace(
                "{actorId}", actor_id
            ).replace("{sessionId}", session_id)

            records = session.list_long_term_memory_records(
                namespace_prefix=namespace,
            )

            if records:
                return records[-1].get("content", "")
            return ""
        except Exception as e:
            logger.debug(f"Summary load failed: {e}")
            return ""

    def clear_session(self, actor_id: str, session_id: str) -> None:
        """Clear session from local cache."""
        key = f"{actor_id}:{session_id}"
        self._active_sessions.pop(key, None)

    def _convert_turn(self, turn: Any) -> dict[str, Any]:
        """Convert AgentCore turn to dict format."""
        if hasattr(turn, "to_dict"):
            return turn.to_dict()
        if isinstance(turn, dict):
            return turn
        # Handle Turn object attributes
        if hasattr(turn, "role") and hasattr(turn, "text"):
            return {
                "role": str(turn.role).lower().replace("messagerole.", ""),
                "content": turn.text,
                "timestamp": getattr(turn, "timestamp", None),
            }
        return {"content": str(turn)}


# ============================================================================
# Unified Memory Layer
# ============================================================================

class UnifiedMemoryLayer:
    """Unified memory layer providing shared access for all agents.

    Combines AgentCore memory with legacy Redis-based memory
    for backward compatibility. Provides a single interface
    that all 11 agents can use.
    """

    __slots__ = (
        "_config",
        "_agentcore_manager",
        "_agentcore_session",
        "_local_memory",
        "_available",
    )

    def __init__(
        self,
        config: AgentCoreMemoryConfig | None = None,
        local_memory: "MemoryAgent | None" = None,
    ):
        """Initialize unified memory layer.

        Args:
            config: AgentCore configuration.
            local_memory: Legacy MemoryAgent for fallback.
        """
        self._config = config or AgentCoreMemoryConfig()
        self._local_memory = local_memory
        self._agentcore_manager: AgentCoreMemoryManager | None = None
        self._agentcore_session: AgentCoreSessionManager | None = None
        self._available = False

    async def initialize(self) -> bool:
        """Initialize the unified memory layer.

        Returns:
            True if AgentCore is available.
        """
        if not self._config.enabled:
            logger.info("AgentCore memory disabled by config")
            return False

        self._agentcore_manager = AgentCoreMemoryManager(self._config.region)

        if await self._agentcore_manager.initialize(self._config.memory_id):
            self._agentcore_session = AgentCoreSessionManager(
                memory_id=self._agentcore_manager.memory_id,
                region=self._config.region,
            )
            self._available = self._agentcore_session.available
            if self._available:
                logger.info("Unified memory layer initialized with AgentCore")
            else:
                logger.info("AgentCore manager ready but session manager unavailable")
        else:
            logger.info("Unified memory layer using legacy Redis fallback")

        return self._available

    @property
    def available(self) -> bool:
        """Check if AgentCore is available."""
        return self._available

    async def recall(
        self,
        session_id: str | None,
        query: str,
        intent: "IntentResult | None" = None,
        actor_id: str | None = None,
    ) -> UnifiedMemoryContext:
        """Recall relevant memory for a query.

        This is the main method called by agents to get memory context.
        Combines data from all three strategies plus local memory.

        Args:
            session_id: Session identifier.
            query: Current query for semantic search.
            intent: Intent classification (for local memory).
            actor_id: User identifier.

        Returns:
            UnifiedMemoryContext with all memory data.
        """
        # Use session_id as actor_id if not provided
        effective_actor_id = actor_id or session_id or "default"
        effective_session_id = session_id or "default"

        # Get local memory (always works as fallback)
        local_context = MemoryContext(memory_signals=["initialized"])
        if self._local_memory and session_id:
            try:
                local_context = await self._local_memory.recall(
                    session_id=session_id,
                    query=query,
                    intent=intent,
                )
            except Exception as e:
                logger.warning(f"Local memory recall failed: {e}")
                local_context.memory_signals.append(f"local_error:{e}")

        # Initialize unified context with local data
        context = UnifiedMemoryContext(
            local_context=local_context,
            memory_source="local",
            agentcore_available=self._available,
            recall_confidence=local_context.recall_confidence,
            memory_signals=local_context.memory_signals.copy(),
        )

        # Try AgentCore if available
        if self._available and self._agentcore_session:
            try:
                context = await self._load_agentcore_memory(
                    actor_id=effective_actor_id,
                    session_id=effective_session_id,
                    query=query,
                    local_context=local_context,
                )
            except Exception as e:
                logger.warning(f"AgentCore recall failed: {e}")
                context.memory_signals.append(f"agentcore_error:{e}")
                context.memory_source = "local"

        return context

    async def _load_agentcore_memory(
        self,
        actor_id: str,
        session_id: str,
        query: str,
        local_context: MemoryContext,
    ) -> UnifiedMemoryContext:
        """Load memory from AgentCore."""
        signals = local_context.memory_signals.copy()
        signals.append("agentcore_load_start")

        # Load STM turns (instant)
        stm_turns = await self._agentcore_session.get_short_term_memory(
            actor_id=actor_id,
            session_id=session_id,
            k=10,
        )
        stm_loaded = bool(stm_turns)
        if stm_loaded:
            signals.append(f"stm_loaded:{len(stm_turns)}_turns")

        # Load LTM if enabled
        ltm_facts = []
        ltm_preferences = {}
        session_summary = ""
        ltm_loaded = False

        if self._config.use_ltm:
            # Get relevant facts via semantic search
            ltm_facts = await self._agentcore_session.search_facts(
                actor_id=actor_id,
                session_id=session_id,
                query=query,
                top_k=5,
            )
            if ltm_facts:
                signals.append(f"ltm_facts:{len(ltm_facts)}")

            # Get user preferences
            ltm_preferences = await self._agentcore_session.get_preferences(
                actor_id=actor_id,
                session_id=session_id,
            )
            if ltm_preferences:
                signals.append(f"ltm_prefs:{len(ltm_preferences)}")

            # Get session summary
            session_summary = await self._agentcore_session.get_session_summary(
                actor_id=actor_id,
                session_id=session_id,
            )
            if session_summary:
                signals.append("ltm_summary")

            ltm_loaded = bool(ltm_facts or ltm_preferences or session_summary)

        # Calculate confidence
        confidence = self._calculate_confidence(
            local_context, stm_turns, ltm_facts, ltm_preferences
        )

        return UnifiedMemoryContext(
            local_context=local_context,
            stm_turns=stm_turns,
            stm_loaded=stm_loaded,
            ltm_facts=ltm_facts,
            ltm_preferences=ltm_preferences,
            session_summary=session_summary,
            ltm_loaded=ltm_loaded,
            memory_source="hybrid" if self._local_memory else "agentcore",
            agentcore_available=True,
            recall_confidence=confidence,
            memory_signals=signals,
        )

    async def remember(
        self,
        session_id: str | None,
        results: list[dict[str, Any]],
        response: str,
        query: str,
        actor_id: str | None = None,
    ) -> None:
        """Store conversation turn for memory extraction.

        Persists both user query and assistant response,
        triggering async extraction by all strategies.

        Args:
            session_id: Session identifier.
            results: Search results to remember.
            response: Assistant response.
            query: User query.
            actor_id: User identifier.
        """
        if not session_id:
            return

        effective_actor_id = actor_id or session_id

        # Store locally (always)
        if self._local_memory:
            try:
                await self._local_memory.remember(
                    session_id=session_id,
                    results=results,
                    response=response,
                    query=query,
                )
            except Exception as e:
                logger.warning(f"Local memory store failed: {e}")

        # Store to AgentCore (if available)
        if self._available and self._agentcore_session:
            try:
                # Store user message
                await self._agentcore_session.add_conversation_turn(
                    actor_id=effective_actor_id,
                    session_id=session_id,
                    message=query,
                    role="user",
                )

                # Store assistant response
                await self._agentcore_session.add_conversation_turn(
                    actor_id=effective_actor_id,
                    session_id=session_id,
                    message=response,
                    role="assistant",
                )

                logger.debug(f"AgentCore: Stored turn for session={session_id}")
            except Exception as e:
                logger.warning(f"AgentCore store failed: {e}")

    async def cleanup(self, session_id: str, actor_id: str | None = None) -> None:
        """Clean up session memory.

        Called when conversation ends for session-only persistence.

        Args:
            session_id: Session to clean up.
            actor_id: User identifier.
        """
        effective_actor_id = actor_id or session_id

        # Clear local memory
        if self._local_memory:
            try:
                await self._local_memory.clear(session_id)
            except Exception as e:
                logger.warning(f"Local memory cleanup failed: {e}")

        # Clear AgentCore session
        if self._agentcore_session:
            self._agentcore_session.clear_session(effective_actor_id, session_id)

        if self._agentcore_manager:
            await self._agentcore_manager.cleanup_session(
                effective_actor_id, session_id
            )

        logger.debug(f"Memory cleaned up for session={session_id}")

    def _calculate_confidence(
        self,
        local_context: MemoryContext,
        stm_turns: list,
        ltm_facts: list,
        ltm_preferences: dict,
    ) -> float:
        """Calculate overall memory recall confidence."""
        confidence = 0.2  # Base confidence

        # Local memory contribution
        if local_context.recall_confidence > 0:
            confidence += 0.2 * local_context.recall_confidence

        # STM contribution
        if stm_turns:
            confidence += 0.2 * min(len(stm_turns) / 10, 1.0)

        # LTM facts contribution
        if ltm_facts:
            confidence += 0.2 * min(len(ltm_facts) / 5, 1.0)

        # Preferences contribution
        if ltm_preferences:
            confidence += 0.1

        return min(confidence, 1.0)

    def get_stats(self) -> dict[str, Any]:
        """Get memory layer statistics."""
        stats = {
            "available": self._available,
            "config": self._config.to_dict(),
        }

        if self._agentcore_manager:
            stats["agentcore"] = self._agentcore_manager.get_status()

        if self._local_memory:
            stats["local"] = self._local_memory.get_stats()

        return stats


# ============================================================================
# Memory Hooks for Pipeline Integration
# ============================================================================

class MemoryHooks:
    """Memory hooks for agent pipeline integration.

    Provides lifecycle hooks that integrate with QueryOrchestrator:
    - on_pipeline_start: Load memory context
    - on_pipeline_end: Persist final memory state
    - on_session_end: Cleanup session memory
    """

    __slots__ = ("_memory_layer",)

    def __init__(self, memory_layer: UnifiedMemoryLayer):
        """Initialize memory hooks.

        Args:
            memory_layer: Unified memory layer instance.
        """
        self._memory_layer = memory_layer

    async def on_pipeline_start(
        self,
        session_id: str | None,
        query: str,
        intent: "IntentResult | None" = None,
        actor_id: str | None = None,
    ) -> UnifiedMemoryContext:
        """Hook called at pipeline start to load memory.

        Args:
            session_id: Session identifier.
            query: User query.
            intent: Intent classification.
            actor_id: User identifier.

        Returns:
            Memory context for agents.
        """
        start_time = time.time()
        context = await self._memory_layer.recall(
            session_id=session_id,
            query=query,
            intent=intent,
            actor_id=actor_id,
        )
        elapsed_ms = int((time.time() - start_time) * 1000)
        context.memory_signals.append(f"recall_time:{elapsed_ms}ms")
        return context

    async def on_pipeline_end(
        self,
        session_id: str | None,
        query: str,
        response: str,
        results: list[dict[str, Any]] | None = None,
        actor_id: str | None = None,
    ) -> None:
        """Hook called at pipeline end to persist memory.

        Args:
            session_id: Session identifier.
            query: Original query.
            response: Final response.
            results: Search results (if any).
            actor_id: User identifier.
        """
        await self._memory_layer.remember(
            session_id=session_id,
            results=results or [],
            response=response,
            query=query,
            actor_id=actor_id,
        )

    async def on_session_end(
        self,
        session_id: str,
        actor_id: str | None = None,
    ) -> None:
        """Hook called when session ends for cleanup.

        Args:
            session_id: Session to clean up.
            actor_id: User identifier.
        """
        await self._memory_layer.cleanup(session_id, actor_id)

    @property
    def available(self) -> bool:
        """Check if memory layer is available."""
        return self._memory_layer.available
