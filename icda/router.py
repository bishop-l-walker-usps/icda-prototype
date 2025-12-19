import json
from time import time

from .cache import RedisCache
from .database import CustomerDB
from .guardrails import Guardrails, GuardrailFlags
from .nova import NovaClient
from .session import Session, SessionManager
from .vector_index import VectorIndex, RouteType


class Router:
    """Routes queries: Guardrails → Cache → Vector Route → DB/Nova with session context.

    IMPORTANT: Guardrails-aware caching:
    - When guardrails are OFF (all disabled), cache is SKIPPED entirely
    - Different guardrail states = different expected responses
    - Only cache responses when guardrails are active
    """
    __slots__ = ("cache", "vector_index", "db", "nova", "sessions")

    def __init__(self, cache: RedisCache, vector_index: VectorIndex, db: CustomerDB, nova: NovaClient, sessions: SessionManager):
        self.cache = cache
        self.vector_index = vector_index
        self.db = db
        self.nova = nova
        self.sessions = sessions

    def _are_guardrails_active(self, flags: GuardrailFlags | None) -> bool:
        """Check if any guardrails are enabled.

        Args:
            flags: Guardrail flags (None = all enabled by default).

        Returns:
            True if at least one guardrail is active.
        """
        if flags is None:
            return True  # Default: all guardrails enabled
        return any([
            flags.pii,
            flags.financial,
            flags.credentials,
            flags.offtopic,
        ])

    async def route(
        self,
        query: str,
        bypass_cache: bool = False,
        guardrails: dict | None = None,
        session_id: str | None = None
    ) -> dict:
        start = time()
        q = query.strip()

        # Get or create session
        session = await self.sessions.get(session_id)

        # Determine guardrails state
        flags = GuardrailFlags(**guardrails) if guardrails else None
        guardrails_active = self._are_guardrails_active(flags)

        # 1. Guardrails check (only if active)
        if guardrails_active:
            if blocked := Guardrails.check(q, flags):
                return self._response(
                    q, blocked, RouteType.CACHE_HIT, start,
                    blocked=True,
                    session_id=session.session_id,
                    guardrails_active=guardrails_active,
                    guardrails_bypassed=False,
                )

        # 2. Cache check - SKIP when guardrails bypassed or session has context
        # Different guardrail states = different expected responses, so don't mix
        key = RedisCache.make_key(q)
        should_use_cache = (
            not bypass_cache and
            not session.messages and
            guardrails_active  # NEW: Don't use cache when guardrails bypassed
        )

        if should_use_cache:
            if hit := await self.cache.get(key):
                data = json.loads(hit)
                return self._response(
                    q, data["response"], RouteType.CACHE_HIT, start,
                    cached=True,
                    session_id=session.session_id,
                    guardrails_active=guardrails_active,
                    guardrails_bypassed=False,
                )

        # 3. Vector routing
        route_type, metadata = await self.vector_index.find_route(q)

        # 4. Execute route
        if route_type == RouteType.DATABASE:
            tool = metadata.get("tool", "search_customers")
            result = self.db.execute(tool, q)
            if result["success"]:
                response = self._format_db_result(result, tool)
                # Store in session
                session.add_message("user", q)
                session.add_message("assistant", response)
                await self.sessions.save(session)
                # Cache only if no prior context AND guardrails active
                should_cache = len(session.messages) <= 2 and guardrails_active
                if should_cache:
                    await self.cache.set(key, json.dumps({"response": response}))
                return self._response(
                    q, response, RouteType.DATABASE, start,
                    tool=tool,
                    session_id=session.session_id,
                    guardrails_active=guardrails_active,
                    guardrails_bypassed=not guardrails_active,
                )
            route_type = RouteType.NOVA

        # 5. Nova for complex queries - uses 8-agent pipeline when available
        # The orchestrator handles:
        # - Intent classification
        # - Dynamic tool selection
        # - Parallel search + knowledge retrieval
        # - Quality enforcement and PII redaction
        history = session.get_history(max_messages=20) if session.messages else None

        # RAG context is now handled by the KnowledgeAgent in orchestrated mode
        # But we still provide fallback RAG for simple mode
        context = None
        if not self.nova.orchestrator:
            rag_result = await self.vector_index.search_customers_semantic(q, limit=5)
            if rag_result["success"] and rag_result["count"] > 0:
                customers = rag_result["data"]
                context_lines = ["Found relevant customer records:"]
                for c in customers:
                    context_lines.append(f"- {c['name']} (CRID: {c['crid']}): {c['city']}, {c['state']}, {c['move_count']} moves. Status: {c['status']}")
                context = "\n".join(context_lines)

        # Pass session_id for orchestrator context tracking
        result = await self.nova.query(
            q,
            history=history,
            context=context,
            session_id=session.session_id,
        )

        if result["success"]:
            response = result["response"]
            # Update session
            session.add_message("user", q)
            session.add_message("assistant", response)
            await self.sessions.save(session)
            # Cache only standalone queries AND when guardrails active
            should_cache = len(session.messages) <= 2 and guardrails_active
            if should_cache:
                await self.cache.set(key, json.dumps({"response": response}))
            return self._response(
                q,
                response,
                RouteType.NOVA,
                start,
                tool=result.get("tool"),
                session_id=session.session_id,
                quality_score=result.get("quality_score"),
                nova_route=result.get("route"),
                guardrails_active=guardrails_active,
                guardrails_bypassed=not guardrails_active,
                # Enhanced pipeline data
                token_usage=result.get("token_usage"),
                trace=result.get("trace"),
                model_used=result.get("model_used"),
                pagination=result.get("pagination"),
                results=result.get("results"),
            )

        return self._response(
            q, result.get("error", "Unknown error"), RouteType.NOVA, start,
            success=False,
            session_id=session.session_id,
            guardrails_active=guardrails_active,
            guardrails_bypassed=not guardrails_active,
        )

    def _format_db_result(self, result: dict, tool: str) -> str:
        match tool:
            case "lookup_crid":
                c = result["data"]
                return f"Customer {c['name']} ({c['crid']}): {c['city']}, {c['state']}. Moved {c['move_count']} times."
            case "search_customers":
                total = result["total"]
                customers = result["data"]
                lines = [f"Found {total} customers:"]
                for c in customers:
                    lines.append(f"- {c['name']} ({c['crid']}): {c['city']}, {c['state']} - {c['move_count']} moves")
                return "\n".join(lines)
            case "get_stats":
                stats = result["data"]
                sorted_stats = sorted(stats.items(), key=lambda x: -x[1])
                lines = [f"Customer count by state (Total: {result['total']}):"]
                for state, count in sorted_stats:
                    lines.append(f"- {state}: {count}")
                return "\n".join(lines)
        return json.dumps(result)

    def _response(self, query: str, response: str, route: RouteType, start: float, **kwargs) -> dict:
        """Build response dict with optional orchestrator metadata.

        Args:
            query: Original query.
            response: Response text.
            route: Route type taken.
            start: Start time for latency calculation.
            **kwargs: Additional response fields including:
                - success: Whether query succeeded.
                - cached: Whether response was from cache.
                - blocked: Whether guardrails blocked.
                - tool: Tool used (if any).
                - session_id: Session identifier.
                - quality_score: Quality score from enforcer.
                - nova_route: Sub-route within Nova.
                - guardrails_active: Whether guardrails were active.
                - guardrails_bypassed: Whether guardrails were bypassed.
                - token_usage: Token usage stats.
                - trace: Pipeline trace.
                - model_used: Nova model used.
                - pagination: Pagination info.
                - results: Raw search results.

        Returns:
            Response dict with all available metadata.
        """
        result = {
            "success": kwargs.get("success", True),
            "query": query,
            "response": response,
            "route": route.value,
            "cached": kwargs.get("cached", False),
            "blocked": kwargs.get("blocked", False),
            "tool": kwargs.get("tool"),
            "latency_ms": int((time() - start) * 1000),
            "session_id": kwargs.get("session_id"),
            # Guardrails state
            "guardrails_active": kwargs.get("guardrails_active", True),
            "guardrails_bypassed": kwargs.get("guardrails_bypassed", False),
        }

        # Add orchestrator metadata if available
        if kwargs.get("quality_score") is not None:
            result["quality_score"] = kwargs["quality_score"]
        if kwargs.get("nova_route"):
            result["nova_route"] = kwargs["nova_route"]

        # Enhanced pipeline data
        if kwargs.get("token_usage") is not None:
            token_usage = kwargs["token_usage"]
            # Convert to dict if it's a TokenUsage object
            result["token_usage"] = token_usage.to_dict() if hasattr(token_usage, "to_dict") else token_usage

        if kwargs.get("trace") is not None:
            trace = kwargs["trace"]
            # Convert to dict if it's a PipelineTrace object
            result["trace"] = trace.to_dict() if hasattr(trace, "to_dict") else trace

        if kwargs.get("model_used"):
            result["model_used"] = kwargs["model_used"]

        if kwargs.get("pagination") is not None:
            pagination = kwargs["pagination"]
            # Convert to dict if it's a PaginationInfo object
            result["pagination"] = pagination.to_dict() if hasattr(pagination, "to_dict") else pagination

        if kwargs.get("results") is not None:
            result["results"] = kwargs["results"]

        return result
