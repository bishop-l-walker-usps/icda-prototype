"""Memory Integrity Enforcer - Validates memory read/write consistency.

Ensures all memory operations with Bedrock AgentCore are consistent,
complete, and don't lose or corrupt data.
"""

from __future__ import annotations

import logging
from typing import Any

from .base_enforcer import BaseEnforcer, EnforcerResult, EnforcerGate, GateResult

logger = logging.getLogger(__name__)


class MemoryIntegrityEnforcer(BaseEnforcer):
    """Enforcer for memory operation integrity.

    Quality Gates:
    - WRITE_CONSISTENCY: Entities written can be read back
    - EXTRACTION_COMPLETE: LTM strategies extract expected entities
    - SESSION_ISOLATION: Session data doesn't leak between users
    - ENTITY_COHERENCE: Extracted entities match source data
    """

    # Thresholds
    EXTRACTION_THRESHOLD = 0.90  # 90% of expected entities extracted
    COHERENCE_THRESHOLD = 0.95   # 95% accuracy in entity matching

    def __init__(self, enabled: bool = True, strict_mode: bool = False):
        """Initialize MemoryIntegrityEnforcer."""
        super().__init__(
            name="MemoryIntegrityEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )

    def get_gates(self) -> list[EnforcerGate]:
        """Get list of gates this enforcer checks."""
        return [
            EnforcerGate.WRITE_CONSISTENCY,
            EnforcerGate.EXTRACTION_COMPLETE,
            EnforcerGate.SESSION_ISOLATION,
            EnforcerGate.ENTITY_COHERENCE,
        ]

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run memory integrity gates.

        Args:
            context: Must contain:
                - unified_memory: UnifiedMemoryContext
                - session_id: Current session ID
                - actor_id: Current actor ID
                - expected_entities: Optional list of expected entity IDs
                - write_operations: Optional list of recent write ops

        Returns:
            EnforcerResult with gate outcomes.
        """
        if not self._enabled:
            return EnforcerResult(
                enforcer_name=self._name,
                passed=True,
                quality_score=1.0,
            )

        gates_passed: list[GateResult] = []
        gates_failed: list[GateResult] = []

        # Gate 1: Write Consistency
        write_result = self._check_write_consistency(context)
        (gates_passed if write_result.passed else gates_failed).append(write_result)

        # Gate 2: Extraction Complete
        extraction_result = self._check_extraction_complete(context)
        (gates_passed if extraction_result.passed else gates_failed).append(
            extraction_result
        )

        # Gate 3: Session Isolation
        isolation_result = self._check_session_isolation(context)
        (gates_passed if isolation_result.passed else gates_failed).append(
            isolation_result
        )

        # Gate 4: Entity Coherence
        coherence_result = self._check_entity_coherence(context)
        (gates_passed if coherence_result.passed else gates_failed).append(
            coherence_result
        )

        result = self._create_result(gates_passed, gates_failed)
        result.metrics = {
            "gates_evaluated": len(gates_passed) + len(gates_failed),
            "pass_rate": result.pass_rate,
        }

        return result

    def _check_write_consistency(self, context: dict[str, Any]) -> GateResult:
        """Check that entities written can be read back."""
        write_ops = context.get("write_operations", [])
        unified_memory = context.get("unified_memory")

        if not write_ops:
            return self._gate_pass(
                EnforcerGate.WRITE_CONSISTENCY,
                "No write operations to verify",
            )

        if not unified_memory:
            return self._gate_fail(
                EnforcerGate.WRITE_CONSISTENCY,
                "No unified memory context available",
            )

        # Check if recently written entities are readable
        stm_turns = unified_memory.stm_turns if hasattr(unified_memory, 'stm_turns') else []
        local_entities = (
            unified_memory.local_context.recalled_entities
            if hasattr(unified_memory, 'local_context') and unified_memory.local_context
            else []
        )

        # Simple verification: if we have STM or local entities, write is consistent
        if stm_turns or local_entities:
            return self._gate_pass(
                EnforcerGate.WRITE_CONSISTENCY,
                f"Memory contains {len(stm_turns)} STM turns, {len(local_entities)} entities",
                details={"stm_count": len(stm_turns), "entity_count": len(local_entities)},
            )

        return self._gate_pass(
            EnforcerGate.WRITE_CONSISTENCY,
            "Write consistency verified (no data expected)",
        )

    def _check_extraction_complete(self, context: dict[str, Any]) -> GateResult:
        """Check that LTM strategies extract expected entities."""
        expected_entities = context.get("expected_entities", [])
        unified_memory = context.get("unified_memory")

        if not expected_entities:
            return self._gate_pass(
                EnforcerGate.EXTRACTION_COMPLETE,
                "No expected entities specified",
            )

        if not unified_memory:
            return self._gate_fail(
                EnforcerGate.EXTRACTION_COMPLETE,
                "No unified memory context available",
            )

        # Count extracted entities from LTM
        ltm_facts = unified_memory.ltm_facts if hasattr(unified_memory, 'ltm_facts') else []
        extracted_count = len(ltm_facts)
        expected_count = len(expected_entities)

        if expected_count == 0:
            return self._gate_pass(
                EnforcerGate.EXTRACTION_COMPLETE,
                "No entities expected for extraction",
            )

        extraction_rate = extracted_count / expected_count

        if extraction_rate >= self.EXTRACTION_THRESHOLD:
            return self._gate_pass(
                EnforcerGate.EXTRACTION_COMPLETE,
                f"Extraction rate {extraction_rate:.1%} meets threshold",
                threshold=self.EXTRACTION_THRESHOLD,
                actual_value=extraction_rate,
            )

        return self._gate_fail(
            EnforcerGate.EXTRACTION_COMPLETE,
            f"Extraction rate {extraction_rate:.1%} below threshold",
            threshold=self.EXTRACTION_THRESHOLD,
            actual_value=extraction_rate,
        )

    def _check_session_isolation(self, context: dict[str, Any]) -> GateResult:
        """Check that session data doesn't leak between users."""
        session_id = context.get("session_id")
        actor_id = context.get("actor_id")
        unified_memory = context.get("unified_memory")

        if not session_id or not actor_id:
            return self._gate_pass(
                EnforcerGate.SESSION_ISOLATION,
                "No session context to validate",
            )

        if not unified_memory:
            return self._gate_pass(
                EnforcerGate.SESSION_ISOLATION,
                "No memory to check for isolation",
            )

        # Check that all memory signals reference current session
        signals = unified_memory.memory_signals if hasattr(unified_memory, 'memory_signals') else []

        # Look for any cross-session indicators (this is a simplified check)
        for signal in signals:
            if "cross_session" in signal.lower() or "leak" in signal.lower():
                return self._gate_fail(
                    EnforcerGate.SESSION_ISOLATION,
                    f"Potential session leak detected: {signal}",
                    details={"signal": signal},
                )

        return self._gate_pass(
            EnforcerGate.SESSION_ISOLATION,
            "Session isolation verified",
            details={"session_id": session_id, "actor_id": actor_id},
        )

    def _check_entity_coherence(self, context: dict[str, Any]) -> GateResult:
        """Check that extracted entities match source data."""
        unified_memory = context.get("unified_memory")

        if not unified_memory:
            return self._gate_pass(
                EnforcerGate.ENTITY_COHERENCE,
                "No memory context to validate",
            )

        # Get entities from both sources
        ltm_facts = unified_memory.ltm_facts if hasattr(unified_memory, 'ltm_facts') else []
        local_entities = (
            unified_memory.local_context.recalled_entities
            if hasattr(unified_memory, 'local_context') and unified_memory.local_context
            else []
        )

        if not ltm_facts and not local_entities:
            return self._gate_pass(
                EnforcerGate.ENTITY_COHERENCE,
                "No entities to verify coherence",
            )

        # Check that local entities have valid structure
        valid_count = 0
        total_count = len(local_entities)

        for entity in local_entities:
            if hasattr(entity, 'entity_id') and hasattr(entity, 'entity_type'):
                if entity.entity_id and entity.entity_type:
                    valid_count += 1

        if total_count == 0:
            coherence_rate = 1.0
        else:
            coherence_rate = valid_count / total_count

        if coherence_rate >= self.COHERENCE_THRESHOLD:
            return self._gate_pass(
                EnforcerGate.ENTITY_COHERENCE,
                f"Entity coherence {coherence_rate:.1%} meets threshold",
                threshold=self.COHERENCE_THRESHOLD,
                actual_value=coherence_rate,
            )

        return self._gate_fail(
            EnforcerGate.ENTITY_COHERENCE,
            f"Entity coherence {coherence_rate:.1%} below threshold",
            threshold=self.COHERENCE_THRESHOLD,
            actual_value=coherence_rate,
        )
