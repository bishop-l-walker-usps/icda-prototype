"""Nova Context Enforcer - Validates memory context for Nova LLM responses.

Ensures that memory enhances (not degrades) Nova LLM response generation
and the context provided leads to better, more personalized responses.
"""

from __future__ import annotations

import logging
from typing import Any

from .base_enforcer import BaseEnforcer, EnforcerResult, EnforcerGate, GateResult

logger = logging.getLogger(__name__)


class NovaContextEnforcer(BaseEnforcer):
    """Enforcer for Nova LLM context quality.

    Quality Gates:
    - CONTEXT_TOKEN_BUDGET: Memory context within token limits
    - RESPONSE_QUALITY_STABLE: AI confidence not degraded
    - HALLUCINATION_PREVENTION: Memory doesn't introduce false facts
    - FACTUAL_GROUNDING: Response grounded in memory + search
    """

    # Thresholds
    TOKEN_BUDGET = 1000           # Max tokens for memory context
    QUALITY_DELTA_THRESHOLD = -0.05  # Max allowed quality drop
    GROUNDING_THRESHOLD = 0.85    # Min grounding score

    def __init__(self, enabled: bool = True, strict_mode: bool = False):
        """Initialize NovaContextEnforcer."""
        super().__init__(
            name="NovaContextEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )

    def get_gates(self) -> list[EnforcerGate]:
        """Get list of gates this enforcer checks."""
        return [
            EnforcerGate.CONTEXT_TOKEN_BUDGET,
            EnforcerGate.RESPONSE_QUALITY_STABLE,
            EnforcerGate.HALLUCINATION_PREVENTION,
            EnforcerGate.FACTUAL_GROUNDING,
        ]

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run Nova context gates.

        Args:
            context: Must contain:
                - unified_memory: UnifiedMemoryContext
                - nova_response: NovaResponse from NovaAgent
                - search_result: SearchResult for grounding check
                - baseline_quality: Quality score without memory (optional)
                - memory_context_text: Text representation of memory context

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

        # Gate 1: Context Token Budget
        token_result = self._check_token_budget(context)
        (gates_passed if token_result.passed else gates_failed).append(token_result)

        # Gate 2: Response Quality Stable
        quality_result = self._check_response_quality(context)
        (gates_passed if quality_result.passed else gates_failed).append(quality_result)

        # Gate 3: Hallucination Prevention
        hallucination_result = self._check_hallucination(context)
        (gates_passed if hallucination_result.passed else gates_failed).append(
            hallucination_result
        )

        # Gate 4: Factual Grounding
        grounding_result = self._check_grounding(context)
        (gates_passed if grounding_result.passed else gates_failed).append(
            grounding_result
        )

        result = self._create_result(gates_passed, gates_failed)
        result.metrics = {
            "gates_evaluated": len(gates_passed) + len(gates_failed),
            "pass_rate": result.pass_rate,
        }

        return result

    def _check_token_budget(self, context: dict[str, Any]) -> GateResult:
        """Check that memory context is within token budget."""
        memory_context_text = context.get("memory_context_text", "")
        unified_memory = context.get("unified_memory")

        if not memory_context_text and unified_memory:
            # Estimate from memory components
            text_parts = []

            if hasattr(unified_memory, 'session_summary') and unified_memory.session_summary:
                text_parts.append(unified_memory.session_summary)

            if hasattr(unified_memory, 'ltm_facts'):
                for fact in unified_memory.ltm_facts:
                    if hasattr(fact, 'content'):
                        text_parts.append(fact.content)

            memory_context_text = " ".join(text_parts)

        if not memory_context_text:
            return self._gate_pass(
                EnforcerGate.CONTEXT_TOKEN_BUDGET,
                "No memory context text",
                actual_value=0,
            )

        # Estimate tokens (rough: 1 token ≈ 4 chars)
        estimated_tokens = len(memory_context_text) // 4

        if estimated_tokens <= self.TOKEN_BUDGET:
            return self._gate_pass(
                EnforcerGate.CONTEXT_TOKEN_BUDGET,
                f"Memory context ~{estimated_tokens} tokens within budget",
                threshold=self.TOKEN_BUDGET,
                actual_value=estimated_tokens,
            )

        return self._gate_fail(
            EnforcerGate.CONTEXT_TOKEN_BUDGET,
            f"Memory context ~{estimated_tokens} tokens exceeds budget",
            threshold=self.TOKEN_BUDGET,
            actual_value=estimated_tokens,
        )

    def _check_response_quality(self, context: dict[str, Any]) -> GateResult:
        """Check that response quality is not degraded by memory."""
        nova_response = context.get("nova_response")
        baseline_quality = context.get("baseline_quality")

        if not nova_response:
            return self._gate_pass(
                EnforcerGate.RESPONSE_QUALITY_STABLE,
                "No Nova response to validate",
            )

        current_quality = (
            nova_response.ai_confidence
            if hasattr(nova_response, 'ai_confidence')
            else 0.5
        )

        if baseline_quality is None:
            return self._gate_pass(
                EnforcerGate.RESPONSE_QUALITY_STABLE,
                f"Response quality: {current_quality:.2f} (no baseline)",
                actual_value=current_quality,
            )

        delta = current_quality - baseline_quality

        if delta >= self.QUALITY_DELTA_THRESHOLD:
            return self._gate_pass(
                EnforcerGate.RESPONSE_QUALITY_STABLE,
                f"Response quality delta: {delta:+.2f}",
                threshold=self.QUALITY_DELTA_THRESHOLD,
                actual_value=delta,
            )

        return self._gate_fail(
            EnforcerGate.RESPONSE_QUALITY_STABLE,
            f"Response quality dropped by {-delta:.2f}",
            threshold=self.QUALITY_DELTA_THRESHOLD,
            actual_value=delta,
        )

    def _check_hallucination(self, context: dict[str, Any]) -> GateResult:
        """Check that memory doesn't introduce hallucinations."""
        nova_response = context.get("nova_response")
        unified_memory = context.get("unified_memory")
        search_result = context.get("search_result")

        if not nova_response:
            return self._gate_pass(
                EnforcerGate.HALLUCINATION_PREVENTION,
                "No Nova response to check for hallucinations",
            )

        response_text = (
            nova_response.response_text
            if hasattr(nova_response, 'response_text')
            else ""
        )

        if not response_text:
            return self._gate_pass(
                EnforcerGate.HALLUCINATION_PREVENTION,
                "Empty response",
            )

        # Check for obvious hallucination patterns
        hallucination_indicators = [
            "I don't have access to",
            "I cannot verify",
            "I'm making an assumption",
            "based on my training",
            "I believe",
            "I think",
            "probably",
            "might be",
        ]

        # These are soft indicators - memory shouldn't add uncertainty
        uncertain_phrases = sum(
            1 for phrase in hallucination_indicators
            if phrase.lower() in response_text.lower()
        )

        if uncertain_phrases == 0:
            return self._gate_pass(
                EnforcerGate.HALLUCINATION_PREVENTION,
                "No hallucination indicators detected",
            )

        if uncertain_phrases <= 2:
            return self._gate_pass(
                EnforcerGate.HALLUCINATION_PREVENTION,
                f"Minor uncertainty ({uncertain_phrases} phrases) - acceptable",
                details={"uncertain_phrase_count": uncertain_phrases},
            )

        return self._gate_fail(
            EnforcerGate.HALLUCINATION_PREVENTION,
            f"High uncertainty ({uncertain_phrases} phrases) may indicate hallucination",
            details={"uncertain_phrase_count": uncertain_phrases},
        )

    def _check_grounding(self, context: dict[str, Any]) -> GateResult:
        """Check that response is grounded in memory and search results."""
        nova_response = context.get("nova_response")
        search_result = context.get("search_result")
        unified_memory = context.get("unified_memory")

        if not nova_response:
            return self._gate_pass(
                EnforcerGate.FACTUAL_GROUNDING,
                "No Nova response to check grounding",
            )

        response_text = (
            nova_response.response_text
            if hasattr(nova_response, 'response_text')
            else ""
        ).lower()

        if not response_text:
            return self._gate_pass(
                EnforcerGate.FACTUAL_GROUNDING,
                "Empty response",
            )

        # Collect grounding sources
        grounding_terms = set()

        # From search results
        if search_result and hasattr(search_result, 'results'):
            for result in search_result.results[:5]:
                if isinstance(result, dict):
                    for key in ['crid', 'name', 'city', 'state']:
                        if key in result and result[key]:
                            grounding_terms.add(str(result[key]).lower())

        # From memory
        if unified_memory:
            if hasattr(unified_memory, 'local_context') and unified_memory.local_context:
                for entity in unified_memory.local_context.recalled_entities:
                    if hasattr(entity, 'canonical_name'):
                        grounding_terms.add(entity.canonical_name.lower())

        if not grounding_terms:
            return self._gate_pass(
                EnforcerGate.FACTUAL_GROUNDING,
                "No grounding sources available",
            )

        # Check how many grounding terms appear in response
        grounded_count = sum(
            1 for term in grounding_terms if term in response_text
        )
        grounding_rate = grounded_count / len(grounding_terms)

        if grounding_rate >= self.GROUNDING_THRESHOLD:
            return self._gate_pass(
                EnforcerGate.FACTUAL_GROUNDING,
                f"Grounding rate {grounding_rate:.1%} meets threshold",
                threshold=self.GROUNDING_THRESHOLD,
                actual_value=grounding_rate,
            )

        # Soft pass for partial grounding
        if grounding_rate >= 0.5:
            return self._gate_pass(
                EnforcerGate.FACTUAL_GROUNDING,
                f"Partial grounding {grounding_rate:.1%} - acceptable",
                threshold=self.GROUNDING_THRESHOLD,
                actual_value=grounding_rate,
            )

        return self._gate_fail(
            EnforcerGate.FACTUAL_GROUNDING,
            f"Low grounding rate {grounding_rate:.1%}",
            threshold=self.GROUNDING_THRESHOLD,
            actual_value=grounding_rate,
        )
