"""RAG Context Enforcer - Validates knowledge chunks are included in Nova context.

This enforcer ensures that when the KnowledgeAgent retrieves relevant chunks
with sufficient confidence, those chunks are actually included in the
context sent to Nova for response generation.

Key Gates:
- RAG_CONTEXT_INCLUDED: Knowledge chunks present when rag_confidence > threshold
- RAG_CONFIDENCE_THRESHOLD: RAG confidence meets minimum (0.3 default)
- KNOWLEDGE_CHUNK_QUALITY: Retrieved chunks have sufficient relevance
- CONTEXT_RELEVANCE_SCORE: Context maintains relevance to query
"""

from __future__ import annotations

import logging
from typing import Any

from .base_enforcer import BaseEnforcer, EnforcerResult, EnforcerGate, GateResult

logger = logging.getLogger(__name__)


class RAGContextEnforcer(BaseEnforcer):
    """Enforcer for RAG context inclusion validation.

    Validates that:
    1. Knowledge chunks ARE included in Nova context when rag_confidence > threshold
    2. Minimum chunk count is met when confidence is high
    3. Chunk quality meets relevance standards
    4. Context maintains query relevance
    """

    __slots__ = (
        "_rag_confidence_threshold",
        "_min_chunks_high_confidence",
        "_min_chunk_relevance",
    )

    # Default configuration
    DEFAULT_RAG_CONFIDENCE_THRESHOLD = 0.3
    DEFAULT_MIN_CHUNKS_HIGH_CONFIDENCE = 3
    DEFAULT_MIN_CHUNK_RELEVANCE = 0.5

    def __init__(
        self,
        enabled: bool = True,
        strict_mode: bool = False,
        rag_confidence_threshold: float = DEFAULT_RAG_CONFIDENCE_THRESHOLD,
        min_chunks_high_confidence: int = DEFAULT_MIN_CHUNKS_HIGH_CONFIDENCE,
        min_chunk_relevance: float = DEFAULT_MIN_CHUNK_RELEVANCE,
    ):
        """Initialize RAGContextEnforcer.

        Args:
            enabled: Whether enforcer is active.
            strict_mode: If True, any gate failure fails entire check.
            rag_confidence_threshold: Minimum RAG confidence to require context.
            min_chunks_high_confidence: Minimum chunks when confidence > 0.5.
            min_chunk_relevance: Minimum relevance score per chunk.
        """
        super().__init__(
            name="RAGContextEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )
        self._rag_confidence_threshold = rag_confidence_threshold
        self._min_chunks_high_confidence = min_chunks_high_confidence
        self._min_chunk_relevance = min_chunk_relevance

    def get_gates(self) -> list[EnforcerGate]:
        """Get list of gates this enforcer checks."""
        return [
            EnforcerGate.RAG_CONTEXT_INCLUDED,
            EnforcerGate.RAG_CONFIDENCE_THRESHOLD,
            EnforcerGate.KNOWLEDGE_CHUNK_QUALITY,
            EnforcerGate.CONTEXT_RELEVANCE_SCORE,
        ]

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run RAG context validation gates.

        Args:
            context: Dictionary containing:
                - knowledge_chunks: List of retrieved knowledge chunks
                - rag_confidence: Confidence score from KnowledgeAgent
                - nova_context: Context dict being sent to Nova
                - query: Original query string

        Returns:
            EnforcerResult with gate results and recommendations.
        """
        if not self._enabled:
            return EnforcerResult(
                enforcer_name=self._name,
                passed=True,
                quality_score=1.0,
            )

        gates_passed: list[GateResult] = []
        gates_failed: list[GateResult] = []

        # Extract context values
        knowledge_chunks = context.get("knowledge_chunks", [])
        rag_confidence = context.get("rag_confidence", 0.0)
        nova_context = context.get("nova_context", {})
        query = context.get("query", "")

        # Gate 1: RAG_CONFIDENCE_THRESHOLD
        gate1 = self._check_confidence_threshold(rag_confidence)
        if gate1.passed:
            gates_passed.append(gate1)
        else:
            gates_failed.append(gate1)

        # Gate 2: RAG_CONTEXT_INCLUDED (only if confidence above threshold)
        gate2 = self._check_context_included(
            knowledge_chunks, rag_confidence, nova_context
        )
        if gate2.passed:
            gates_passed.append(gate2)
        else:
            gates_failed.append(gate2)

        # Gate 3: KNOWLEDGE_CHUNK_QUALITY
        gate3 = self._check_chunk_quality(knowledge_chunks)
        if gate3.passed:
            gates_passed.append(gate3)
        else:
            gates_failed.append(gate3)

        # Gate 4: CONTEXT_RELEVANCE_SCORE
        gate4 = self._check_context_relevance(knowledge_chunks, query)
        if gate4.passed:
            gates_passed.append(gate4)
        else:
            gates_failed.append(gate4)

        return self._create_result(gates_passed, gates_failed)

    def _check_confidence_threshold(self, rag_confidence: float) -> GateResult:
        """Check if RAG confidence meets threshold."""
        passed = rag_confidence >= self._rag_confidence_threshold

        if passed:
            return self._gate_pass(
                EnforcerGate.RAG_CONFIDENCE_THRESHOLD,
                f"RAG confidence {rag_confidence:.2f} meets threshold {self._rag_confidence_threshold}",
                threshold=self._rag_confidence_threshold,
                actual_value=rag_confidence,
            )
        else:
            return self._gate_fail(
                EnforcerGate.RAG_CONFIDENCE_THRESHOLD,
                f"RAG confidence {rag_confidence:.2f} below threshold {self._rag_confidence_threshold}",
                threshold=self._rag_confidence_threshold,
                actual_value=rag_confidence,
                details={"recommendation": "Knowledge retrieval may need tuning"},
            )

    def _check_context_included(
        self,
        knowledge_chunks: list[dict[str, Any]],
        rag_confidence: float,
        nova_context: dict[str, Any],
    ) -> GateResult:
        """Check if knowledge chunks are included in Nova context when needed."""
        # If confidence is low, we don't require chunks
        if rag_confidence < self._rag_confidence_threshold:
            return self._gate_pass(
                EnforcerGate.RAG_CONTEXT_INCLUDED,
                "RAG confidence below threshold, chunks not required",
                details={"rag_confidence": rag_confidence},
            )

        # Check if chunks were retrieved
        if not knowledge_chunks:
            return self._gate_fail(
                EnforcerGate.RAG_CONTEXT_INCLUDED,
                "No knowledge chunks retrieved despite high confidence",
                details={
                    "rag_confidence": rag_confidence,
                    "chunks_count": 0,
                },
            )

        # Check if chunks are in Nova context
        knowledge_in_context = nova_context.get("knowledge", [])
        knowledge_text = nova_context.get("knowledge_context", "")

        chunks_included = len(knowledge_in_context) > 0 or len(knowledge_text) > 0

        if not chunks_included:
            return self._gate_fail(
                EnforcerGate.RAG_CONTEXT_INCLUDED,
                f"Knowledge chunks retrieved ({len(knowledge_chunks)}) but NOT included in Nova context",
                details={
                    "retrieved_chunks": len(knowledge_chunks),
                    "included_chunks": 0,
                    "rag_confidence": rag_confidence,
                    "critical": True,
                },
            )

        # Check minimum chunk count for high confidence
        min_required = self._min_chunks_high_confidence if rag_confidence > 0.5 else 1
        actual_included = len(knowledge_in_context) if knowledge_in_context else 1

        if actual_included < min_required:
            return self._gate_fail(
                EnforcerGate.RAG_CONTEXT_INCLUDED,
                f"Only {actual_included} chunks included, minimum {min_required} required",
                threshold=float(min_required),
                actual_value=float(actual_included),
                details={
                    "rag_confidence": rag_confidence,
                    "retrieved_chunks": len(knowledge_chunks),
                },
            )

        return self._gate_pass(
            EnforcerGate.RAG_CONTEXT_INCLUDED,
            f"{actual_included} knowledge chunks included in Nova context",
            details={
                "retrieved_chunks": len(knowledge_chunks),
                "included_chunks": actual_included,
                "rag_confidence": rag_confidence,
            },
        )

    def _check_chunk_quality(
        self, knowledge_chunks: list[dict[str, Any]]
    ) -> GateResult:
        """Check quality of retrieved chunks."""
        if not knowledge_chunks:
            return self._gate_pass(
                EnforcerGate.KNOWLEDGE_CHUNK_QUALITY,
                "No chunks to evaluate",
            )

        # Calculate average relevance
        relevance_scores = []
        for chunk in knowledge_chunks:
            score = chunk.get("score", chunk.get("relevance", 0.5))
            relevance_scores.append(score)

        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        high_quality_count = sum(1 for s in relevance_scores if s >= self._min_chunk_relevance)

        if avg_relevance < self._min_chunk_relevance:
            return self._gate_fail(
                EnforcerGate.KNOWLEDGE_CHUNK_QUALITY,
                f"Average chunk relevance {avg_relevance:.2f} below threshold {self._min_chunk_relevance}",
                threshold=self._min_chunk_relevance,
                actual_value=avg_relevance,
                details={
                    "chunk_count": len(knowledge_chunks),
                    "high_quality_chunks": high_quality_count,
                },
            )

        return self._gate_pass(
            EnforcerGate.KNOWLEDGE_CHUNK_QUALITY,
            f"Chunk quality good: avg relevance {avg_relevance:.2f}",
            threshold=self._min_chunk_relevance,
            actual_value=avg_relevance,
            details={
                "chunk_count": len(knowledge_chunks),
                "high_quality_chunks": high_quality_count,
            },
        )

    def _check_context_relevance(
        self,
        knowledge_chunks: list[dict[str, Any]],
        query: str,
    ) -> GateResult:
        """Check if knowledge context is relevant to query."""
        if not knowledge_chunks or not query:
            return self._gate_pass(
                EnforcerGate.CONTEXT_RELEVANCE_SCORE,
                "No context to evaluate relevance",
            )

        # Simple relevance check: do chunks contain query terms?
        query_terms = set(query.lower().split())
        relevant_chunks = 0

        for chunk in knowledge_chunks:
            chunk_text = chunk.get("content", chunk.get("text", "")).lower()
            matching_terms = sum(1 for term in query_terms if term in chunk_text)
            if matching_terms > 0:
                relevant_chunks += 1

        relevance_ratio = relevant_chunks / len(knowledge_chunks)

        if relevance_ratio < 0.3:  # At least 30% of chunks should be relevant
            return self._gate_fail(
                EnforcerGate.CONTEXT_RELEVANCE_SCORE,
                f"Only {relevant_chunks}/{len(knowledge_chunks)} chunks are query-relevant",
                threshold=0.3,
                actual_value=relevance_ratio,
                details={
                    "query_terms": list(query_terms)[:10],
                    "relevant_chunks": relevant_chunks,
                },
            )

        return self._gate_pass(
            EnforcerGate.CONTEXT_RELEVANCE_SCORE,
            f"Context relevance good: {relevant_chunks}/{len(knowledge_chunks)} chunks relevant",
            threshold=0.3,
            actual_value=relevance_ratio,
            details={
                "relevant_chunks": relevant_chunks,
                "total_chunks": len(knowledge_chunks),
            },
        )
