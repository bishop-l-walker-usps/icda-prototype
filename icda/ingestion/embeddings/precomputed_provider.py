"""Precomputed embedding provider.

Pass-through provider for embeddings that are pre-computed
by external systems (e.g., C library processing NCOA data).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from icda.ingestion.embeddings.base_provider import (
    BaseEmbeddingProvider,
    EmbeddingResult,
    ProviderStatus,
)

logger = logging.getLogger(__name__)


class PrecomputedEmbeddingProvider(BaseEmbeddingProvider):
    """Provider for pre-computed embeddings.

    This provider doesn't generate embeddings - it validates and
    passes through embeddings that were pre-computed by external
    systems (e.g., C library processing NCOA data).

    Features:
    - Validates embedding dimensions
    - Applies optional L2 normalization
    - Tracks statistics for monitoring
    """

    __slots__ = ("_expected_dimension", "_strict_dimension")

    def __init__(
        self,
        expected_dimension: int = 1024,
        normalize: bool = True,
        strict_dimension: bool = False,
    ):
        """Initialize precomputed provider.

        Args:
            expected_dimension: Expected embedding dimension.
            normalize: Whether to L2 normalize embeddings.
            strict_dimension: If True, reject embeddings with wrong dimension.
        """
        super().__init__(
            provider_name="precomputed",
            model_name="external",
            dimension=expected_dimension,
            normalize=normalize,
        )
        self._expected_dimension = expected_dimension
        self._strict_dimension = strict_dimension

    async def initialize(self) -> bool:
        """Initialize provider - always succeeds for precomputed.

        Returns:
            True always.
        """
        self._status = ProviderStatus.AVAILABLE
        logger.info(
            f"PrecomputedEmbeddingProvider initialized "
            f"(expected_dim={self._expected_dimension}, normalize={self._normalize})"
        )
        return True

    async def embed(self, text: str) -> EmbeddingResult | None:
        """Not supported for precomputed provider.

        Use embed_precomputed() instead.

        Args:
            text: Input text (ignored).

        Returns:
            None - use embed_precomputed() instead.
        """
        logger.warning(
            "PrecomputedEmbeddingProvider.embed() called - "
            "use embed_precomputed() for pre-computed embeddings"
        )
        return None

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[EmbeddingResult]:
        """Not supported for precomputed provider.

        Use embed_precomputed_batch() instead.

        Args:
            texts: List of texts (ignored).
            batch_size: Batch size (ignored).

        Returns:
            Empty list - use embed_precomputed_batch() instead.
        """
        logger.warning(
            "PrecomputedEmbeddingProvider.embed_batch() called - "
            "use embed_precomputed_batch() for pre-computed embeddings"
        )
        return []

    async def embed_precomputed(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingResult | None:
        """Process a pre-computed embedding.

        Validates dimension and optionally normalizes.

        Args:
            text: Original text the embedding represents.
            embedding: Pre-computed embedding vector.
            metadata: Optional metadata about the embedding.

        Returns:
            EmbeddingResult or None if validation fails.
        """
        start = time.time()

        # Validate embedding
        if not embedding or len(embedding) == 0:
            self._stats.record_failure("Empty embedding")
            return None

        # Check dimension
        actual_dim = len(embedding)
        if self._strict_dimension and actual_dim != self._expected_dimension:
            self._stats.record_failure(
                f"Dimension mismatch: expected {self._expected_dimension}, got {actual_dim}"
            )
            logger.warning(
                f"Precomputed embedding dimension mismatch: "
                f"expected {self._expected_dimension}, got {actual_dim}"
            )
            return None

        # Apply normalization if needed
        normalized = False
        if self._normalize:
            embedding = self._l2_normalize(embedding)
            normalized = True

        latency_ms = self._time_ms(start)
        self._stats.record_success(latency_ms)

        result = EmbeddingResult(
            text=text,
            embedding=embedding,
            dimension=actual_dim,
            provider="precomputed",
            normalized=normalized,
            latency_ms=latency_ms,
            model="external",
            metadata=metadata or {},
        )

        return result

    async def embed_precomputed_batch(
        self,
        items: list[tuple[str, list[float]]],
    ) -> list[EmbeddingResult | None]:
        """Process batch of pre-computed embeddings.

        Args:
            items: List of (text, embedding) tuples.

        Returns:
            List of EmbeddingResults (None for failures).
        """
        results: list[EmbeddingResult | None] = []

        for text, embedding in items:
            result = await self.embed_precomputed(text, embedding)
            results.append(result)

        return results

    async def health_check(self) -> bool:
        """Health check - always healthy for precomputed.

        Returns:
            True always.
        """
        return True

    def validate_embedding(self, embedding: list[float]) -> tuple[bool, str]:
        """Validate an embedding vector.

        Args:
            embedding: Embedding to validate.

        Returns:
            Tuple of (is_valid, message).
        """
        if not embedding:
            return False, "Embedding is empty"

        actual_dim = len(embedding)

        if self._strict_dimension and actual_dim != self._expected_dimension:
            return False, f"Dimension {actual_dim} != expected {self._expected_dimension}"

        # Check for NaN/Inf values
        for i, val in enumerate(embedding):
            if val != val:  # NaN check
                return False, f"NaN value at index {i}"
            if abs(val) == float("inf"):
                return False, f"Inf value at index {i}"

        return True, "Valid"
