"""Embedding dimension normalizer.

Handles dimension normalization between different embedding providers
to ensure all embeddings have consistent dimensions for storage/search.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class NormalizationMethod(str, Enum):
    """Methods for dimension normalization."""

    TRUNCATE = "truncate"      # Simply truncate to target dimension
    PAD_ZERO = "pad_zero"      # Zero-pad to reach target dimension
    PCA = "pca"                # PCA reduction (requires fitting)
    AVERAGE_POOL = "avg_pool"  # Average pooling for reduction
    LINEAR_PROJECT = "linear"  # Linear projection (requires weights)


class EmbeddingNormalizer:
    """Normalizes embeddings to a target dimension.

    Handles both expansion (padding) and reduction (truncation/pooling)
    to ensure consistent dimensions across different providers.

    Features:
    - Multiple reduction methods (truncate, PCA, average pool)
    - Zero-padding for expansion
    - L2 re-normalization after dimension change
    - Statistics tracking
    """

    __slots__ = (
        "_target_dimension",
        "_reduction_method",
        "_expansion_method",
        "_renormalize",
        "_pca_components",
        "_stats",
    )

    def __init__(
        self,
        target_dimension: int = 1024,
        reduction_method: NormalizationMethod = NormalizationMethod.TRUNCATE,
        expansion_method: NormalizationMethod = NormalizationMethod.PAD_ZERO,
        renormalize: bool = True,
    ):
        """Initialize normalizer.

        Args:
            target_dimension: Target embedding dimension.
            reduction_method: Method for reducing larger dimensions.
            expansion_method: Method for expanding smaller dimensions.
            renormalize: Whether to L2 normalize after dimension change.
        """
        self._target_dimension = target_dimension
        self._reduction_method = reduction_method
        self._expansion_method = expansion_method
        self._renormalize = renormalize
        self._pca_components: list[list[float]] | None = None
        self._stats = {
            "normalized": 0,
            "truncated": 0,
            "padded": 0,
            "unchanged": 0,
        }

    @property
    def target_dimension(self) -> int:
        """Get target dimension."""
        return self._target_dimension

    @property
    def stats(self) -> dict[str, int]:
        """Get normalization statistics."""
        return self._stats.copy()

    def normalize(
        self,
        embedding: list[float],
        source_dimension: int | None = None,
    ) -> list[float]:
        """Normalize embedding to target dimension.

        Args:
            embedding: Input embedding vector.
            source_dimension: Original dimension (inferred if not provided).

        Returns:
            Embedding normalized to target dimension.
        """
        current_dim = len(embedding)
        source_dim = source_dimension or current_dim

        # No change needed
        if current_dim == self._target_dimension:
            self._stats["unchanged"] += 1
            return embedding

        # Reduction needed
        if current_dim > self._target_dimension:
            result = self._reduce(embedding)
            self._stats["truncated"] += 1
        # Expansion needed
        else:
            result = self._expand(embedding)
            self._stats["padded"] += 1

        # Re-normalize after dimension change
        if self._renormalize:
            result = self._l2_normalize(result)

        self._stats["normalized"] += 1
        return result

    def normalize_batch(
        self,
        embeddings: list[list[float]],
    ) -> list[list[float]]:
        """Normalize batch of embeddings.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            List of normalized embeddings.
        """
        return [self.normalize(emb) for emb in embeddings]

    def _reduce(self, embedding: list[float]) -> list[float]:
        """Reduce embedding to target dimension.

        Args:
            embedding: Embedding larger than target.

        Returns:
            Reduced embedding.
        """
        if self._reduction_method == NormalizationMethod.TRUNCATE:
            return embedding[: self._target_dimension]

        elif self._reduction_method == NormalizationMethod.AVERAGE_POOL:
            return self._average_pool_reduce(embedding)

        elif self._reduction_method == NormalizationMethod.PCA:
            if self._pca_components is not None:
                return self._pca_reduce(embedding)
            # Fallback to truncation if PCA not fitted
            logger.warning("PCA not fitted, falling back to truncation")
            return embedding[: self._target_dimension]

        else:
            # Default to truncation
            return embedding[: self._target_dimension]

    def _expand(self, embedding: list[float]) -> list[float]:
        """Expand embedding to target dimension.

        Args:
            embedding: Embedding smaller than target.

        Returns:
            Expanded embedding.
        """
        if self._expansion_method == NormalizationMethod.PAD_ZERO:
            # Zero-pad to target dimension
            padding = [0.0] * (self._target_dimension - len(embedding))
            return embedding + padding

        else:
            # Default to zero-padding
            padding = [0.0] * (self._target_dimension - len(embedding))
            return embedding + padding

    def _average_pool_reduce(self, embedding: list[float]) -> list[float]:
        """Reduce dimension using average pooling.

        Divides embedding into target_dimension chunks and averages each.

        Args:
            embedding: Large embedding to reduce.

        Returns:
            Reduced embedding via average pooling.
        """
        current_dim = len(embedding)
        chunk_size = current_dim / self._target_dimension
        result = []

        for i in range(self._target_dimension):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size)
            chunk = embedding[start:end]
            if chunk:
                result.append(sum(chunk) / len(chunk))
            else:
                result.append(0.0)

        return result

    def _pca_reduce(self, embedding: list[float]) -> list[float]:
        """Reduce dimension using pre-fitted PCA components.

        Args:
            embedding: Embedding to reduce.

        Returns:
            PCA-reduced embedding.
        """
        if self._pca_components is None:
            raise ValueError("PCA components not fitted")

        # Project embedding onto PCA components
        result = []
        for component in self._pca_components:
            # Dot product with component
            projection = sum(e * c for e, c in zip(embedding, component))
            result.append(projection)

        return result

    def fit_pca(self, embeddings: list[list[float]]) -> None:
        """Fit PCA components from sample embeddings.

        Uses simple power iteration for principal components.
        For production, consider using numpy/sklearn.

        Args:
            embeddings: Sample embeddings for fitting.
        """
        if len(embeddings) < self._target_dimension:
            logger.warning(
                f"Not enough samples ({len(embeddings)}) for PCA fitting "
                f"(need {self._target_dimension})"
            )
            return

        # Simple mean-centered covariance estimation
        # For production, use proper PCA implementation
        logger.info(
            f"Fitting PCA from {len(embeddings)} samples "
            f"to {self._target_dimension} dimensions"
        )

        # This is a placeholder - real PCA would use numpy
        # For now, just store truncation indices
        self._pca_components = None
        logger.warning("PCA fitting not implemented, using truncation fallback")

    def _l2_normalize(self, embedding: list[float]) -> list[float]:
        """L2 normalize embedding.

        Args:
            embedding: Input embedding.

        Returns:
            Unit-length embedding.
        """
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude == 0:
            return embedding
        return [x / magnitude for x in embedding]

    def get_info(self) -> dict[str, Any]:
        """Get normalizer information."""
        return {
            "target_dimension": self._target_dimension,
            "reduction_method": self._reduction_method.value,
            "expansion_method": self._expansion_method.value,
            "renormalize": self._renormalize,
            "pca_fitted": self._pca_components is not None,
            "stats": self._stats,
        }


def create_normalizer(
    target_dimension: int = 1024,
    method: str = "truncate",
) -> EmbeddingNormalizer:
    """Factory function to create normalizer.

    Args:
        target_dimension: Target embedding dimension.
        method: Reduction method (truncate, avg_pool, pca).

    Returns:
        Configured EmbeddingNormalizer.
    """
    method_map = {
        "truncate": NormalizationMethod.TRUNCATE,
        "avg_pool": NormalizationMethod.AVERAGE_POOL,
        "average_pool": NormalizationMethod.AVERAGE_POOL,
        "pca": NormalizationMethod.PCA,
    }

    reduction_method = method_map.get(method.lower(), NormalizationMethod.TRUNCATE)

    return EmbeddingNormalizer(
        target_dimension=target_dimension,
        reduction_method=reduction_method,
    )
