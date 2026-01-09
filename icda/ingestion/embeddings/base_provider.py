"""Base embedding provider interface.

Abstract base class for all embedding providers with consistent
interface for single and batch embedding generation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ProviderStatus(str, Enum):
    """Status of an embedding provider."""

    AVAILABLE = "available"        # Ready to use
    UNAVAILABLE = "unavailable"    # Cannot connect/authenticate
    RATE_LIMITED = "rate_limited"  # Temporarily unavailable
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker tripped
    INITIALIZING = "initializing"  # Starting up


@dataclass(slots=True)
class EmbeddingResult:
    """Result of embedding generation.

    Attributes:
        text: Original input text.
        embedding: Generated embedding vector.
        dimension: Dimension of the embedding.
        provider: Provider that generated the embedding.
        normalized: Whether embedding has been L2 normalized.
        latency_ms: Time taken to generate in milliseconds.
        model: Model name/ID used.
        metadata: Additional provider-specific metadata.
    """

    text: str
    embedding: list[float]
    dimension: int
    provider: str
    normalized: bool = False
    latency_ms: int = 0
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if embedding is valid."""
        return (
            self.embedding is not None
            and len(self.embedding) > 0
            and len(self.embedding) == self.dimension
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes embedding for size)."""
        return {
            "text_length": len(self.text),
            "dimension": self.dimension,
            "provider": self.provider,
            "normalized": self.normalized,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "is_valid": self.is_valid,
        }


@dataclass(slots=True)
class ProviderStats:
    """Statistics for embedding provider operations."""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0
    last_call_time: str | None = None
    last_error: str | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.calls == 0:
            return 1.0
        return self.successes / self.calls

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successes == 0:
            return 0.0
        return self.total_latency_ms / self.successes

    def record_success(self, latency_ms: int, tokens: int = 0) -> None:
        """Record a successful call."""
        self.calls += 1
        self.successes += 1
        self.total_latency_ms += latency_ms
        self.total_tokens += tokens
        self.last_call_time = datetime.utcnow().isoformat()

    def record_failure(self, error: str) -> None:
        """Record a failed call."""
        self.calls += 1
        self.failures += 1
        self.last_error = error
        self.last_call_time = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate * 100, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "last_call_time": self.last_call_time,
            "last_error": self.last_error,
        }


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding providers must implement:
    - initialize(): Set up client connection
    - embed(): Generate embedding for single text
    - embed_batch(): Generate embeddings for multiple texts
    - health_check(): Verify provider is operational

    Providers should:
    - Report their native dimension
    - Track statistics via _stats
    - Handle rate limiting gracefully
    - Support L2 normalization
    """

    __slots__ = (
        "_provider_name",
        "_model_name",
        "_dimension",
        "_status",
        "_stats",
        "_normalize",
    )

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        dimension: int,
        normalize: bool = True,
    ):
        """Initialize provider.

        Args:
            provider_name: Provider identifier (titan, openai, etc.).
            model_name: Model name/ID.
            dimension: Native embedding dimension.
            normalize: Whether to L2 normalize embeddings.
        """
        self._provider_name = provider_name
        self._model_name = model_name
        self._dimension = dimension
        self._status = ProviderStatus.INITIALIZING
        self._stats = ProviderStats()
        self._normalize = normalize

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._provider_name

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Get native embedding dimension."""
        return self._dimension

    @property
    def status(self) -> ProviderStatus:
        """Get current provider status."""
        return self._status

    @property
    def available(self) -> bool:
        """Check if provider is available."""
        return self._status == ProviderStatus.AVAILABLE

    @property
    def stats(self) -> ProviderStats:
        """Get provider statistics."""
        return self._stats

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize provider connection.

        Should:
        - Set up client/connection
        - Verify credentials
        - Update _status

        Returns:
            True if initialization successful.
        """
        pass

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult | None:
        """Generate embedding for single text.

        Args:
            text: Input text to embed.

        Returns:
            EmbeddingResult or None if failed.
        """
        pass

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.
            batch_size: Maximum texts per API call.

        Returns:
            List of EmbeddingResults (may contain None for failures).
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is operational.

        Returns:
            True if provider is healthy.
        """
        pass

    def _l2_normalize(self, embedding: list[float]) -> list[float]:
        """L2 normalize an embedding vector.

        Args:
            embedding: Raw embedding vector.

        Returns:
            Normalized embedding with unit length.
        """
        import math

        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude == 0:
            return embedding
        return [x / magnitude for x in embedding]

    def _create_result(
        self,
        text: str,
        embedding: list[float],
        latency_ms: int,
        normalized: bool = False,
    ) -> EmbeddingResult:
        """Create an EmbeddingResult.

        If self._normalize is True and embedding not already normalized,
        applies L2 normalization.

        Args:
            text: Original input text.
            embedding: Generated embedding.
            latency_ms: Generation time.
            normalized: Whether embedding is already normalized.

        Returns:
            EmbeddingResult with optional normalization applied.
        """
        if self._normalize and not normalized:
            embedding = self._l2_normalize(embedding)
            normalized = True

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            dimension=len(embedding),
            provider=self._provider_name,
            normalized=normalized,
            latency_ms=latency_ms,
            model=self._model_name,
        )

    def _time_ms(self, start: float) -> int:
        """Calculate elapsed time in milliseconds."""
        return int((time.time() - start) * 1000)

    def get_info(self) -> dict[str, Any]:
        """Get provider information."""
        return {
            "provider": self._provider_name,
            "model": self._model_name,
            "dimension": self._dimension,
            "status": self._status.value,
            "available": self.available,
            "normalize": self._normalize,
            "stats": self._stats.to_dict(),
        }
