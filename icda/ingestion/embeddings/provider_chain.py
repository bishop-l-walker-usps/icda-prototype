"""Embedding provider chain with fallback and circuit breaker.

Manages multiple embedding providers with automatic fallback
when providers fail or are rate limited.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from icda.ingestion.embeddings.base_provider import (
    BaseEmbeddingProvider,
    EmbeddingResult,
    ProviderStatus,
)
from icda.ingestion.embeddings.embedding_normalizer import EmbeddingNormalizer
from icda.ingestion.embeddings.precomputed_provider import PrecomputedEmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CircuitBreakerState:
    """State for a circuit breaker."""

    failures: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half_open
    last_success_time: float = 0.0

    def record_failure(self) -> None:
        """Record a failure."""
        self.failures += 1
        self.last_failure_time = time.time()

    def record_success(self) -> None:
        """Record a success."""
        self.failures = 0
        self.last_success_time = time.time()
        self.state = "closed"

    def reset(self) -> None:
        """Reset circuit breaker."""
        self.failures = 0
        self.state = "closed"


@dataclass(slots=True)
class ChainStats:
    """Statistics for provider chain operations."""

    total_calls: int = 0
    precomputed_used: int = 0
    provider_used: dict[str, int] = field(default_factory=dict)
    fallbacks: int = 0
    failures: int = 0
    total_latency_ms: int = 0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across all calls."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "precomputed_used": self.precomputed_used,
            "provider_used": self.provider_used,
            "fallbacks": self.fallbacks,
            "failures": self.failures,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


class EmbeddingProviderChain:
    """Manages fallback chain of embedding providers.

    Features:
    - Primary -> Secondary -> Tertiary fallback
    - Circuit breaker per provider
    - Automatic dimension normalization
    - Pre-computed embedding passthrough
    - Statistics tracking

    Usage:
        chain = EmbeddingProviderChain(
            providers=[titan_provider, openai_provider, local_provider],
            target_dimension=1024,
        )
        await chain.initialize()

        # With pre-computed embedding
        result = await chain.embed(text, precomputed=[0.1, 0.2, ...])

        # Generate new embedding
        result = await chain.embed(text)
    """

    __slots__ = (
        "_providers",
        "_target_dimension",
        "_normalizer",
        "_precomputed_provider",
        "_circuit_breakers",
        "_circuit_threshold",
        "_circuit_timeout",
        "_stats",
        "_initialized",
    )

    def __init__(
        self,
        providers: list[BaseEmbeddingProvider],
        target_dimension: int = 1024,
        enable_normalization: bool = True,
        circuit_threshold: int = 5,
        circuit_timeout: int = 300,
    ):
        """Initialize provider chain.

        Args:
            providers: Ordered list of providers (first = primary).
            target_dimension: Target embedding dimension.
            enable_normalization: Whether to normalize dimensions.
            circuit_threshold: Failures before circuit opens.
            circuit_timeout: Seconds before circuit resets.
        """
        self._providers = providers
        self._target_dimension = target_dimension
        self._circuit_threshold = circuit_threshold
        self._circuit_timeout = circuit_timeout
        self._stats = ChainStats()
        self._initialized = False

        # Create normalizer
        self._normalizer = (
            EmbeddingNormalizer(target_dimension=target_dimension)
            if enable_normalization
            else None
        )

        # Create precomputed provider
        self._precomputed_provider = PrecomputedEmbeddingProvider(
            expected_dimension=target_dimension,
            normalize=True,
        )

        # Initialize circuit breakers
        self._circuit_breakers: dict[str, CircuitBreakerState] = {}
        for provider in providers:
            self._circuit_breakers[provider.provider_name] = CircuitBreakerState()

    @property
    def target_dimension(self) -> int:
        """Get target dimension."""
        return self._target_dimension

    @property
    def stats(self) -> ChainStats:
        """Get chain statistics."""
        return self._stats

    @property
    def available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return [
            p.provider_name
            for p in self._providers
            if p.available and not self._is_circuit_open(p.provider_name)
        ]

    async def initialize(self) -> bool:
        """Initialize all providers in the chain.

        Returns:
            True if at least one provider is available.
        """
        # Initialize precomputed provider
        await self._precomputed_provider.initialize()

        # Initialize all providers
        for provider in self._providers:
            try:
                await provider.initialize()
                logger.info(
                    f"Provider {provider.provider_name} initialized: "
                    f"available={provider.available}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize provider {provider.provider_name}: {e}"
                )

        self._initialized = True

        available = self.available_providers
        logger.info(
            f"EmbeddingProviderChain initialized with {len(available)} available "
            f"providers: {available}"
        )

        return len(available) > 0

    async def embed(
        self,
        text: str,
        precomputed: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingResult | None:
        """Generate or process embedding.

        If precomputed embedding provided, validates and normalizes it.
        Otherwise, tries providers in order until one succeeds.

        Args:
            text: Input text.
            precomputed: Optional pre-computed embedding vector.
            metadata: Optional metadata.

        Returns:
            EmbeddingResult or None if all providers fail.
        """
        self._stats.total_calls += 1
        start = time.time()

        # Handle pre-computed embedding
        if precomputed:
            result = await self._handle_precomputed(text, precomputed, metadata)
            if result:
                self._stats.precomputed_used += 1
                self._stats.total_latency_ms += result.latency_ms
                return result

        # Try providers in order
        for provider in self._providers:
            # Skip unavailable providers
            if not provider.available:
                continue

            # Check circuit breaker
            if self._is_circuit_open(provider.provider_name):
                continue

            try:
                result = await provider.embed(text)

                if result:
                    # Normalize if needed
                    if self._normalizer and result.dimension != self._target_dimension:
                        result.embedding = self._normalizer.normalize(result.embedding)
                        result = EmbeddingResult(
                            text=result.text,
                            embedding=result.embedding,
                            dimension=self._target_dimension,
                            provider=result.provider,
                            normalized=True,
                            latency_ms=result.latency_ms,
                            model=result.model,
                            metadata=result.metadata,
                        )

                    # Record success
                    self._record_success(provider.provider_name)
                    self._stats.provider_used[provider.provider_name] = (
                        self._stats.provider_used.get(provider.provider_name, 0) + 1
                    )
                    self._stats.total_latency_ms += result.latency_ms

                    return result

            except Exception as e:
                logger.warning(
                    f"Provider {provider.provider_name} failed: {e}, trying next"
                )
                self._record_failure(provider.provider_name)
                self._stats.fallbacks += 1
                continue

        # All providers failed
        self._stats.failures += 1
        logger.error(f"All embedding providers failed for text: {text[:50]}...")
        return None

    async def embed_batch(
        self,
        texts: list[str],
        precomputed_embeddings: list[list[float] | None] | None = None,
        batch_size: int = 100,
    ) -> list[EmbeddingResult | None]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of input texts.
            precomputed_embeddings: Optional list of pre-computed embeddings
                                   (None for texts that need generation).
            batch_size: Batch size for API calls.

        Returns:
            List of EmbeddingResults (None for failures).
        """
        results: list[EmbeddingResult | None] = []

        # Align precomputed list with texts
        if precomputed_embeddings is None:
            precomputed_embeddings = [None] * len(texts)

        for i, (text, precomputed) in enumerate(zip(texts, precomputed_embeddings)):
            result = await self.embed(text, precomputed=precomputed)
            results.append(result)

        return results

    async def _handle_precomputed(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> EmbeddingResult | None:
        """Handle pre-computed embedding.

        Args:
            text: Original text.
            embedding: Pre-computed embedding.
            metadata: Optional metadata.

        Returns:
            Processed EmbeddingResult or None.
        """
        # Normalize dimension if needed
        if self._normalizer and len(embedding) != self._target_dimension:
            embedding = self._normalizer.normalize(embedding)

        return await self._precomputed_provider.embed_precomputed(
            text=text,
            embedding=embedding,
            metadata=metadata,
        )

    def _is_circuit_open(self, provider_name: str) -> bool:
        """Check if circuit breaker is open for a provider.

        Args:
            provider_name: Provider to check.

        Returns:
            True if circuit is open (provider should be skipped).
        """
        breaker = self._circuit_breakers.get(provider_name)
        if not breaker:
            return False

        if breaker.state == "open":
            # Check if timeout has passed
            elapsed = time.time() - breaker.last_failure_time
            if elapsed > self._circuit_timeout:
                # Move to half-open state
                breaker.state = "half_open"
                logger.info(
                    f"Circuit breaker for {provider_name} moved to half-open"
                )
                return False
            return True

        return False

    def _record_failure(self, provider_name: str) -> None:
        """Record a provider failure.

        Args:
            provider_name: Provider that failed.
        """
        breaker = self._circuit_breakers.get(provider_name)
        if not breaker:
            return

        breaker.record_failure()

        # Open circuit if threshold reached
        if breaker.failures >= self._circuit_threshold:
            breaker.state = "open"
            logger.warning(
                f"Circuit breaker OPEN for {provider_name} after "
                f"{breaker.failures} failures"
            )

    def _record_success(self, provider_name: str) -> None:
        """Record a provider success.

        Args:
            provider_name: Provider that succeeded.
        """
        breaker = self._circuit_breakers.get(provider_name)
        if breaker:
            breaker.record_success()

    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers.

        Returns:
            Dict mapping provider name to health status.
        """
        health = {}
        for provider in self._providers:
            try:
                healthy = await provider.health_check()
                health[provider.provider_name] = healthy
            except Exception:
                health[provider.provider_name] = False
        return health

    def get_info(self) -> dict[str, Any]:
        """Get chain information."""
        return {
            "target_dimension": self._target_dimension,
            "initialized": self._initialized,
            "providers": [p.get_info() for p in self._providers],
            "available_providers": self.available_providers,
            "circuit_breakers": {
                name: {"state": cb.state, "failures": cb.failures}
                for name, cb in self._circuit_breakers.items()
            },
            "stats": self._stats.to_dict(),
        }
