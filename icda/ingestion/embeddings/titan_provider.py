"""AWS Titan embedding provider.

Wraps the existing EmbeddingClient for use in the provider chain,
providing consistent interface with other embedding providers.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from icda.ingestion.embeddings.base_provider import (
    BaseEmbeddingProvider,
    EmbeddingResult,
    ProviderStatus,
)

if TYPE_CHECKING:
    from icda.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


class TitanEmbeddingProvider(BaseEmbeddingProvider):
    """AWS Bedrock Titan embedding provider.

    Wraps the existing icda.embeddings.EmbeddingClient for
    integration with the embedding provider chain.

    Features:
    - Uses existing EmbeddingClient infrastructure
    - Graceful degradation if AWS unavailable
    - Batch embedding with chunking
    """

    __slots__ = ("_client", "_region")

    def __init__(
        self,
        existing_client: EmbeddingClient | None = None,
        region: str = "us-east-1",
        model: str = "amazon.titan-embed-text-v2:0",
        dimension: int = 1024,
        normalize: bool = True,
    ):
        """Initialize Titan provider.

        Args:
            existing_client: Existing EmbeddingClient to wrap.
            region: AWS region for Bedrock.
            model: Titan model name.
            dimension: Embedding dimension.
            normalize: Whether to L2 normalize.
        """
        super().__init__(
            provider_name="titan",
            model_name=model,
            dimension=dimension,
            normalize=normalize,
        )
        self._client = existing_client
        self._region = region

    async def initialize(self) -> bool:
        """Initialize provider and verify AWS access.

        If no existing client provided, creates a new EmbeddingClient.

        Returns:
            True if AWS Bedrock is accessible.
        """
        try:
            if self._client is None:
                # Import here to avoid circular imports
                from icda.embeddings import EmbeddingClient

                self._client = EmbeddingClient(
                    region=self._region,
                    model=self._model_name,
                    dimensions=self._dimension,
                )

            # Check if client is available (has valid AWS credentials)
            if self._client.available:
                self._status = ProviderStatus.AVAILABLE
                logger.info(
                    f"TitanEmbeddingProvider initialized "
                    f"(region={self._region}, model={self._model_name})"
                )
                return True
            else:
                self._status = ProviderStatus.UNAVAILABLE
                logger.warning(
                    "TitanEmbeddingProvider: AWS credentials not available, "
                    "provider will be unavailable"
                )
                return False

        except Exception as e:
            self._status = ProviderStatus.UNAVAILABLE
            logger.error(f"TitanEmbeddingProvider initialization failed: {e}")
            return False

    async def embed(self, text: str) -> EmbeddingResult | None:
        """Generate embedding using Titan.

        Args:
            text: Input text to embed.

        Returns:
            EmbeddingResult or None if failed.
        """
        if not self.available or self._client is None:
            return None

        start = time.time()

        try:
            # EmbeddingClient.embed() is synchronous
            embedding = self._client.embed(text)

            if not embedding:
                self._stats.record_failure("Empty embedding returned")
                return None

            latency_ms = self._time_ms(start)
            self._stats.record_success(latency_ms, tokens=len(text.split()))

            return self._create_result(
                text=text,
                embedding=embedding,
                latency_ms=latency_ms,
                normalized=True,  # Titan returns normalized embeddings
            )

        except Exception as e:
            self._stats.record_failure(str(e))
            logger.error(f"Titan embedding failed: {e}")
            return None

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for batch of texts.

        Note: Titan doesn't have a native batch API, so this
        processes texts sequentially.

        Args:
            texts: List of texts to embed.
            batch_size: Ignored for Titan (sequential processing).

        Returns:
            List of EmbeddingResults.
        """
        results: list[EmbeddingResult] = []

        for text in texts:
            result = await self.embed(text)
            if result:
                results.append(result)

        return results

    async def health_check(self) -> bool:
        """Check if Titan is operational.

        Attempts a small embedding to verify connectivity.

        Returns:
            True if Titan responds successfully.
        """
        if not self.available or self._client is None:
            return False

        try:
            # Try a minimal embedding
            embedding = self._client.embed("health check")
            return embedding is not None and len(embedding) > 0
        except Exception as e:
            logger.warning(f"Titan health check failed: {e}")
            return False

    @classmethod
    def from_existing_client(
        cls,
        client: EmbeddingClient,
    ) -> TitanEmbeddingProvider:
        """Create provider from existing EmbeddingClient.

        Args:
            client: Existing EmbeddingClient instance.

        Returns:
            Configured TitanEmbeddingProvider.
        """
        provider = cls(
            existing_client=client,
            region=getattr(client, "region", "us-east-1"),
            model=getattr(client, "model", "amazon.titan-embed-text-v2:0"),
            dimension=getattr(client, "dimensions", 1024),
        )
        return provider
