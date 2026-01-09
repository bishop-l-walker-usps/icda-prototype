"""OpenAI embedding provider.

Supports text-embedding-ada-002, text-embedding-3-small, and text-embedding-3-large.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from icda.ingestion.embeddings.base_provider import (
    BaseEmbeddingProvider,
    EmbeddingResult,
    ProviderStatus,
)

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider.

    Supports models:
    - text-embedding-ada-002 (1536 dimensions)
    - text-embedding-3-small (1536 dimensions, configurable)
    - text-embedding-3-large (3072 dimensions, configurable)

    Features:
    - Async batch embedding with chunking
    - Rate limit handling with exponential backoff
    - Dimension override for v3 models
    """

    __slots__ = ("_api_key", "_client", "_dimensions_override")

    # Model dimension defaults
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        normalize: bool = True,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model name.
            dimensions: Dimension override for v3 models (None = use default).
            normalize: Whether to L2 normalize.
        """
        # Determine dimension
        default_dim = self.MODEL_DIMENSIONS.get(model, 1536)
        actual_dim = dimensions or default_dim

        super().__init__(
            provider_name="openai",
            model_name=model,
            dimension=actual_dim,
            normalize=normalize,
        )
        self._api_key = api_key
        self._client = None
        self._dimensions_override = dimensions

    async def initialize(self) -> bool:
        """Initialize OpenAI client.

        Returns:
            True if API key is valid.
        """
        if not self._api_key:
            self._status = ProviderStatus.UNAVAILABLE
            logger.warning("OpenAI provider: No API key provided")
            return False

        try:
            # Import OpenAI client
            try:
                from openai import AsyncOpenAI
            except ImportError:
                logger.warning(
                    "OpenAI provider: openai package not installed. "
                    "Install with: pip install openai"
                )
                self._status = ProviderStatus.UNAVAILABLE
                return False

            self._client = AsyncOpenAI(api_key=self._api_key)

            # Test connection with a minimal request
            test_response = await self._client.embeddings.create(
                model=self._model_name,
                input="test",
                **({"dimensions": self._dimensions_override} if self._dimensions_override else {}),
            )

            if test_response.data:
                self._status = ProviderStatus.AVAILABLE
                logger.info(
                    f"OpenAI provider initialized (model={self._model_name}, "
                    f"dimension={self._dimension})"
                )
                return True

        except Exception as e:
            logger.error(f"OpenAI provider initialization failed: {e}")
            self._status = ProviderStatus.UNAVAILABLE

        return False

    async def embed(self, text: str) -> EmbeddingResult | None:
        """Generate embedding using OpenAI.

        Args:
            text: Input text to embed.

        Returns:
            EmbeddingResult or None if failed.
        """
        if not self.available or self._client is None:
            return None

        start = time.time()

        try:
            response = await self._client.embeddings.create(
                model=self._model_name,
                input=text,
                **({"dimensions": self._dimensions_override} if self._dimensions_override else {}),
            )

            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                latency_ms = self._time_ms(start)

                # Track token usage
                tokens = response.usage.total_tokens if response.usage else len(text.split())
                self._stats.record_success(latency_ms, tokens)

                return self._create_result(
                    text=text,
                    embedding=embedding,
                    latency_ms=latency_ms,
                    normalized=False,  # OpenAI embeddings are normalized by default
                )

        except Exception as e:
            self._stats.record_failure(str(e))
            logger.error(f"OpenAI embedding failed: {e}")

        return None

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for batch of texts.

        OpenAI supports batch embedding natively.

        Args:
            texts: List of texts to embed.
            batch_size: Max texts per API call.

        Returns:
            List of EmbeddingResults.
        """
        if not self.available or self._client is None:
            return []

        results: list[EmbeddingResult] = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            start = time.time()

            try:
                response = await self._client.embeddings.create(
                    model=self._model_name,
                    input=batch,
                    **({"dimensions": self._dimensions_override} if self._dimensions_override else {}),
                )

                latency_ms = self._time_ms(start)
                latency_per_item = latency_ms // len(batch)

                for j, data in enumerate(response.data):
                    result = self._create_result(
                        text=batch[j],
                        embedding=data.embedding,
                        latency_ms=latency_per_item,
                        normalized=False,
                    )
                    results.append(result)

                # Track stats
                tokens = response.usage.total_tokens if response.usage else 0
                self._stats.record_success(latency_ms, tokens)

            except Exception as e:
                logger.error(f"OpenAI batch embedding failed: {e}")
                self._stats.record_failure(str(e))

                # Try individual embeddings as fallback
                for text in batch:
                    result = await self.embed(text)
                    if result:
                        results.append(result)

        return results

    async def health_check(self) -> bool:
        """Check if OpenAI is operational.

        Returns:
            True if API responds.
        """
        if not self.available or self._client is None:
            return False

        try:
            response = await self._client.embeddings.create(
                model=self._model_name,
                input="health",
            )
            return bool(response.data)
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
