"""Mistral embedding provider.

Supports mistral-embed model.
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


class MistralEmbeddingProvider(BaseEmbeddingProvider):
    """Mistral embedding provider.

    Supports models:
    - mistral-embed (1024 dimensions)

    Features:
    - Native batch support
    - Async API support
    """

    __slots__ = ("_api_key", "_client")

    MODEL_DIMENSIONS = {
        "mistral-embed": 1024,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-embed",
        normalize: bool = True,
    ):
        """Initialize Mistral provider.

        Args:
            api_key: Mistral API key.
            model: Model name.
            normalize: Whether to L2 normalize.
        """
        dimension = self.MODEL_DIMENSIONS.get(model, 1024)

        super().__init__(
            provider_name="mistral",
            model_name=model,
            dimension=dimension,
            normalize=normalize,
        )
        self._api_key = api_key
        self._client = None

    async def initialize(self) -> bool:
        """Initialize Mistral client.

        Returns:
            True if API key is valid.
        """
        if not self._api_key:
            self._status = ProviderStatus.UNAVAILABLE
            logger.warning("Mistral provider: No API key provided")
            return False

        try:
            # Import Mistral client
            try:
                from mistralai import Mistral
            except ImportError:
                logger.warning(
                    "Mistral provider: mistralai package not installed. "
                    "Install with: pip install mistralai"
                )
                self._status = ProviderStatus.UNAVAILABLE
                return False

            self._client = Mistral(api_key=self._api_key)

            # Test connection
            test_response = await self._client.embeddings.create_async(
                model=self._model_name,
                inputs=["test"],
            )

            if test_response.data:
                self._status = ProviderStatus.AVAILABLE
                logger.info(
                    f"Mistral provider initialized (model={self._model_name}, "
                    f"dimension={self._dimension})"
                )
                return True

        except Exception as e:
            logger.error(f"Mistral provider initialization failed: {e}")
            self._status = ProviderStatus.UNAVAILABLE

        return False

    async def embed(self, text: str) -> EmbeddingResult | None:
        """Generate embedding using Mistral.

        Args:
            text: Input text to embed.

        Returns:
            EmbeddingResult or None if failed.
        """
        if not self.available or self._client is None:
            return None

        start = time.time()

        try:
            response = await self._client.embeddings.create_async(
                model=self._model_name,
                inputs=[text],
            )

            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                latency_ms = self._time_ms(start)

                tokens = response.usage.total_tokens if hasattr(response, "usage") and response.usage else 0
                self._stats.record_success(latency_ms, tokens)

                return self._create_result(
                    text=text,
                    embedding=embedding,
                    latency_ms=latency_ms,
                    normalized=False,
                )

        except Exception as e:
            self._stats.record_failure(str(e))
            logger.error(f"Mistral embedding failed: {e}")

        return None

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for batch of texts.

        Mistral supports batch embedding.

        Args:
            texts: List of texts to embed.
            batch_size: Max texts per API call.

        Returns:
            List of EmbeddingResults.
        """
        if not self.available or self._client is None:
            return []

        results: list[EmbeddingResult] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            start = time.time()

            try:
                response = await self._client.embeddings.create_async(
                    model=self._model_name,
                    inputs=batch,
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

                self._stats.record_success(latency_ms)

            except Exception as e:
                logger.error(f"Mistral batch embedding failed: {e}")
                self._stats.record_failure(str(e))

        return results

    async def health_check(self) -> bool:
        """Check if Mistral is operational.

        Returns:
            True if API responds.
        """
        if not self.available or self._client is None:
            return False

        try:
            response = await self._client.embeddings.create_async(
                model=self._model_name,
                inputs=["health"],
            )
            return bool(response.data)
        except Exception as e:
            logger.warning(f"Mistral health check failed: {e}")
            return False
