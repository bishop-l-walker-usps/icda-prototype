"""Voyage AI embedding provider.

Supports voyage-2, voyage-large-2, and voyage-code-2.
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


class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """Voyage AI embedding provider.

    Supports models:
    - voyage-2 (1024 dimensions)
    - voyage-large-2 (1536 dimensions)
    - voyage-code-2 (1536 dimensions)
    - voyage-lite-02-instruct (1024 dimensions)

    Features:
    - Input type specification (document, query)
    - Native batch support
    - Truncation handling
    """

    __slots__ = ("_api_key", "_client", "_input_type", "_truncation")

    MODEL_DIMENSIONS = {
        "voyage-2": 1024,
        "voyage-large-2": 1536,
        "voyage-code-2": 1536,
        "voyage-lite-02-instruct": 1024,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-2",
        input_type: str | None = None,
        truncation: bool = True,
        normalize: bool = True,
    ):
        """Initialize Voyage provider.

        Args:
            api_key: Voyage API key.
            model: Model name.
            input_type: Optional input type (document, query).
            truncation: Whether to truncate long inputs.
            normalize: Whether to L2 normalize.
        """
        dimension = self.MODEL_DIMENSIONS.get(model, 1024)

        super().__init__(
            provider_name="voyage",
            model_name=model,
            dimension=dimension,
            normalize=normalize,
        )
        self._api_key = api_key
        self._client = None
        self._input_type = input_type
        self._truncation = truncation

    async def initialize(self) -> bool:
        """Initialize Voyage client.

        Returns:
            True if API key is valid.
        """
        if not self._api_key:
            self._status = ProviderStatus.UNAVAILABLE
            logger.warning("Voyage provider: No API key provided")
            return False

        try:
            # Import Voyage client
            try:
                import voyageai
            except ImportError:
                logger.warning(
                    "Voyage provider: voyageai package not installed. "
                    "Install with: pip install voyageai"
                )
                self._status = ProviderStatus.UNAVAILABLE
                return False

            self._client = voyageai.AsyncClient(api_key=self._api_key)

            # Test connection
            test_response = await self._client.embed(
                texts=["test"],
                model=self._model_name,
                truncation=self._truncation,
            )

            if test_response.embeddings:
                self._status = ProviderStatus.AVAILABLE
                logger.info(
                    f"Voyage provider initialized (model={self._model_name}, "
                    f"dimension={self._dimension})"
                )
                return True

        except Exception as e:
            logger.error(f"Voyage provider initialization failed: {e}")
            self._status = ProviderStatus.UNAVAILABLE

        return False

    async def embed(self, text: str) -> EmbeddingResult | None:
        """Generate embedding using Voyage.

        Args:
            text: Input text to embed.

        Returns:
            EmbeddingResult or None if failed.
        """
        if not self.available or self._client is None:
            return None

        start = time.time()

        try:
            kwargs: dict[str, Any] = {
                "texts": [text],
                "model": self._model_name,
                "truncation": self._truncation,
            }
            if self._input_type:
                kwargs["input_type"] = self._input_type

            response = await self._client.embed(**kwargs)

            if response.embeddings and len(response.embeddings) > 0:
                embedding = response.embeddings[0]
                latency_ms = self._time_ms(start)

                tokens = response.total_tokens if hasattr(response, "total_tokens") else 0
                self._stats.record_success(latency_ms, tokens)

                return self._create_result(
                    text=text,
                    embedding=list(embedding),
                    latency_ms=latency_ms,
                    normalized=False,
                )

        except Exception as e:
            self._stats.record_failure(str(e))
            logger.error(f"Voyage embedding failed: {e}")

        return None

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 128,  # Voyage supports up to 128
    ) -> list[EmbeddingResult]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of texts to embed.
            batch_size: Max texts per API call.

        Returns:
            List of EmbeddingResults.
        """
        if not self.available or self._client is None:
            return []

        batch_size = min(batch_size, 128)
        results: list[EmbeddingResult] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            start = time.time()

            try:
                kwargs: dict[str, Any] = {
                    "texts": batch,
                    "model": self._model_name,
                    "truncation": self._truncation,
                }
                if self._input_type:
                    kwargs["input_type"] = self._input_type

                response = await self._client.embed(**kwargs)

                latency_ms = self._time_ms(start)
                latency_per_item = latency_ms // len(batch)

                for j, embedding in enumerate(response.embeddings):
                    result = self._create_result(
                        text=batch[j],
                        embedding=list(embedding),
                        latency_ms=latency_per_item,
                        normalized=False,
                    )
                    results.append(result)

                self._stats.record_success(latency_ms)

            except Exception as e:
                logger.error(f"Voyage batch embedding failed: {e}")
                self._stats.record_failure(str(e))

        return results

    async def health_check(self) -> bool:
        """Check if Voyage is operational.

        Returns:
            True if API responds.
        """
        if not self.available or self._client is None:
            return False

        try:
            response = await self._client.embed(
                texts=["health"],
                model=self._model_name,
            )
            return bool(response.embeddings)
        except Exception as e:
            logger.warning(f"Voyage health check failed: {e}")
            return False

    def set_input_type(self, input_type: str | None) -> None:
        """Set input type for embedding.

        Args:
            input_type: document, query, or None.
        """
        self._input_type = input_type
