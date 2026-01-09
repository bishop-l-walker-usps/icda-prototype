"""Cohere embedding provider.

Supports embed-english-v3.0 and embed-multilingual-v3.0.
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


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embedding provider.

    Supports models:
    - embed-english-v3.0 (1024 dimensions)
    - embed-multilingual-v3.0 (1024 dimensions)
    - embed-english-light-v3.0 (384 dimensions)
    - embed-multilingual-light-v3.0 (384 dimensions)

    Features:
    - Input type specification (search_document, search_query)
    - Native batch support
    - Truncation handling
    """

    __slots__ = ("_api_key", "_client", "_input_type", "_truncate")

    # Model dimension defaults
    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
        truncate: str = "END",
        normalize: bool = True,
    ):
        """Initialize Cohere provider.

        Args:
            api_key: Cohere API key.
            model: Model name.
            input_type: Input type (search_document, search_query, classification, clustering).
            truncate: Truncation strategy (NONE, START, END).
            normalize: Whether to L2 normalize.
        """
        dimension = self.MODEL_DIMENSIONS.get(model, 1024)

        super().__init__(
            provider_name="cohere",
            model_name=model,
            dimension=dimension,
            normalize=normalize,
        )
        self._api_key = api_key
        self._client = None
        self._input_type = input_type
        self._truncate = truncate

    async def initialize(self) -> bool:
        """Initialize Cohere client.

        Returns:
            True if API key is valid.
        """
        if not self._api_key:
            self._status = ProviderStatus.UNAVAILABLE
            logger.warning("Cohere provider: No API key provided")
            return False

        try:
            # Import Cohere client
            try:
                import cohere
            except ImportError:
                logger.warning(
                    "Cohere provider: cohere package not installed. "
                    "Install with: pip install cohere"
                )
                self._status = ProviderStatus.UNAVAILABLE
                return False

            self._client = cohere.AsyncClient(api_key=self._api_key)

            # Test connection
            test_response = await self._client.embed(
                model=self._model_name,
                texts=["test"],
                input_type=self._input_type,
                truncate=self._truncate,
            )

            if test_response.embeddings:
                self._status = ProviderStatus.AVAILABLE
                logger.info(
                    f"Cohere provider initialized (model={self._model_name}, "
                    f"dimension={self._dimension})"
                )
                return True

        except Exception as e:
            logger.error(f"Cohere provider initialization failed: {e}")
            self._status = ProviderStatus.UNAVAILABLE

        return False

    async def embed(self, text: str) -> EmbeddingResult | None:
        """Generate embedding using Cohere.

        Args:
            text: Input text to embed.

        Returns:
            EmbeddingResult or None if failed.
        """
        if not self.available or self._client is None:
            return None

        start = time.time()

        try:
            response = await self._client.embed(
                model=self._model_name,
                texts=[text],
                input_type=self._input_type,
                truncate=self._truncate,
            )

            if response.embeddings and len(response.embeddings) > 0:
                embedding = response.embeddings[0]
                latency_ms = self._time_ms(start)

                self._stats.record_success(latency_ms)

                return self._create_result(
                    text=text,
                    embedding=list(embedding),
                    latency_ms=latency_ms,
                    normalized=False,
                )

        except Exception as e:
            self._stats.record_failure(str(e))
            logger.error(f"Cohere embedding failed: {e}")

        return None

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 96,  # Cohere max is 96
    ) -> list[EmbeddingResult]:
        """Generate embeddings for batch of texts.

        Cohere supports native batch embedding (max 96 texts).

        Args:
            texts: List of texts to embed.
            batch_size: Max texts per API call (max 96 for Cohere).

        Returns:
            List of EmbeddingResults.
        """
        if not self.available or self._client is None:
            return []

        # Enforce Cohere's batch limit
        batch_size = min(batch_size, 96)
        results: list[EmbeddingResult] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            start = time.time()

            try:
                response = await self._client.embed(
                    model=self._model_name,
                    texts=batch,
                    input_type=self._input_type,
                    truncate=self._truncate,
                )

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
                logger.error(f"Cohere batch embedding failed: {e}")
                self._stats.record_failure(str(e))

        return results

    async def health_check(self) -> bool:
        """Check if Cohere is operational.

        Returns:
            True if API responds.
        """
        if not self.available or self._client is None:
            return False

        try:
            response = await self._client.embed(
                model=self._model_name,
                texts=["health"],
                input_type=self._input_type,
            )
            return bool(response.embeddings)
        except Exception as e:
            logger.warning(f"Cohere health check failed: {e}")
            return False

    def set_input_type(self, input_type: str) -> None:
        """Set input type for embedding.

        Args:
            input_type: search_document, search_query, classification, or clustering.
        """
        valid_types = ["search_document", "search_query", "classification", "clustering"]
        if input_type in valid_types:
            self._input_type = input_type
        else:
            logger.warning(f"Invalid input type: {input_type}, using {self._input_type}")
