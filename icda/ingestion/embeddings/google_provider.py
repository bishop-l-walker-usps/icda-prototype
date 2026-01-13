"""Google embedding provider.

Supports text-embedding-004 and other Google embedding models.
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


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """Google embedding provider using Generative AI API.

    Supports models:
    - text-embedding-004 (768 dimensions)
    - embedding-001 (768 dimensions)

    Features:
    - Task type specification
    - Title support for retrieval tasks
    - Output dimensionality configuration
    """

    __slots__ = ("_api_key", "_client", "_task_type", "_output_dimensionality")

    MODEL_DIMENSIONS = {
        "text-embedding-004": 768,
        "embedding-001": 768,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int | None = None,
        normalize: bool = True,
    ):
        """Initialize Google provider.

        Args:
            api_key: Google API key.
            model: Model name.
            task_type: Task type (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.).
            output_dimensionality: Optional output dimension override.
            normalize: Whether to L2 normalize.
        """
        dimension = output_dimensionality or self.MODEL_DIMENSIONS.get(model, 768)

        super().__init__(
            provider_name="google",
            model_name=model,
            dimension=dimension,
            normalize=normalize,
        )
        self._api_key = api_key
        self._client = None
        self._task_type = task_type
        self._output_dimensionality = output_dimensionality

    async def initialize(self) -> bool:
        """Initialize Google client.

        Returns:
            True if API key is valid.
        """
        if not self._api_key:
            self._status = ProviderStatus.UNAVAILABLE
            logger.warning("Google provider: No API key provided")
            return False

        try:
            # Import Google Generative AI client
            try:
                import google.generativeai as genai
            except ImportError:
                logger.warning(
                    "Google provider: google-generativeai package not installed. "
                    "Install with: pip install google-generativeai"
                )
                self._status = ProviderStatus.UNAVAILABLE
                return False

            genai.configure(api_key=self._api_key)
            self._client = genai

            # Test connection
            test_result = genai.embed_content(
                model=f"models/{self._model_name}",
                content="test",
                task_type=self._task_type,
            )

            if test_result and "embedding" in test_result:
                self._status = ProviderStatus.AVAILABLE
                logger.info(
                    f"Google provider initialized (model={self._model_name}, "
                    f"dimension={self._dimension})"
                )
                return True

        except Exception as e:
            logger.error(f"Google provider initialization failed: {e}")
            self._status = ProviderStatus.UNAVAILABLE

        return False

    async def embed(self, text: str) -> EmbeddingResult | None:
        """Generate embedding using Google.

        Note: google-generativeai is synchronous, so we wrap in executor.

        Args:
            text: Input text to embed.

        Returns:
            EmbeddingResult or None if failed.
        """
        if not self.available or self._client is None:
            return None

        start = time.time()

        try:
            import asyncio

            # Run synchronous API in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._client.embed_content(
                    model=f"models/{self._model_name}",
                    content=text,
                    task_type=self._task_type,
                    **({"output_dimensionality": self._output_dimensionality}
                       if self._output_dimensionality else {}),
                ),
            )

            if result and "embedding" in result:
                embedding = result["embedding"]
                latency_ms = self._time_ms(start)

                self._stats.record_success(latency_ms)

                return self._create_result(
                    text=text,
                    embedding=embedding,
                    latency_ms=latency_ms,
                    normalized=False,
                )

        except Exception as e:
            self._stats.record_failure(str(e))
            logger.error(f"Google embedding failed: {e}")

        return None

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for batch of texts.

        Google supports batch embedding.

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
                import asyncio

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._client.embed_content(
                        model=f"models/{self._model_name}",
                        content=batch,
                        task_type=self._task_type,
                        **({"output_dimensionality": self._output_dimensionality}
                           if self._output_dimensionality else {}),
                    ),
                )

                latency_ms = self._time_ms(start)
                latency_per_item = latency_ms // len(batch)

                if result and "embedding" in result:
                    # Single text returns single embedding
                    embeddings = result["embedding"]
                    if isinstance(embeddings[0], list):
                        # Batch response
                        for j, embedding in enumerate(embeddings):
                            results.append(
                                self._create_result(
                                    text=batch[j],
                                    embedding=embedding,
                                    latency_ms=latency_per_item,
                                    normalized=False,
                                )
                            )
                    else:
                        # Single response
                        results.append(
                            self._create_result(
                                text=batch[0],
                                embedding=embeddings,
                                latency_ms=latency_ms,
                                normalized=False,
                            )
                        )

                self._stats.record_success(latency_ms)

            except Exception as e:
                logger.error(f"Google batch embedding failed: {e}")
                self._stats.record_failure(str(e))

                # Fallback to individual
                for text in batch:
                    result = await self.embed(text)
                    if result:
                        results.append(result)

        return results

    async def health_check(self) -> bool:
        """Check if Google is operational.

        Returns:
            True if API responds.
        """
        if not self.available or self._client is None:
            return False

        try:
            import asyncio

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._client.embed_content(
                    model=f"models/{self._model_name}",
                    content="health",
                    task_type=self._task_type,
                ),
            )
            return bool(result and "embedding" in result)
        except Exception as e:
            logger.warning(f"Google health check failed: {e}")
            return False

    def set_task_type(self, task_type: str) -> None:
        """Set task type for embedding.

        Args:
            task_type: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY,
                      CLASSIFICATION, CLUSTERING.
        """
        valid_types = [
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
        ]
        if task_type in valid_types:
            self._task_type = task_type
        else:
            logger.warning(f"Invalid task type: {task_type}")
