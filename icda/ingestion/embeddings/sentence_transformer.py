"""Sentence Transformers embedding provider.

Local embedding using Hugging Face sentence-transformers models.
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


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Sentence Transformers local embedding provider.

    Runs embedding models locally using sentence-transformers library.
    No API key required.

    Supported models (examples):
    - all-MiniLM-L6-v2 (384 dimensions) - Fast, good quality
    - all-mpnet-base-v2 (768 dimensions) - Better quality
    - multi-qa-MiniLM-L6-cos-v1 (384 dimensions) - QA optimized
    - paraphrase-MiniLM-L6-v2 (384 dimensions) - Paraphrase detection

    Features:
    - Runs entirely locally (no API calls)
    - GPU acceleration support (cuda, mps)
    - Batch processing with automatic batching
    """

    __slots__ = ("_model", "_device", "_max_seq_length")

    # Common model dimensions
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "multi-qa-MiniLM-L6-cos-v1": 384,
        "paraphrase-MiniLM-L6-v2": 384,
        "all-distilroberta-v1": 768,
        "sentence-t5-base": 768,
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        """Initialize Sentence Transformer provider.

        Args:
            model_name: Model name from Hugging Face.
            device: Device to run on (cpu, cuda, mps).
            normalize: Whether to L2 normalize.
        """
        # Get dimension from known models or default
        dimension = self.MODEL_DIMENSIONS.get(model_name, 384)

        super().__init__(
            provider_name="sentence_transformers",
            model_name=model_name,
            dimension=dimension,
            normalize=normalize,
        )
        self._model = None
        self._device = device
        self._max_seq_length = 512

    async def initialize(self) -> bool:
        """Initialize model (load into memory).

        Returns:
            True if model loads successfully.
        """
        try:
            # Import sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.warning(
                    "SentenceTransformer provider: sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                self._status = ProviderStatus.UNAVAILABLE
                return False

            # Load model in executor to not block
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self._model_name, device=self._device),
            )

            # Update dimension from loaded model
            self._dimension = self._model.get_sentence_embedding_dimension()
            self._max_seq_length = self._model.max_seq_length

            self._status = ProviderStatus.AVAILABLE
            logger.info(
                f"SentenceTransformer provider initialized "
                f"(model={self._model_name}, device={self._device}, "
                f"dimension={self._dimension})"
            )
            return True

        except Exception as e:
            logger.error(f"SentenceTransformer initialization failed: {e}")
            self._status = ProviderStatus.UNAVAILABLE
            return False

    async def embed(self, text: str) -> EmbeddingResult | None:
        """Generate embedding using local model.

        Args:
            text: Input text to embed.

        Returns:
            EmbeddingResult or None if failed.
        """
        if not self.available or self._model is None:
            return None

        start = time.time()

        try:
            loop = asyncio.get_event_loop()

            # Run encoding in executor (it's synchronous)
            embedding = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=self._normalize,
                ).tolist(),
            )

            latency_ms = self._time_ms(start)
            self._stats.record_success(latency_ms)

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                dimension=len(embedding),
                provider=self._provider_name,
                normalized=self._normalize,
                latency_ms=latency_ms,
                model=self._model_name,
            )

        except Exception as e:
            self._stats.record_failure(str(e))
            logger.error(f"SentenceTransformer embedding failed: {e}")
            return None

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[EmbeddingResult]:
        """Generate embeddings for batch of texts.

        Sentence Transformers handles batching efficiently.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for encoding.

        Returns:
            List of EmbeddingResults.
        """
        if not self.available or self._model is None:
            return []

        start = time.time()

        try:
            loop = asyncio.get_event_loop()

            # Run batch encoding in executor
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                ),
            )

            latency_ms = self._time_ms(start)
            latency_per_item = latency_ms // len(texts) if texts else 0

            results = []
            for i, embedding in enumerate(embeddings):
                results.append(
                    EmbeddingResult(
                        text=texts[i],
                        embedding=embedding.tolist(),
                        dimension=len(embedding),
                        provider=self._provider_name,
                        normalized=self._normalize,
                        latency_ms=latency_per_item,
                        model=self._model_name,
                    )
                )

            self._stats.record_success(latency_ms)
            return results

        except Exception as e:
            logger.error(f"SentenceTransformer batch embedding failed: {e}")
            self._stats.record_failure(str(e))
            return []

    async def health_check(self) -> bool:
        """Check if model is loaded and operational.

        Returns:
            True if model responds.
        """
        if not self.available or self._model is None:
            return False

        try:
            result = await self.embed("health check")
            return result is not None and result.is_valid
        except Exception as e:
            logger.warning(f"SentenceTransformer health check failed: {e}")
            return False

    def get_info(self) -> dict[str, Any]:
        """Get provider information including device."""
        info = super().get_info()
        info.update(
            {
                "device": self._device,
                "max_seq_length": self._max_seq_length,
                "local": True,
            }
        )
        return info
