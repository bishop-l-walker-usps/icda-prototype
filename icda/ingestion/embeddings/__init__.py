"""Embedding provider abstraction layer.

Provides a unified interface for multiple embedding providers with
automatic fallback chains and dimension normalization.

Supported Providers:
- Precomputed: Pass-through for pre-computed vectors
- Titan: AWS Bedrock Titan Embeddings
- OpenAI: text-embedding-ada-002, text-embedding-3-small/large
- Cohere: embed-english-v3.0
- Voyage: voyage-2
- Google: text-embedding-004
- Mistral: mistral-embed
- SentenceTransformers: Local models (all-MiniLM-L6-v2, etc.)
"""

from icda.ingestion.embeddings.base_provider import (
    BaseEmbeddingProvider,
    EmbeddingResult,
    ProviderStatus,
)
from icda.ingestion.embeddings.embedding_normalizer import (
    EmbeddingNormalizer,
    NormalizationMethod,
)
from icda.ingestion.embeddings.provider_chain import EmbeddingProviderChain

__all__ = [
    "BaseEmbeddingProvider",
    "EmbeddingResult",
    "ProviderStatus",
    "EmbeddingNormalizer",
    "NormalizationMethod",
    "EmbeddingProviderChain",
]
