"""Tests for embedding providers and chain."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from icda.ingestion.embeddings.base_provider import (
    BaseEmbeddingProvider,
    EmbeddingResult,
    ProviderStatus,
)
from icda.ingestion.embeddings.precomputed_provider import PrecomputedEmbeddingProvider
from icda.ingestion.embeddings.embedding_normalizer import EmbeddingNormalizer
from icda.ingestion.embeddings.provider_chain import EmbeddingProviderChain


class TestPrecomputedProvider:
    """Test PrecomputedEmbeddingProvider."""

    @pytest.fixture
    def provider(self):
        """Create precomputed provider."""
        return PrecomputedEmbeddingProvider(expected_dimension=1024)

    @pytest.mark.asyncio
    async def test_initialize(self, provider):
        """Test provider initialization."""
        result = await provider.initialize()
        assert result is True
        assert provider.available is True

    @pytest.mark.asyncio
    async def test_no_text_embedding(self, provider):
        """Test that text embedding returns None."""
        await provider.initialize()

        result = await provider.embed("Some text")
        assert result is None


class TestEmbeddingNormalizer:
    """Test EmbeddingNormalizer."""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer targeting 1024 dimensions."""
        return EmbeddingNormalizer(target_dimension=1024)

    def test_no_change_same_dimension(self, normalizer):
        """Test no change when dimensions match."""
        embedding = [0.1] * 1024
        result = normalizer.normalize(embedding, source_dimension=1024)

        assert len(result) == 1024
        assert result == embedding


class TestProviderChain:
    """Test EmbeddingProviderChain."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = MagicMock(spec=BaseEmbeddingProvider)
        provider.provider_name = "mock"
        provider.available = True
        provider.dimension = 1024
        provider.embed = AsyncMock(return_value=EmbeddingResult(
            text="test",
            embedding=[0.1] * 1024,
            dimension=1024,
            provider="mock",
        ))
        return provider

    @pytest.fixture
    def failing_provider(self):
        """Create a failing provider."""
        provider = MagicMock(spec=BaseEmbeddingProvider)
        provider.provider_name = "failing"
        provider.available = True
        provider.dimension = 1024
        provider.embed = AsyncMock(side_effect=Exception("Provider failed"))
        return provider

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, failing_provider):
        """Test behavior when all providers fail."""
        another_failing = MagicMock(spec=BaseEmbeddingProvider)
        another_failing.provider_name = "failing2"
        another_failing.available = True
        another_failing.embed = AsyncMock(side_effect=Exception("Also failed"))

        chain = EmbeddingProviderChain(
            providers=[failing_provider, another_failing],
            target_dimension=1024,
        )
        await chain.initialize()

        result = await chain.embed("test text")
        assert result is None


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_creation(self):
        """Test creating EmbeddingResult."""
        result = EmbeddingResult(
            text="test input",
            embedding=[0.1, 0.2, 0.3],
            dimension=3,
            provider="test",
        )

        assert result.text == "test input"
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.dimension == 3
        assert result.provider == "test"

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = EmbeddingResult(
            text="test",
            embedding=[0.1, 0.2],
            dimension=2,
            provider="test",
            latency_ms=100,
        )

        d = result.to_dict()
        assert d["dimension"] == 2
        assert d["provider"] == "test"
        assert d["latency_ms"] == 100

    def test_is_valid(self):
        """Test is_valid property."""
        valid_result = EmbeddingResult(
            text="test",
            embedding=[0.1, 0.2],
            dimension=2,
            provider="test",
        )
        assert valid_result.is_valid is True

        invalid_result = EmbeddingResult(
            text="test",
            embedding=[],
            dimension=0,
            provider="test",
        )
        assert invalid_result.is_valid is False


class TestProviderStatus:
    """Test ProviderStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert ProviderStatus.AVAILABLE.value == "available"
        assert ProviderStatus.UNAVAILABLE.value == "unavailable"
        assert ProviderStatus.RATE_LIMITED.value == "rate_limited"
