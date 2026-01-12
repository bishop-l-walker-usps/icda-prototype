"""Tests for Redis Stack unified client."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


class TestRedisStackClient:
    """Test suite for RedisStackClient."""

    @pytest.fixture
    def client(self):
        """Create a fresh client instance."""
        from icda.redis_stack.client import RedisStackClient
        return RedisStackClient()

    def test_init_state(self, client):
        """Test initial state of client."""
        assert client.redis is None
        assert client._connected is False
        assert client.search_available is False
        assert client.json_available is False
        assert client.timeseries_available is False
        assert client.bloom_available is False

    def test_get_module_status_disconnected(self, client):
        """Test module status when disconnected."""
        status = client._get_module_status()
        assert status["connected"] is False
        assert status["search"] is False
        assert status["json"] is False

    @pytest.mark.asyncio
    async def test_connect_no_url(self, client):
        """Test connect with no URL returns gracefully."""
        result = await client.connect("")
        assert result["connected"] is False

    @pytest.mark.asyncio
    async def test_connect_with_mock_redis(self, client):
        """Test connection with mocked Redis."""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis.module_list = AsyncMock(return_value=[
                {"name": "search"},
                {"name": "ReJSON"},
                {"name": "timeseries"},
                {"name": "bf"},
            ])
            mock_from_url.return_value = mock_redis

            result = await client.connect("redis://localhost:6379")

            assert result["connected"] is True
            assert client._connected is True
            assert client.search_available is True
            assert client.json_available is True
            assert client.timeseries_available is True
            assert client.bloom_available is True

    @pytest.mark.asyncio
    async def test_connect_timeout(self, client):
        """Test connection handles timeout gracefully."""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_from_url.return_value = mock_redis

            result = await client.connect("redis://localhost:6379", timeout=0.1)

            assert result["connected"] is False

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self, client):
        """Test health check when disconnected."""
        result = await client.health_check()
        assert result["healthy"] is False
        assert "Not connected" in result["errors"]

    @pytest.mark.asyncio
    async def test_health_check_connected(self, client):
        """Test health check when connected."""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis.module_list = AsyncMock(return_value=[])
            mock_from_url.return_value = mock_redis

            await client.connect("redis://localhost:6379")

            result = await client.health_check()
            assert result["healthy"] is True
            assert result["latency_ms"] is not None

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test close method."""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis.module_list = AsyncMock(return_value=[])
            mock_redis.aclose = AsyncMock()
            mock_from_url.return_value = mock_redis

            await client.connect("redis://localhost:6379")
            await client.close()

            mock_redis.aclose.assert_called_once()
            assert client._connected is False


class TestConvenienceMethods:
    """Test convenience methods on RedisStackClient."""

    @pytest.fixture
    def client_with_mocks(self):
        """Create client with mocked modules."""
        from icda.redis_stack.client import RedisStackClient

        client = RedisStackClient()
        client._connected = True

        # Mock timeseries
        client.timeseries_available = True
        client.timeseries = AsyncMock()
        client.timeseries.record_query = AsyncMock()

        # Mock bloom
        client.bloom_available = True
        client.bloom = AsyncMock()
        client.bloom.query_seen = AsyncMock(return_value=False)
        client.bloom.add_query = AsyncMock()
        client.bloom.track_query_frequency = AsyncMock()
        client.bloom.get_top_queries = AsyncMock(return_value=[("test", 5)])

        # Mock streams
        client.streams = AsyncMock()
        client.streams.add_query_event = AsyncMock(return_value="1234-0")

        # Mock pubsub
        client.pubsub = AsyncMock()
        client.pubsub.publish_index_progress = AsyncMock(return_value=1)

        return client

    @pytest.mark.asyncio
    async def test_record_query_metric(self, client_with_mocks):
        """Test recording query metrics."""
        await client_with_mocks.record_query_metric(100.5, cache_hit=True, agent="search")
        client_with_mocks.timeseries.record_query.assert_called_once_with(
            100.5, True, "search", False
        )

    @pytest.mark.asyncio
    async def test_record_query_event(self, client_with_mocks):
        """Test recording query events to streams."""
        result = await client_with_mocks.record_query_event(
            query="test query",
            response="test response",
            latency_ms=100,
            agents=["intent", "search"],
            cache_hit=False,
            trace_id="trace-123",
        )
        assert result == "1234-0"
        client_with_mocks.streams.add_query_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_query_seen(self, client_with_mocks):
        """Test checking if query was seen."""
        result = await client_with_mocks.check_query_seen("test query")
        assert result is False
        client_with_mocks.bloom.query_seen.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_query_seen(self, client_with_mocks):
        """Test marking query as seen."""
        await client_with_mocks.mark_query_seen("test query")
        client_with_mocks.bloom.add_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_trending_queries(self, client_with_mocks):
        """Test getting trending queries."""
        result = await client_with_mocks.get_trending_queries(10)
        assert len(result) == 1
        assert result[0] == ("test", 5)

    @pytest.mark.asyncio
    async def test_publish_index_progress(self, client_with_mocks):
        """Test publishing index progress."""
        await client_with_mocks.publish_index_progress(
            index_name="customers",
            indexed=500,
            total=1000,
            status="running",
        )
        client_with_mocks.pubsub.publish_index_progress.assert_called_once()


class TestGracefulDegradation:
    """Test graceful degradation when modules unavailable."""

    @pytest.fixture
    def client_no_modules(self):
        """Create client with no modules available."""
        from icda.redis_stack.client import RedisStackClient

        client = RedisStackClient()
        client._connected = True
        client.timeseries_available = False
        client.bloom_available = False
        client.streams = None
        client.pubsub = None
        return client

    @pytest.mark.asyncio
    async def test_record_metric_no_timeseries(self, client_no_modules):
        """Test metric recording falls back gracefully."""
        # Should not raise, just log
        await client_no_modules.record_query_metric(100)

    @pytest.mark.asyncio
    async def test_check_query_seen_no_bloom(self, client_no_modules):
        """Test query check returns False when Bloom unavailable."""
        result = await client_no_modules.check_query_seen("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_record_event_no_streams(self, client_no_modules):
        """Test event recording returns None when Streams unavailable."""
        result = await client_no_modules.record_query_event(
            query="test",
            response="response",
            latency_ms=100,
            agents=[],
            cache_hit=False,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_trending_no_bloom(self, client_no_modules):
        """Test trending queries returns empty when Bloom unavailable."""
        result = await client_no_modules.get_trending_queries()
        assert result == []
