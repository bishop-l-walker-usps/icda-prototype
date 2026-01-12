"""Tests for individual Redis Stack module wrappers."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestRedisTimeSeriesWrapper:
    """Tests for RedisTimeSeries wrapper."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper with mocked Redis."""
        from icda.redis_stack.redis_timeseries import RedisTimeSeriesWrapper

        mock_redis = AsyncMock()
        mock_redis.execute_command = AsyncMock()
        return RedisTimeSeriesWrapper(mock_redis)

    @pytest.mark.asyncio
    async def test_ensure_timeseries(self, wrapper):
        """Test time series creation."""
        wrapper.redis.execute_command = AsyncMock(
            side_effect=Exception("not exists")
        )
        await wrapper.ensure_timeseries()
        # Should create multiple time series
        assert wrapper.redis.execute_command.call_count > 0

    @pytest.mark.asyncio
    async def test_record_query(self, wrapper):
        """Test recording query metrics."""
        wrapper.redis.execute_command = AsyncMock(return_value="OK")

        await wrapper.record_query(100.5, cache_hit=True, agent="search")

        # Should record volume, latency, cache hit, and agent latency
        assert wrapper.redis.execute_command.call_count >= 3

    @pytest.mark.asyncio
    async def test_get_query_stats(self, wrapper):
        """Test getting query statistics."""
        wrapper.redis.execute_command = AsyncMock(return_value=[
            [1704067200000, "100"],
            [1704067260000, "150"],
        ])

        stats = await wrapper.get_query_stats(range_ms=3600000)

        assert "query_count" in stats
        assert "latency_avg" in stats
        assert "cache_hit_rate" in stats


class TestRedisBloomWrapper:
    """Tests for RedisBloom wrapper."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper with mocked Redis."""
        from icda.redis_stack.redis_bloom import RedisBloomWrapper

        mock_redis = AsyncMock()
        mock_redis.execute_command = AsyncMock()
        return RedisBloomWrapper(mock_redis)

    @pytest.mark.asyncio
    async def test_query_seen_false(self, wrapper):
        """Test query not seen."""
        wrapper.redis.execute_command = AsyncMock(return_value=0)

        result = await wrapper.query_seen("test query")

        assert result is False

    @pytest.mark.asyncio
    async def test_query_seen_true(self, wrapper):
        """Test query was seen."""
        wrapper.redis.execute_command = AsyncMock(return_value=1)

        result = await wrapper.query_seen("test query")

        assert result is True

    @pytest.mark.asyncio
    async def test_add_query(self, wrapper):
        """Test adding query to bloom filter."""
        wrapper.redis.execute_command = AsyncMock(return_value=1)

        await wrapper.add_query("test query")

        wrapper.redis.execute_command.assert_called()

    @pytest.mark.asyncio
    async def test_track_query_frequency(self, wrapper):
        """Test tracking query frequency."""
        wrapper.redis.execute_command = AsyncMock(return_value="OK")

        await wrapper.track_query_frequency("test query")

        # Should update CMS and TopK
        assert wrapper.redis.execute_command.call_count >= 2

    @pytest.mark.asyncio
    async def test_get_top_queries(self, wrapper):
        """Test getting top queries."""
        wrapper.redis.execute_command = AsyncMock(return_value=[
            "query1", "10",
            "query2", "5",
        ])

        result = await wrapper.get_top_queries(5)

        assert len(result) == 2
        assert result[0] == ("query1", 10)
        assert result[1] == ("query2", 5)

    def test_normalize_query(self, wrapper):
        """Test query normalization."""
        assert wrapper._normalize_query("  TEST Query  ") == "test query"

    def test_hash_content(self, wrapper):
        """Test content hashing."""
        hash1 = wrapper._hash_content("test content")
        hash2 = wrapper._hash_content("test content")
        hash3 = wrapper._hash_content("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 32  # 32 hex chars


class TestRedisPubSubManager:
    """Tests for Redis Pub/Sub manager."""

    @pytest.fixture
    def manager(self):
        """Create manager with mocked Redis."""
        from icda.redis_stack.redis_pubsub import RedisPubSubManager

        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)
        mock_redis.pubsub = MagicMock(return_value=AsyncMock())
        return RedisPubSubManager(mock_redis)

    @pytest.mark.asyncio
    async def test_publish_dict(self, manager):
        """Test publishing dict message."""
        result = await manager.publish("test:channel", {"key": "value"})

        assert result == 1
        manager.redis.publish.assert_called_once()
        call_args = manager.redis.publish.call_args
        assert call_args[0][0] == "test:channel"
        assert json.loads(call_args[0][1]) == {"key": "value"}

    @pytest.mark.asyncio
    async def test_publish_string(self, manager):
        """Test publishing string message."""
        result = await manager.publish("test:channel", "simple message")

        assert result == 1
        manager.redis.publish.assert_called_with("test:channel", "simple message")

    @pytest.mark.asyncio
    async def test_publish_index_progress(self, manager):
        """Test publishing index progress."""
        from icda.redis_stack.models import IndexProgress

        progress = IndexProgress(
            index_name="customers",
            indexed=500,
            total=1000,
            status="running",
        )

        result = await manager.publish_index_progress(progress)

        assert result == 1

    @pytest.mark.asyncio
    async def test_publish_admin_alert(self, manager):
        """Test publishing admin alert."""
        result = await manager.publish_admin_alert(
            level="error",
            message="Test error",
            details={"code": 500},
        )

        assert result == 1


class TestRedisStreamsManager:
    """Tests for Redis Streams manager."""

    @pytest.fixture
    def manager(self):
        """Create manager with mocked Redis."""
        from icda.redis_stack.redis_streams import RedisStreamsManager

        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value="1704067200000-0")
        mock_redis.xrange = AsyncMock(return_value=[])
        mock_redis.xrevrange = AsyncMock(return_value=[])
        mock_redis.exists = AsyncMock(return_value=True)
        mock_redis.xgroup_create = AsyncMock()
        return RedisStreamsManager(mock_redis)

    @pytest.mark.asyncio
    async def test_add_query_event(self, manager):
        """Test adding query event."""
        from icda.redis_stack.models import QueryEvent

        event = QueryEvent(
            query="test query",
            response_preview="test response",
            latency_ms=100,
            agent_chain=["intent", "search"],
            cache_hit=False,
        )

        result = await manager.add_query_event(event)

        assert result == "1704067200000-0"
        manager.redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_customer_event(self, manager):
        """Test adding customer event."""
        from icda.redis_stack.models import CustomerEvent

        event = CustomerEvent(
            crid="CRID-001",
            action="updated",
            changes={"status": "INACTIVE"},
        )

        result = await manager.add_customer_event(event)

        assert result == "1704067200000-0"

    @pytest.mark.asyncio
    async def test_add_error_event(self, manager):
        """Test adding error event."""
        result = await manager.add_error_event(
            error_type="RuntimeError",
            message="Test error",
            context={"endpoint": "/api/query"},
        )

        assert result == "1704067200000-0"

    @pytest.mark.asyncio
    async def test_read_recent(self, manager):
        """Test reading recent events."""
        manager.redis.xrevrange = AsyncMock(return_value=[
            ("1704067200000-0", {"query": "test", "latency_ms": "100"}),
            ("1704067100000-0", {"query": "test2", "latency_ms": "50"}),
        ])

        result = await manager.read_recent("stream:query:audit", count=10)

        assert len(result) == 2
        assert result[0]["query"] == "test"


class TestRedisSearchEnhanced:
    """Tests for RediSearch enhanced wrapper."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper with mocked Redis."""
        from icda.redis_stack.redis_search import RedisSearchEnhanced

        mock_redis = AsyncMock()
        mock_redis.execute_command = AsyncMock()
        return RedisSearchEnhanced(mock_redis)

    @pytest.mark.asyncio
    async def test_add_suggestion(self, wrapper):
        """Test adding suggestion."""
        wrapper.redis.execute_command = AsyncMock(return_value=1)

        result = await wrapper.add_suggestion("suggest:test", "test string", score=1.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_get_suggestions(self, wrapper):
        """Test getting suggestions."""
        wrapper.redis.execute_command = AsyncMock(return_value=[
            "suggestion1", "0.9",
            "suggestion2", "0.8",
        ])

        result = await wrapper.get_suggestions(
            "suggest:test", "sug", fuzzy=True, max_results=5, with_scores=True
        )

        assert len(result) == 2
        assert result[0].text == "suggestion1"
        assert result[0].score == 0.9

    @pytest.mark.asyncio
    async def test_facet_search(self, wrapper):
        """Test faceted search."""
        wrapper.redis.execute_command = AsyncMock(return_value=[
            10,  # total
            ["state", "TX", "count", "5"],
            ["state", "CA", "count", "3"],
        ])

        result = await wrapper.facet_search("idx:customers", "*", "state", limit=10)

        assert result.field == "state"
        assert len(result.values) == 2


class TestRedisJSONWrapper:
    """Tests for RedisJSON wrapper."""

    @pytest.fixture
    def wrapper(self):
        """Create wrapper with mocked Redis."""
        from icda.redis_stack.redis_json import RedisJSONWrapper

        mock_redis = AsyncMock()
        mock_redis.execute_command = AsyncMock()
        return RedisJSONWrapper(mock_redis)

    @pytest.mark.asyncio
    async def test_set(self, wrapper):
        """Test setting JSON value."""
        wrapper.redis.execute_command = AsyncMock(return_value="OK")

        result = await wrapper.set("test:key", "$", {"name": "test"})

        assert result is True
        wrapper.redis.execute_command.assert_called_with(
            "JSON.SET", "test:key", "$", '{"name": "test"}'
        )

    @pytest.mark.asyncio
    async def test_get(self, wrapper):
        """Test getting JSON value."""
        wrapper.redis.execute_command = AsyncMock(return_value='{"name": "test"}')

        result = await wrapper.get("test:key")

        assert result == {"name": "test"}

    @pytest.mark.asyncio
    async def test_incr(self, wrapper):
        """Test incrementing numeric value."""
        wrapper.redis.execute_command = AsyncMock(return_value="[6]")

        result = await wrapper.incr("test:key", "$.count", 1)

        assert result == 6.0

    @pytest.mark.asyncio
    async def test_arr_append(self, wrapper):
        """Test appending to array."""
        wrapper.redis.execute_command = AsyncMock(return_value=[3])

        result = await wrapper.arr_append("test:key", "$.items", "new_item")

        assert result == 3

    @pytest.mark.asyncio
    async def test_get_customer(self, wrapper):
        """Test getting customer document."""
        wrapper.redis.execute_command = AsyncMock(
            return_value='{"crid": "CRID-001", "name": "John"}'
        )

        result = await wrapper.get_customer("CRID-001")

        assert result["crid"] == "CRID-001"


class TestModels:
    """Tests for data models."""

    def test_query_event_to_dict(self):
        """Test QueryEvent serialization."""
        from icda.redis_stack.models import QueryEvent

        event = QueryEvent(
            query="test query",
            response_preview="response" * 100,  # Long response
            latency_ms=100,
            agent_chain=["intent", "search", "nova"],
            cache_hit=True,
            trace_id="trace-123",
        )

        result = event.to_dict()

        assert result["query"] == "test query"
        assert len(result["response_preview"]) <= 200
        assert result["agent_chain"] == "intent,search,nova"
        assert result["cache_hit"] == "1"

    def test_customer_event_to_dict(self):
        """Test CustomerEvent serialization."""
        from icda.redis_stack.models import CustomerEvent

        event = CustomerEvent(
            crid="CRID-001",
            action="updated",
            changes={"status": "INACTIVE"},
        )

        result = event.to_dict()

        assert result["crid"] == "CRID-001"
        assert result["action"] == "updated"
        assert json.loads(result["changes"]) == {"status": "INACTIVE"}

    def test_index_progress_to_dict(self):
        """Test IndexProgress serialization."""
        from icda.redis_stack.models import IndexProgress

        progress = IndexProgress(
            index_name="customers",
            indexed=500,
            total=1000,
            errors=5,
            status="running",
            elapsed_seconds=30.5,
            rate_per_second=16.4,
        )

        result = progress.to_dict()

        assert result["index_name"] == "customers"
        assert result["percent"] == 50.0
        assert result["status"] == "running"
