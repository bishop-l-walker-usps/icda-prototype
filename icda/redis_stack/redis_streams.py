"""Redis Streams manager for event sourcing and audit trails.

Streams:
- stream:customer:changes  - Customer CRUD events
- stream:query:audit       - Query execution audit trail
- stream:agent:execution   - Agent pipeline traces
- stream:errors            - Error events for alerting
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from .models import QueryEvent, CustomerEvent

logger = logging.getLogger(__name__)


class RedisStreamsManager:
    """Manager for Redis Streams event sourcing.

    Provides:
    - Event publishing to streams
    - Stream reading with pagination
    - Consumer group management
    - Audit trail queries
    """

    # Stream names
    STREAM_CUSTOMER = "stream:customer:changes"
    STREAM_QUERY = "stream:query:audit"
    STREAM_AGENT = "stream:agent:execution"
    STREAM_ERRORS = "stream:errors"

    # Stream settings
    MAX_LEN = 100000  # Max entries per stream
    TRIM_APPROX = True  # Use approximate trimming for performance

    def __init__(self, redis):
        self.redis = redis

    async def ensure_streams(self) -> None:
        """Create streams and consumer groups."""
        streams = [
            self.STREAM_CUSTOMER,
            self.STREAM_QUERY,
            self.STREAM_AGENT,
            self.STREAM_ERRORS,
        ]

        for stream in streams:
            await self._ensure_stream(stream)

        # Create default consumer groups
        await self._ensure_consumer_group(self.STREAM_QUERY, "analytics-group")
        await self._ensure_consumer_group(self.STREAM_ERRORS, "alerting-group")

    async def _ensure_stream(self, stream: str) -> bool:
        """Ensure stream exists by adding a dummy entry if needed."""
        try:
            # Check if stream exists
            exists = await self.redis.exists(stream)
            if not exists:
                # Create with dummy entry, then trim
                await self.redis.xadd(
                    stream,
                    {"_init": "1"},
                    maxlen=1,
                )
                logger.debug(f"Created stream: {stream}")
            return True
        except Exception as e:
            logger.warning(f"Failed to ensure stream {stream}: {e}")
            return False

    async def _ensure_consumer_group(
        self,
        stream: str,
        group: str,
        start_id: str = "$"
    ) -> bool:
        """Create consumer group if not exists."""
        try:
            await self.redis.xgroup_create(
                stream,
                group,
                id=start_id,
                mkstream=True,
            )
            logger.debug(f"Created consumer group: {group} on {stream}")
            return True
        except Exception as e:
            if "BUSYGROUP" in str(e):
                return True  # Already exists
            logger.warning(f"Failed to create consumer group {group}: {e}")
            return False

    # =========================================================================
    # Event Publishing
    # =========================================================================

    async def _add_event(
        self,
        stream: str,
        data: dict[str, Any],
        maxlen: int | None = None,
    ) -> str | None:
        """Add event to stream.

        Returns:
            Event ID or None on failure
        """
        try:
            # Convert all values to strings (Redis Streams requirement)
            str_data = {k: str(v) if not isinstance(v, str) else v for k, v in data.items()}

            event_id = await self.redis.xadd(
                stream,
                str_data,
                maxlen=maxlen or self.MAX_LEN,
                approximate=self.TRIM_APPROX,
            )
            return event_id
        except Exception as e:
            logger.debug(f"Failed to add event to {stream}: {e}")
            return None

    async def add_query_event(self, event: QueryEvent) -> str | None:
        """Record query to audit stream."""
        return await self._add_event(self.STREAM_QUERY, event.to_dict())

    async def add_customer_event(self, event: CustomerEvent) -> str | None:
        """Record customer change event."""
        return await self._add_event(self.STREAM_CUSTOMER, event.to_dict())

    async def add_agent_event(
        self,
        trace_id: str,
        agent: str,
        action: str,
        latency_ms: int,
        input_preview: str = "",
        output_preview: str = "",
        success: bool = True,
        error: str | None = None,
    ) -> str | None:
        """Record agent execution event."""
        return await self._add_event(
            self.STREAM_AGENT,
            {
                "trace_id": trace_id,
                "agent": agent,
                "action": action,
                "latency_ms": str(latency_ms),
                "input_preview": input_preview[:200] if input_preview else "",
                "output_preview": output_preview[:200] if output_preview else "",
                "success": "1" if success else "0",
                "error": error or "",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    async def add_error_event(
        self,
        error_type: str,
        message: str,
        stack_trace: str = "",
        context: dict | None = None,
    ) -> str | None:
        """Record error event for alerting."""
        return await self._add_event(
            self.STREAM_ERRORS,
            {
                "error_type": error_type,
                "message": message,
                "stack_trace": stack_trace[:1000] if stack_trace else "",
                "context": json.dumps(context or {}),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    # =========================================================================
    # Event Reading
    # =========================================================================

    async def read_stream(
        self,
        stream: str,
        count: int = 100,
        start_id: str = "-",
        end_id: str = "+",
    ) -> list[dict]:
        """Read events from stream.

        Args:
            stream: Stream name
            count: Max events to return
            start_id: Start ID (- for oldest)
            end_id: End ID (+ for newest)

        Returns:
            List of event dicts with 'id' and 'data' keys
        """
        try:
            entries = await self.redis.xrange(
                stream,
                min=start_id,
                max=end_id,
                count=count,
            )

            results = []
            for entry_id, data in entries:
                # Parse JSON fields
                parsed = {"id": entry_id}
                for k, v in data.items():
                    if k in ("changes", "context") and v:
                        try:
                            parsed[k] = json.loads(v)
                        except json.JSONDecodeError:
                            parsed[k] = v
                    else:
                        parsed[k] = v
                results.append(parsed)

            return results

        except Exception as e:
            logger.warning(f"Failed to read stream {stream}: {e}")
            return []

    async def read_recent(
        self,
        stream: str,
        count: int = 100,
    ) -> list[dict]:
        """Read most recent events from stream."""
        try:
            entries = await self.redis.xrevrange(
                stream,
                count=count,
            )

            results = []
            for entry_id, data in entries:
                parsed = {"id": entry_id}
                for k, v in data.items():
                    if k in ("changes", "context") and v:
                        try:
                            parsed[k] = json.loads(v)
                        except json.JSONDecodeError:
                            parsed[k] = v
                    else:
                        parsed[k] = v
                results.append(parsed)

            return results

        except Exception as e:
            logger.warning(f"Failed to read recent from {stream}: {e}")
            return []

    async def get_query_audit(
        self,
        limit: int = 100,
        from_id: str | None = None,
    ) -> list[dict]:
        """Get query audit trail."""
        if from_id:
            return await self.read_stream(self.STREAM_QUERY, count=limit, start_id=from_id)
        return await self.read_recent(self.STREAM_QUERY, count=limit)

    async def get_customer_changes(
        self,
        limit: int = 100,
        crid: str | None = None,
    ) -> list[dict]:
        """Get customer change events."""
        events = await self.read_recent(self.STREAM_CUSTOMER, count=limit * 2)

        if crid:
            events = [e for e in events if e.get("crid") == crid]

        return events[:limit]

    async def get_agent_trace(self, trace_id: str) -> list[dict]:
        """Get all agent events for a trace ID."""
        # Read a large batch and filter (streams don't support field queries)
        events = await self.read_recent(self.STREAM_AGENT, count=1000)
        return [e for e in events if e.get("trace_id") == trace_id]

    async def get_errors(
        self,
        limit: int = 100,
        error_type: str | None = None,
    ) -> list[dict]:
        """Get recent errors."""
        events = await self.read_recent(self.STREAM_ERRORS, count=limit * 2)

        if error_type:
            events = [e for e in events if e.get("error_type") == error_type]

        return events[:limit]

    # =========================================================================
    # Consumer Groups
    # =========================================================================

    async def read_group(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int | None = None,
    ) -> list[dict]:
        """Read events as consumer group member.

        Args:
            stream: Stream name
            group: Consumer group name
            consumer: Consumer name
            count: Max events to read
            block_ms: Block for this many ms (None for non-blocking)

        Returns:
            List of event dicts
        """
        try:
            entries = await self.redis.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream: ">"},
                count=count,
                block=block_ms,
            )

            if not entries:
                return []

            results = []
            for stream_name, stream_entries in entries:
                for entry_id, data in stream_entries:
                    results.append({"id": entry_id, "stream": stream_name, **data})

            return results

        except Exception as e:
            logger.warning(f"Failed to read group {group} from {stream}: {e}")
            return []

    async def ack_event(self, stream: str, group: str, event_id: str) -> bool:
        """Acknowledge event processing."""
        try:
            await self.redis.xack(stream, group, event_id)
            return True
        except Exception as e:
            logger.warning(f"Failed to ack event {event_id}: {e}")
            return False

    # =========================================================================
    # Maintenance
    # =========================================================================

    async def get_stream_info(self, stream: str) -> dict:
        """Get stream statistics."""
        try:
            info = await self.redis.xinfo_stream(stream)
            return {
                "length": info.get("length", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": info.get("groups", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    async def trim_stream(self, stream: str, maxlen: int | None = None) -> int:
        """Manually trim stream to max length."""
        try:
            return await self.redis.xtrim(
                stream,
                maxlen=maxlen or self.MAX_LEN,
                approximate=self.TRIM_APPROX,
            )
        except Exception as e:
            logger.warning(f"Failed to trim stream {stream}: {e}")
            return 0

    async def get_all_stats(self) -> dict:
        """Get stats for all streams."""
        return {
            "customer": await self.get_stream_info(self.STREAM_CUSTOMER),
            "query": await self.get_stream_info(self.STREAM_QUERY),
            "agent": await self.get_stream_info(self.STREAM_AGENT),
            "errors": await self.get_stream_info(self.STREAM_ERRORS),
        }
