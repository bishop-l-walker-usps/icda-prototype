"""Redis Pub/Sub manager for real-time notifications.

Channels:
- icda:index:progress    - Indexing progress updates
- icda:cache:invalidate  - Cache invalidation events
- icda:health:status     - Health status broadcasts
- icda:admin:alerts      - Admin notifications
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Callable, Any

from .models import IndexProgress

logger = logging.getLogger(__name__)


class RedisPubSubManager:
    """Manager for Redis Pub/Sub operations.

    Provides:
    - Channel publishing
    - Subscription management
    - SSE stream generation
    - Health broadcasts
    """

    # Channel names
    CHANNEL_INDEX_PROGRESS = "icda:index:progress"
    CHANNEL_CACHE_INVALIDATE = "icda:cache:invalidate"
    CHANNEL_HEALTH = "icda:health:status"
    CHANNEL_ADMIN = "icda:admin:alerts"

    def __init__(self, redis):
        self.redis = redis
        self._pubsub = None
        self._subscriptions: dict[str, list[Callable]] = {}
        self._running = False

    async def publish(self, channel: str, message: dict | str) -> int:
        """Publish message to channel.

        Args:
            channel: Channel name
            message: Message dict (will be JSON encoded) or string

        Returns:
            Number of subscribers that received the message
        """
        try:
            if isinstance(message, dict):
                message = json.dumps(message)
            return await self.redis.publish(channel, message)
        except Exception as e:
            logger.debug(f"Pub/Sub publish failed: {e}")
            return 0

    async def publish_index_progress(self, progress: IndexProgress) -> int:
        """Publish indexing progress update."""
        return await self.publish(
            self.CHANNEL_INDEX_PROGRESS,
            progress.to_dict()
        )

    async def publish_cache_invalidate(self, keys: list[str], reason: str = "") -> int:
        """Publish cache invalidation event."""
        return await self.publish(
            self.CHANNEL_CACHE_INVALIDATE,
            {"keys": keys, "reason": reason}
        )

    async def publish_health(self, status: dict) -> int:
        """Publish health status update."""
        return await self.publish(self.CHANNEL_HEALTH, status)

    async def publish_admin_alert(
        self,
        level: str,
        message: str,
        details: dict | None = None
    ) -> int:
        """Publish admin alert.

        Args:
            level: Alert level (info, warning, error, critical)
            message: Alert message
            details: Additional details
        """
        return await self.publish(
            self.CHANNEL_ADMIN,
            {
                "level": level,
                "message": message,
                "details": details or {},
            }
        )

    # =========================================================================
    # Subscription
    # =========================================================================

    async def subscribe(self, *channels: str) -> None:
        """Subscribe to channels."""
        if not self._pubsub:
            self._pubsub = self.redis.pubsub()

        await self._pubsub.subscribe(*channels)
        logger.debug(f"Subscribed to channels: {channels}")

    async def unsubscribe(self, *channels: str) -> None:
        """Unsubscribe from channels."""
        if self._pubsub:
            await self._pubsub.unsubscribe(*channels)

    async def listen(self) -> AsyncIterator[dict]:
        """Async iterator for incoming messages.

        Yields:
            Message dicts with 'channel', 'data', 'type' keys
        """
        if not self._pubsub:
            return

        self._running = True
        try:
            while self._running:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )
                if message:
                    # Decode JSON data if possible
                    data = message.get("data")
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except json.JSONDecodeError:
                            pass
                        message["data"] = data
                    yield message
                else:
                    await asyncio.sleep(0.1)
        finally:
            self._running = False

    async def close(self) -> None:
        """Close pubsub connection."""
        self._running = False
        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None

    # =========================================================================
    # SSE Stream Generator
    # =========================================================================

    async def sse_stream(
        self,
        channels: list[str],
        include_heartbeat: bool = True,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[str]:
        """Generate Server-Sent Events stream.

        Args:
            channels: Channels to subscribe to
            include_heartbeat: Whether to send periodic heartbeats
            heartbeat_interval: Seconds between heartbeats

        Yields:
            SSE formatted strings
        """
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(*channels)

        last_heartbeat = asyncio.get_event_loop().time()

        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )

                if message:
                    channel = message.get("channel", "")
                    data = message.get("data", "")

                    # Parse JSON if possible
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except json.JSONDecodeError:
                            pass

                    # Format as SSE
                    event_type = channel.split(":")[-1] if ":" in channel else channel
                    event_data = json.dumps(data) if isinstance(data, dict) else str(data)

                    yield f"event: {event_type}\ndata: {event_data}\n\n"

                # Send heartbeat
                if include_heartbeat:
                    now = asyncio.get_event_loop().time()
                    if now - last_heartbeat > heartbeat_interval:
                        yield f": heartbeat\n\n"
                        last_heartbeat = now

                await asyncio.sleep(0.1)

        finally:
            await pubsub.unsubscribe(*channels)
            await pubsub.close()

    # =========================================================================
    # Health Broadcast Loop
    # =========================================================================

    async def health_broadcast_loop(
        self,
        interval: float = 30.0,
        health_check_fn: Callable | None = None,
    ) -> None:
        """Background task to broadcast health status periodically.

        Args:
            interval: Seconds between broadcasts
            health_check_fn: Async function that returns health dict
        """
        while True:
            try:
                if health_check_fn:
                    status = await health_check_fn()
                else:
                    status = {"status": "ok", "timestamp": asyncio.get_event_loop().time()}

                await self.publish_health(status)
            except Exception as e:
                logger.debug(f"Health broadcast failed: {e}")

            await asyncio.sleep(interval)
