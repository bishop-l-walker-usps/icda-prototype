"""RedisJSON wrapper for structured document storage.

Provides:
- JSON document storage with JSONPath queries
- Partial updates without full replacement
- Atomic array operations
- Nested object manipulation
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RedisJSONWrapper:
    """Wrapper for RedisJSON operations.

    Key naming convention:
        customer:{crid}    - Customer JSON documents
        session:{id}       - Session JSON documents
        config:{key}       - Configuration documents
    """

    def __init__(self, redis):
        self.redis = redis

    # =========================================================================
    # Basic Operations
    # =========================================================================

    async def set(
        self,
        key: str,
        path: str,
        value: Any,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set JSON value at path.

        Args:
            key: Redis key
            path: JSONPath ($ for root)
            value: Value to set (will be JSON encoded)
            nx: Only set if key doesn't exist
            xx: Only set if key exists

        Returns:
            True if set successfully
        """
        try:
            args = [key, path, json.dumps(value)]
            if nx:
                args.append("NX")
            elif xx:
                args.append("XX")

            result = await self.redis.execute_command("JSON.SET", *args)
            return result == "OK"
        except Exception as e:
            logger.debug(f"JSON.SET failed: {e}")
            return False

    async def get(
        self,
        key: str,
        *paths: str,
    ) -> Any | None:
        """Get JSON value at path(s).

        Args:
            key: Redis key
            paths: JSONPaths to retrieve (default: root $)

        Returns:
            Parsed JSON value or None
        """
        try:
            if not paths:
                paths = ("$",)

            result = await self.redis.execute_command("JSON.GET", key, *paths)

            if result is None:
                return None

            return json.loads(result)
        except Exception as e:
            logger.debug(f"JSON.GET failed: {e}")
            return None

    async def delete(self, key: str, path: str = "$") -> int:
        """Delete JSON value at path.

        Args:
            key: Redis key
            path: JSONPath to delete

        Returns:
            Number of paths deleted
        """
        try:
            result = await self.redis.execute_command("JSON.DEL", key, path)
            return int(result) if result else 0
        except Exception as e:
            logger.debug(f"JSON.DEL failed: {e}")
            return 0

    async def exists(self, key: str, path: str = "$") -> bool:
        """Check if path exists in JSON document."""
        try:
            result = await self.redis.execute_command("JSON.TYPE", key, path)
            return result is not None
        except Exception:
            return False

    # =========================================================================
    # Numeric Operations
    # =========================================================================

    async def incr(self, key: str, path: str, value: int | float = 1) -> float | None:
        """Increment numeric value at path.

        Args:
            key: Redis key
            path: JSONPath to numeric value
            value: Amount to increment

        Returns:
            New value or None on error
        """
        try:
            result = await self.redis.execute_command(
                "JSON.NUMINCRBY", key, path, value
            )
            if result:
                return float(json.loads(result)[0])
            return None
        except Exception as e:
            logger.debug(f"JSON.NUMINCRBY failed: {e}")
            return None

    async def multiply(self, key: str, path: str, value: int | float) -> float | None:
        """Multiply numeric value at path.

        Args:
            key: Redis key
            path: JSONPath to numeric value
            value: Multiplier

        Returns:
            New value or None on error
        """
        try:
            result = await self.redis.execute_command(
                "JSON.NUMMULTBY", key, path, value
            )
            if result:
                return float(json.loads(result)[0])
            return None
        except Exception as e:
            logger.debug(f"JSON.NUMMULTBY failed: {e}")
            return None

    # =========================================================================
    # String Operations
    # =========================================================================

    async def str_append(self, key: str, path: str, value: str) -> int | None:
        """Append to string value at path.

        Returns:
            New string length or None
        """
        try:
            result = await self.redis.execute_command(
                "JSON.STRAPPEND", key, path, json.dumps(value)
            )
            return int(result[0]) if result else None
        except Exception as e:
            logger.debug(f"JSON.STRAPPEND failed: {e}")
            return None

    async def str_len(self, key: str, path: str = "$") -> int | None:
        """Get string length at path."""
        try:
            result = await self.redis.execute_command("JSON.STRLEN", key, path)
            return int(result[0]) if result else None
        except Exception:
            return None

    # =========================================================================
    # Array Operations
    # =========================================================================

    async def arr_append(self, key: str, path: str, *values: Any) -> int | None:
        """Append values to array at path.

        Returns:
            New array length or None
        """
        try:
            json_values = [json.dumps(v) for v in values]
            result = await self.redis.execute_command(
                "JSON.ARRAPPEND", key, path, *json_values
            )
            return int(result[0]) if result else None
        except Exception as e:
            logger.debug(f"JSON.ARRAPPEND failed: {e}")
            return None

    async def arr_insert(
        self,
        key: str,
        path: str,
        index: int,
        *values: Any,
    ) -> int | None:
        """Insert values at index in array.

        Returns:
            New array length or None
        """
        try:
            json_values = [json.dumps(v) for v in values]
            result = await self.redis.execute_command(
                "JSON.ARRINSERT", key, path, index, *json_values
            )
            return int(result[0]) if result else None
        except Exception as e:
            logger.debug(f"JSON.ARRINSERT failed: {e}")
            return None

    async def arr_pop(
        self,
        key: str,
        path: str = "$",
        index: int = -1,
    ) -> Any | None:
        """Remove and return element from array.

        Args:
            key: Redis key
            path: JSONPath to array
            index: Index to pop (-1 for last)

        Returns:
            Popped value or None
        """
        try:
            result = await self.redis.execute_command(
                "JSON.ARRPOP", key, path, index
            )
            if result and result[0]:
                return json.loads(result[0])
            return None
        except Exception as e:
            logger.debug(f"JSON.ARRPOP failed: {e}")
            return None

    async def arr_len(self, key: str, path: str = "$") -> int | None:
        """Get array length at path."""
        try:
            result = await self.redis.execute_command("JSON.ARRLEN", key, path)
            return int(result[0]) if result else None
        except Exception:
            return None

    async def arr_trim(self, key: str, path: str, start: int, stop: int) -> int | None:
        """Trim array to [start, stop] range.

        Returns:
            New array length or None
        """
        try:
            result = await self.redis.execute_command(
                "JSON.ARRTRIM", key, path, start, stop
            )
            return int(result[0]) if result else None
        except Exception as e:
            logger.debug(f"JSON.ARRTRIM failed: {e}")
            return None

    # =========================================================================
    # Customer Document Operations
    # =========================================================================

    async def get_customer(self, crid: str) -> dict | None:
        """Get customer JSON document."""
        return await self.get(f"customer:{crid}")

    async def set_customer(self, crid: str, data: dict) -> bool:
        """Set customer JSON document."""
        return await self.set(f"customer:{crid}", "$", data)

    async def update_customer_field(
        self,
        crid: str,
        field: str,
        value: Any,
    ) -> bool:
        """Update single field in customer document."""
        return await self.set(f"customer:{crid}", f"$.{field}", value, xx=True)

    async def add_customer_history(
        self,
        crid: str,
        event: dict,
    ) -> int | None:
        """Append event to customer history array."""
        key = f"customer:{crid}"

        # Ensure history array exists
        history = await self.get(key, "$.history")
        if history is None or (isinstance(history, list) and len(history) == 0):
            await self.set(key, "$.history", [], xx=True)

        return await self.arr_append(key, "$.history", event)

    async def increment_customer_move_count(self, crid: str) -> float | None:
        """Increment customer move count."""
        return await self.incr(f"customer:{crid}", "$.move_count")

    async def get_customer_history(self, crid: str, limit: int = 10) -> list:
        """Get recent customer history events."""
        result = await self.get(f"customer:{crid}", "$.history")
        if result and isinstance(result, list) and len(result) > 0:
            history = result[0] if isinstance(result[0], list) else result
            return history[-limit:] if len(history) > limit else history
        return []

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def mget(self, *keys: str, path: str = "$") -> list[Any]:
        """Get multiple JSON documents.

        Returns:
            List of parsed values (None for missing keys)
        """
        try:
            result = await self.redis.execute_command("JSON.MGET", *keys, path)
            return [json.loads(r) if r else None for r in result]
        except Exception as e:
            logger.debug(f"JSON.MGET failed: {e}")
            return [None] * len(keys)

    async def get_keys_by_pattern(self, pattern: str) -> list[str]:
        """Get all keys matching pattern."""
        try:
            cursor = 0
            keys = []
            while True:
                cursor, batch = await self.redis.scan(cursor, match=pattern, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break
            return keys
        except Exception as e:
            logger.debug(f"Key scan failed: {e}")
            return []

    # =========================================================================
    # Document Info
    # =========================================================================

    async def get_type(self, key: str, path: str = "$") -> str | None:
        """Get JSON type at path."""
        try:
            result = await self.redis.execute_command("JSON.TYPE", key, path)
            return result[0] if result else None
        except Exception:
            return None

    async def get_memory(self, key: str, path: str = "$") -> int | None:
        """Get memory usage of JSON document."""
        try:
            result = await self.redis.execute_command("JSON.DEBUG", "MEMORY", key, path)
            return int(result) if result else None
        except Exception:
            return None
