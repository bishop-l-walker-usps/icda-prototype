"""Download Token Manager - Secure tokens for large result downloads.

When query results exceed the pagination threshold (default 50),
this module generates download tokens that allow users to retrieve
the full result set via a dedicated download endpoint.

Features:
- Secure token generation using secrets
- Time-limited token validity (15 minutes default)
- In-memory storage with Redis fallback
- Automatic cleanup of expired tokens
"""

import secrets
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import cfg


@dataclass
class DownloadSession:
    """Stores full results for deferred download.

    Attributes:
        token: Unique download token.
        results: Full result set.
        query: Original query string.
        created_at: Creation timestamp.
        expires_at: Expiration timestamp.
        total_count: Total number of results.
    """
    token: str
    results: list[dict[str, Any]]
    query: str
    created_at: float
    expires_at: float
    total_count: int


class DownloadTokenManager:
    """Manages download tokens for large result sets.

    When search results exceed the pagination threshold, this manager:
    1. Stores the full results with a secure token
    2. Returns a preview of the first N results
    3. Provides a token for downloading the full dataset
    """

    __slots__ = (
        "_sessions",
        "_cache",
        "_pagination_threshold",
        "_preview_size",
        "_token_ttl",
    )

    def __init__(
        self,
        cache=None,
        pagination_threshold: int | None = None,
        preview_size: int | None = None,
        token_ttl: int = 900,
    ):
        """Initialize DownloadTokenManager.

        Args:
            cache: Optional Redis cache for persistence.
            pagination_threshold: Results above this trigger pagination.
            preview_size: Number of results to show as preview.
            token_ttl: Token time-to-live in seconds (default 15 min).
        """
        self._sessions: dict[str, DownloadSession] = {}
        self._cache = cache
        self._pagination_threshold = pagination_threshold or cfg.pagination_threshold
        self._preview_size = preview_size or cfg.pagination_preview_size
        self._token_ttl = token_ttl

    def should_paginate(self, total_results: int) -> bool:
        """Check if results should be paginated.

        Args:
            total_results: Total number of results.

        Returns:
            True if results exceed pagination threshold.
        """
        return total_results > self._pagination_threshold

    def create_download_token(
        self,
        results: list[dict[str, Any]],
        query: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Create download token and return preview with pagination info.

        Args:
            results: Full result set.
            query: Original query string.

        Returns:
            Tuple of (preview_results, pagination_info_dict).
        """
        total = len(results)

        # No pagination needed
        if not self.should_paginate(total):
            return results, {
                "total_count": total,
                "returned_count": total,
                "has_more": False,
                "suggest_download": False,
                "preview_size": total,
            }

        # Generate secure token
        token = secrets.token_urlsafe(32)
        now = time.time()
        expires_at = now + self._token_ttl

        # Store full results
        session = DownloadSession(
            token=token,
            results=results,
            query=query,
            created_at=now,
            expires_at=expires_at,
            total_count=total,
        )
        self._sessions[token] = session

        # Also store in Redis if available
        if self._cache and self._cache.available:
            try:
                import asyncio
                # Run the async cache set in sync context
                asyncio.create_task(
                    self._cache.client.setex(
                        f"icda:download:{token}",
                        self._token_ttl,
                        json.dumps({
                            "query": query,
                            "total": total,
                            "data": results,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    )
                )
            except Exception:
                pass  # Fall back to in-memory only

        # Clean expired sessions
        self._cleanup_expired()

        # Return preview
        preview = results[:self._preview_size]
        expires_iso = datetime.fromtimestamp(
            expires_at, tz=timezone.utc
        ).isoformat()

        pagination_info = {
            "total_count": total,
            "returned_count": len(preview),
            "has_more": True,
            "suggest_download": True,
            "download_token": token,
            "download_expires_at": expires_iso,
            "preview_size": self._preview_size,
        }

        return preview, pagination_info

    def get_full_results(self, token: str) -> dict[str, Any] | None:
        """Retrieve full results by download token.

        Args:
            token: Download token.

        Returns:
            Dict with full results or None if token invalid/expired.
        """
        # Check in-memory first
        session = self._sessions.get(token)
        if session:
            if time.time() > session.expires_at:
                del self._sessions[token]
                return None

            return {
                "success": True,
                "query": session.query,
                "total": session.total_count,
                "data": session.results,
                "generated_at": datetime.fromtimestamp(
                    session.created_at, tz=timezone.utc
                ).isoformat(),
            }

        return None

    async def get_full_results_async(self, token: str) -> dict[str, Any] | None:
        """Retrieve full results by download token (async version).

        Args:
            token: Download token.

        Returns:
            Dict with full results or None if token invalid/expired.
        """
        # Check in-memory first
        result = self.get_full_results(token)
        if result:
            return result

        # Check Redis if available
        if self._cache and self._cache.available:
            try:
                data = await self._cache.get(f"icda:download:{token}")
                if data:
                    parsed = json.loads(data)
                    return {
                        "success": True,
                        "query": parsed["query"],
                        "total": parsed["total"],
                        "data": parsed["data"],
                        "generated_at": parsed["timestamp"],
                    }
            except Exception:
                pass

        return None

    def invalidate_token(self, token: str) -> bool:
        """Invalidate a download token.

        Args:
            token: Token to invalidate.

        Returns:
            True if token was found and invalidated.
        """
        if token in self._sessions:
            del self._sessions[token]
            return True
        return False

    def _cleanup_expired(self) -> None:
        """Remove expired sessions from in-memory storage."""
        now = time.time()
        expired = [k for k, v in self._sessions.items() if now > v.expires_at]
        for k in expired:
            del self._sessions[k]

    @property
    def active_tokens(self) -> int:
        """Get count of active download tokens."""
        self._cleanup_expired()
        return len(self._sessions)

    @property
    def pagination_threshold(self) -> int:
        """Get the pagination threshold."""
        return self._pagination_threshold

    @property
    def preview_size(self) -> int:
        """Get the preview size."""
        return self._preview_size
