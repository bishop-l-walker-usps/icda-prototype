"""
Session Manager - Maintains conversation history for context continuity.
Stores message history in Redis with TTL, falls back to in-memory.
"""

import json
import uuid
from time import time
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import RedisCache


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time)

    def to_bedrock(self) -> dict:
        """Convert to Bedrock converse format."""
        return {"role": self.role, "content": [{"text": self.content}]}


@dataclass
class Session:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
    user_id: str | None = None  # Link to authenticated user (email hash)
    user_email: str | None = None  # Email for display purposes

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self.updated_at = time()

    def get_history(self, max_messages: int = 20) -> list[dict]:
        """Get recent message history in Bedrock format."""
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [msg.to_bedrock() for msg in recent]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "messages": [asdict(m) for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "user_id": self.user_id,
            "user_email": self.user_email,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            session_id=data["session_id"],
            messages=[Message(**m) for m in data.get("messages", [])],
            created_at=data.get("created_at", time()),
            updated_at=data.get("updated_at", time()),
            user_id=data.get("user_id"),
            user_email=data.get("user_email"),
        )


class SessionManager:
    """Manages conversation sessions with Redis backend."""

    __slots__ = ("cache", "ttl", "_fallback")

    def __init__(self, cache: "RedisCache", ttl: int = 3600):
        self.cache = cache
        self.ttl = ttl  # 1 hour default
        self._fallback: dict[str, Session] = {}

    def _key(self, session_id: str) -> str:
        return f"icda:session:{session_id}"

    def _user_sessions_key(self, user_id: str) -> str:
        """Key for tracking a user's active sessions."""
        return f"icda:user_sessions:{user_id}"

    async def get(self, session_id: str | None) -> Session:
        """Get or create a session."""
        if not session_id:
            return Session(session_id=str(uuid.uuid4()))

        # Try Redis
        if self.cache.available:
            if data := await self.cache.client.get(self._key(session_id)):
                return Session.from_dict(json.loads(data))
        # Try fallback
        elif session_id in self._fallback:
            return self._fallback[session_id]

        # New session with provided ID
        return Session(session_id=session_id)

    async def get_or_create_for_user(
        self,
        session_id: str | None,
        user_id: str,
        user_email: str,
    ) -> Session:
        """Get or create a session linked to an authenticated user."""
        session = await self.get(session_id)

        # Link to user if not already linked
        if not session.user_id:
            session.user_id = user_id
            session.user_email = user_email

        # Track this session for the user
        if self.cache.available:
            await self.cache.client.sadd(self._user_sessions_key(user_id), session.session_id)
            await self.cache.client.expire(self._user_sessions_key(user_id), self.ttl)

        return session

    async def get_user_sessions(self, user_id: str) -> list[Session]:
        """Get all active sessions for a user."""
        sessions = []

        if self.cache.available:
            session_ids = await self.cache.client.smembers(self._user_sessions_key(user_id))
            for session_id in session_ids:
                session = await self.get(session_id)
                if session and session.messages:  # Only include non-empty sessions
                    sessions.append(session)
        else:
            for session in self._fallback.values():
                if session.user_id == user_id and session.messages:
                    sessions.append(session)

        # Sort by last update (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    async def save(self, session: Session) -> None:
        """Persist session to storage."""
        data = json.dumps(session.to_dict())

        if self.cache.available:
            await self.cache.client.setex(self._key(session.session_id), self.ttl, data)
            # Track for user if linked
            if session.user_id:
                await self.cache.client.sadd(self._user_sessions_key(session.user_id), session.session_id)
                await self.cache.client.expire(self._user_sessions_key(session.user_id), self.ttl)
        else:
            self._fallback[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        """Delete a session."""
        # Get session to find user_id for cleanup
        session = await self.get(session_id)

        if self.cache.available:
            await self.cache.client.delete(self._key(session_id))
            # Remove from user's session list
            if session and session.user_id:
                await self.cache.client.srem(self._user_sessions_key(session.user_id), session_id)
        else:
            self._fallback.pop(session_id, None)

    async def delete_user_sessions(self, user_id: str) -> int:
        """Delete all sessions for a user. Returns count deleted."""
        count = 0

        if self.cache.available:
            session_ids = await self.cache.client.smembers(self._user_sessions_key(user_id))
            for session_id in session_ids:
                await self.cache.client.delete(self._key(session_id))
                count += 1
            await self.cache.client.delete(self._user_sessions_key(user_id))
        else:
            to_delete = [sid for sid, s in self._fallback.items() if s.user_id == user_id]
            for sid in to_delete:
                del self._fallback[sid]
                count += 1

        return count

    async def clear_all(self) -> int:
        """Clear all sessions. Returns count deleted."""
        if self.cache.available:
            keys = []
            async for key in self.cache.client.scan_iter(match="icda:session:*"):
                keys.append(key)
            # Also clear user session mappings
            async for key in self.cache.client.scan_iter(match="icda:user_sessions:*"):
                keys.append(key)
            if keys:
                await self.cache.client.delete(*keys)
            return len(keys)
        else:
            count = len(self._fallback)
            self._fallback.clear()
            return count
