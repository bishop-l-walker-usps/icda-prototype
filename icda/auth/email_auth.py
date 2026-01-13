"""
Simple Email-Based Authentication

Users authenticate by providing their email address.
Sessions are stored in Redis with configurable TTL.
"""

import re
import secrets
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from os import getenv
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

# Email validation regex
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)

# Allowed email domains (empty = allow all)
ALLOWED_DOMAINS: set[str] = set()


@dataclass
class User:
    """Authenticated user context."""
    email: str
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """User ID derived from email hash."""
        return hashlib.sha256(self.email.lower().encode()).hexdigest()[:16]

    @property
    def username(self) -> str:
        """Username is email local part."""
        return self.email.split("@")[0]

    @property
    def domain(self) -> str:
        """Email domain."""
        return self.email.split("@")[1] if "@" in self.email else ""

    def to_dict(self) -> dict:
        """Serialize user for storage."""
        return {
            "email": self.email,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Deserialize user from storage."""
        return cls(
            email=data["email"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EmailAuthConfig:
    """Email authentication configuration."""
    enabled: bool = field(default_factory=lambda: getenv("AUTH_ENABLED", "false").lower() == "true")
    # Session TTL in seconds (default 24 hours)
    session_ttl: int = field(default_factory=lambda: int(getenv("AUTH_SESSION_TTL", "86400")))
    # Allowed email domains (comma-separated, empty = allow all)
    allowed_domains: str = field(default_factory=lambda: getenv("AUTH_ALLOWED_DOMAINS", ""))
    # Require email verification (future: send verification email)
    require_verification: bool = field(default_factory=lambda: getenv("AUTH_REQUIRE_VERIFICATION", "false").lower() == "true")
    # Allow anonymous access when auth is enabled
    allow_anonymous: bool = field(default_factory=lambda: getenv("AUTH_ALLOW_ANONYMOUS", "true").lower() == "true")

    def get_allowed_domains(self) -> set[str]:
        """Get set of allowed email domains."""
        if not self.allowed_domains:
            return set()
        return {d.strip().lower() for d in self.allowed_domains.split(",") if d.strip()}


def validate_email(email: str, config: Optional[EmailAuthConfig] = None) -> tuple[bool, str]:
    """
    Validate email address format and domain.

    Returns:
        (is_valid, error_message)
    """
    if not email:
        return False, "Email is required"

    email = email.strip().lower()

    # Check format
    if not EMAIL_REGEX.match(email):
        return False, "Invalid email format"

    # Check domain restrictions
    if config:
        allowed = config.get_allowed_domains()
        if allowed:
            domain = email.split("@")[1]
            if domain not in allowed:
                return False, f"Email domain not allowed. Allowed: {', '.join(sorted(allowed))}"

    return True, ""


def create_session_token() -> str:
    """Generate a secure session token."""
    return secrets.token_urlsafe(32)


class EmailAuth:
    """
    Simple email-based authentication handler.

    Uses Redis for session storage.
    """

    def __init__(self, config: Optional[EmailAuthConfig] = None, cache=None):
        self.config = config or EmailAuthConfig()
        self._cache = cache  # Redis cache instance
        self._available = False

    @property
    def available(self) -> bool:
        """Check if auth is enabled and cache is available."""
        return self.config.enabled and self._cache is not None and self._cache.available

    def set_cache(self, cache) -> None:
        """Set Redis cache for session storage."""
        self._cache = cache
        self._available = cache is not None and cache.available

    async def login(self, email: str) -> tuple[Optional[User], str]:
        """
        Login/register user by email.

        Args:
            email: User's email address

        Returns:
            (User, error_message) - User if successful, error message if not
        """
        # Validate email
        is_valid, error = validate_email(email, self.config)
        if not is_valid:
            return None, error

        email = email.strip().lower()

        # Create session
        session_id = create_session_token()
        user = User(email=email, session_id=session_id)

        # Store session in Redis
        if self._cache and self._cache.available:
            session_key = f"auth:session:{session_id}"
            await self._cache.set(
                session_key,
                user.to_dict(),
                ttl=self.config.session_ttl,
            )

            # Also store email->session mapping for lookup
            email_key = f"auth:email:{user.id}"
            await self._cache.set(
                email_key,
                {"session_id": session_id, "email": email},
                ttl=self.config.session_ttl,
            )

            logger.info(f"User logged in: {email[:3]}***@{user.domain}")

        return user, ""

    async def validate_session(self, session_id: str) -> Optional[User]:
        """
        Validate session token and return user.

        Args:
            session_id: Session token

        Returns:
            User if valid, None if invalid/expired
        """
        if not session_id or not self._cache or not self._cache.available:
            return None

        session_key = f"auth:session:{session_id}"
        data = await self._cache.get(session_key)

        if not data:
            return None

        try:
            user = User.from_dict(data)

            # Update last active
            user.last_active = datetime.now(timezone.utc)
            await self._cache.set(
                session_key,
                user.to_dict(),
                ttl=self.config.session_ttl,
            )

            return user
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return None

    async def logout(self, session_id: str) -> bool:
        """
        Logout user by invalidating session.

        Args:
            session_id: Session token to invalidate

        Returns:
            True if session was invalidated
        """
        if not session_id or not self._cache or not self._cache.available:
            return False

        session_key = f"auth:session:{session_id}"

        # Get user email to clean up email mapping
        data = await self._cache.get(session_key)
        if data:
            user = User.from_dict(data)
            email_key = f"auth:email:{user.id}"
            await self._cache.delete(email_key)

        # Delete session
        await self._cache.delete(session_key)
        return True

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get active user by email address.

        Args:
            email: User's email address

        Returns:
            User if found and session active, None otherwise
        """
        if not email or not self._cache or not self._cache.available:
            return None

        email = email.strip().lower()
        user_id = hashlib.sha256(email.encode()).hexdigest()[:16]
        email_key = f"auth:email:{user_id}"

        data = await self._cache.get(email_key)
        if not data:
            return None

        session_id = data.get("session_id")
        if session_id:
            return await self.validate_session(session_id)

        return None


# Global singleton
_email_auth: Optional[EmailAuth] = None


def get_email_auth() -> EmailAuth:
    """Get or create global EmailAuth instance."""
    global _email_auth
    if _email_auth is None:
        _email_auth = EmailAuth()
    return _email_auth


def init_email_auth(config: Optional[EmailAuthConfig] = None, cache=None) -> EmailAuth:
    """Initialize global EmailAuth with config and cache."""
    global _email_auth
    _email_auth = EmailAuth(config, cache)
    return _email_auth
