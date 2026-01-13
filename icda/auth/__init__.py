"""
ICDA Simple Email Authentication

Lightweight email-based authentication:
- Users identified by email address
- Session tokens stored in Redis
- No external auth provider required
"""

from .email_auth import (
    EmailAuth,
    EmailAuthConfig,
    User,
    get_email_auth,
    init_email_auth,
    create_session_token,
    validate_email,
)

from .dependencies import (
    get_current_user,
    get_current_user_optional,
)

__all__ = [
    "EmailAuth",
    "EmailAuthConfig",
    "User",
    "get_email_auth",
    "init_email_auth",
    "create_session_token",
    "validate_email",
    "get_current_user",
    "get_current_user_optional",
]
