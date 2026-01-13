"""
FastAPI Dependencies for Email Authentication

Provides dependency injection for:
- Current user (required or optional)
- Session validation
"""

from typing import Optional
from fastapi import Depends, HTTPException, Request, status, Header

from .email_auth import (
    EmailAuth,
    User,
    get_email_auth,
)


async def get_current_user(
    request: Request,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
) -> User:
    """
    Get current authenticated user (REQUIRED).

    Session ID can be provided via:
    - X-Session-ID header
    - Authorization: Bearer <session_id> header
    - session_id query parameter

    Raises 401 if not authenticated.
    """
    auth = get_email_auth()

    if not auth.config.enabled:
        # Auth disabled - create anonymous user
        return User(
            email="anonymous@local",
            session_id="anonymous",
        )

    if not auth.available:
        if auth.config.allow_anonymous:
            return User(
                email="anonymous@local",
                session_id="anonymous",
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable",
        )

    # Try to get session ID from various sources
    session_id = x_session_id

    # Check Authorization header
    if not session_id:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            session_id = auth_header[7:]

    # Check query parameter
    if not session_id:
        session_id = request.query_params.get("session_id")

    if not session_id:
        if auth.config.allow_anonymous:
            return User(
                email="anonymous@local",
                session_id="anonymous",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please login with your email.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await auth.validate_session(session_id)
    if not user:
        if auth.config.allow_anonymous:
            return User(
                email="anonymous@local",
                session_id="anonymous",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Attach user to request state
    request.state.user = user
    return user


async def get_current_user_optional(
    request: Request,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
) -> Optional[User]:
    """
    Get current user if authenticated (OPTIONAL).

    Returns None if not authenticated.
    Does not raise 401.
    """
    auth = get_email_auth()

    if not auth.config.enabled or not auth.available:
        return None

    # Try to get session ID
    session_id = x_session_id

    if not session_id:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            session_id = auth_header[7:]

    if not session_id:
        session_id = request.query_params.get("session_id")

    if not session_id:
        return None

    user = await auth.validate_session(session_id)
    if user:
        request.state.user = user
    return user
