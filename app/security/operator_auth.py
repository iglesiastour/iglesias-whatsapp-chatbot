"""Operator authentication boundary.

Deterministic bearer-token authentication for operator-facing API routes.
Independent of AI, conversation pipeline, WhatsApp, handoff lifecycle, and
PostgreSQL implementation details. Fails closed when operator authentication
is not configured on the server.
"""

import secrets

from fastapi import Header, HTTPException, status

from app.config import settings

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Operator authentication required.",
)

_AUTH_UNAVAILABLE = HTTPException(
    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    detail="Operator authentication is unavailable.",
)


def _configured_operator_token() -> str:
    """Return the configured token, or raise fail-closed if unconfigured."""
    configured = settings.operator_api_token
    if not isinstance(configured, str) or not configured.strip():
        # Fail closed: never reveal whether the value was empty/missing.
        raise _AUTH_UNAVAILABLE
    return configured


def require_operator_credentials(
    authorization: str | None = Header(default=None),
) -> None:
    """Validate an `Authorization: Bearer <token>` header (timing-safe).

    Raises 401 for missing/malformed/incorrect credentials and 503 when the
    server has no operator token configured. Never logs or echoes secrets.
    """
    expected = _configured_operator_token()

    if not authorization:
        raise _UNAUTHORIZED

    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise _UNAUTHORIZED

    credentials = credentials.strip()
    if not credentials or " " in credentials:
        # Empty token or extra-token malformed header.
        raise _UNAUTHORIZED

    if not secrets.compare_digest(credentials.encode("utf-8"), expected.encode("utf-8")):
        raise _UNAUTHORIZED
