import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agentic_rag.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    supplied = credentials.credentials if credentials else ""
    # compare_digest on bytes: constant-time, and safe for non-ASCII input.
    if not secrets.compare_digest(
        supplied.encode(), settings.api_auth_token.encode()
    ):
        raise HTTPException(
            status_code=401,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
