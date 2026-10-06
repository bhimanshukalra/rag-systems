import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from agentic_rag.api.auth import require_auth
from agentic_rag.config import Settings


def _settings() -> Settings:
    return Settings(
        groq_api_key="g", pinecone_api_key="p", api_auth_token="secret-token"
    )


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_valid_token_passes():
    assert require_auth(_creds("secret-token"), _settings()) is None


@pytest.mark.parametrize("credentials", [_creds("wrong"), _creds(""), _creds("sécret"), None])
def test_invalid_or_missing_token_rejected(credentials):
    with pytest.raises(HTTPException) as exc:
        require_auth(credentials, _settings())

    assert exc.value.status_code == 401
    assert exc.value.headers == {"WWW-Authenticate": "Bearer"}
