import socket

import pytest
import requests

from agentic_rag.ingestion.url_safety import (
    UnsafeURLError,
    safe_get,
    validate_source_url,
)

PUBLIC_IP = "93.184.216.34"


def _getaddrinfo_returning(*ips):
    def fake(host, port):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]

    return fake


class _FakeResponse:
    def __init__(self, status_code, location=None):
        self.status_code = status_code
        self.headers = {"Location": location} if location else {}

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307) and "Location" in self.headers

    @property
    def is_permanent_redirect(self):
        return self.status_code in (301, 308) and "Location" in self.headers


# -- validate_source_url: negative fixtures ---------------------------------


def test_rejects_non_http_scheme():
    with pytest.raises(UnsafeURLError):
        validate_source_url("file:///etc/passwd")


def test_rejects_cloud_metadata_ip():
    with pytest.raises(UnsafeURLError):
        validate_source_url("http://169.254.169.254/")


def test_rejects_localhost_hostname():
    with pytest.raises(UnsafeURLError):
        validate_source_url("http://localhost:8000/")


def test_rejects_loopback_ip():
    with pytest.raises(UnsafeURLError):
        validate_source_url("http://127.0.0.1/")


def test_rejects_disallowed_port(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo_returning(PUBLIC_IP))
    with pytest.raises(UnsafeURLError):
        validate_source_url("https://example.test:8443/")


def test_rejects_if_any_resolved_address_is_private(monkeypatch):
    # A host round-robining between a public and a private IP must be
    # rejected on the private address even when it isn't first in the list.
    monkeypatch.setattr(
        socket, "getaddrinfo", _getaddrinfo_returning(PUBLIC_IP, "10.0.0.5")
    )
    with pytest.raises(UnsafeURLError):
        validate_source_url("http://example-internal.test/")


def test_unresolvable_host_is_rejected(monkeypatch):
    def fake(host, port):
        raise socket.gaierror("name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    with pytest.raises(UnsafeURLError):
        validate_source_url("https://this-does-not-resolve.test/")


# -- validate_source_url: positive cases -------------------------------------


def test_allows_public_ip_literal():
    validate_source_url("http://93.184.216.34/")


def test_allows_hostname_resolving_only_to_public_ips(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo_returning(PUBLIC_IP))
    validate_source_url("https://example.test/")


# -- safe_get -----------------------------------------------------------------


def test_safe_get_returns_response_with_no_redirect(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo_returning(PUBLIC_IP))
    calls = []

    def fake_get(url, *, timeout, headers=None, allow_redirects):
        calls.append(url)
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", fake_get)

    response = safe_get("https://example.test/", timeout=5)

    assert response.status_code == 200
    assert calls == ["https://example.test/"]


def test_safe_get_follows_and_revalidates_redirects(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo_returning(PUBLIC_IP))
    responses = iter(
        [
            _FakeResponse(302, location="https://example.test/next"),
            _FakeResponse(200),
        ]
    )
    calls = []

    def fake_get(url, *, timeout, headers=None, allow_redirects):
        calls.append(url)
        return next(responses)

    monkeypatch.setattr(requests, "get", fake_get)

    response = safe_get("https://example.test/", timeout=5)

    assert response.status_code == 200
    assert calls == ["https://example.test/", "https://example.test/next"]


def test_safe_get_rejects_redirect_to_unsafe_target(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo_returning(PUBLIC_IP))

    def fake_get(url, *, timeout, headers=None, allow_redirects):
        return _FakeResponse(302, location="http://169.254.169.254/secret")

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(UnsafeURLError):
        safe_get("https://example.test/", timeout=5)


def test_safe_get_gives_up_after_max_redirects(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo_returning(PUBLIC_IP))

    def fake_get(url, *, timeout, headers=None, allow_redirects):
        return _FakeResponse(302, location=url)  # redirects to itself forever

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(UnsafeURLError):
        safe_get("https://example.test/", timeout=5, max_redirects=2)
