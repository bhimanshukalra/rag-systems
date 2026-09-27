import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests

ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_ALLOWED_PORTS = {80, 443}
DEFAULT_MAX_REDIRECTS = 3


class UnsafeURLError(ValueError):
    """Raised when a source URL fails safety validation."""


def _resolved_ips(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"could not resolve host {hostname!r}: {exc}") from exc

    ips = {ipaddress.ip_address(sockaddr[0]) for *_, sockaddr in addr_infos}
    if not ips:
        raise UnsafeURLError(f"host {hostname!r} did not resolve to any address")
    return list(ips)


def _is_disallowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_source_url(
    url: str, *, allowed_ports: set[int] = DEFAULT_ALLOWED_PORTS
) -> None:
    """Raise UnsafeURLError if `url` is not safe to fetch.

    Checks every DNS-resolved address for the host, not just the first --
    a host can round-robin between a public and a private IP.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"scheme {parsed.scheme!r} is not allowed")

    if not parsed.hostname:
        raise UnsafeURLError("URL has no hostname")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in allowed_ports:
        raise UnsafeURLError(f"port {port} is not allowed")

    for ip in _resolved_ips(parsed.hostname):
        if _is_disallowed(ip):
            raise UnsafeURLError(
                f"host {parsed.hostname!r} resolves to disallowed address {ip}"
            )


def safe_get(
    url: str,
    *,
    timeout: float,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """GET `url`, re-validating the target of every redirect hop before following it.

    Trade-off, not an oversight: this stops the realistic threat here --
    a pasted metadata/localhost URL, including one reached via a redirect
    -- but does not resolve-and-pin the IP for the actual connection, so
    it is not full protection against DNS rebinding between validation
    and connection. That's judged disproportionate for this project's
    threat model (see docs/agentic_rag/planning).
    """
    next_url = url
    for _ in range(max_redirects + 1):
        validate_source_url(next_url)
        response = requests.get(
            next_url,
            timeout=timeout,
            headers=headers,
            allow_redirects=False,
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            if not location:
                raise UnsafeURLError("redirect response missing Location header")
            next_url = urljoin(next_url, location)
            continue
        return response

    raise UnsafeURLError(f"too many redirects (> {max_redirects}) for {url!r}")
