"""URL validation and SSRF protection.

Two layers:
1. ``normalize_url`` - syntax checks (scheme, host, port, credentials, IP literals).
2. ``resolve_public_ips`` - DNS resolution; every resolved address must be public.
   The fetcher then connects to the *checked* IP (DNS pinning), so a second DNS
   lookup cannot swap in an internal address (DNS rebinding).
"""

import asyncio
import ipaddress
import re
import socket
from typing import List, Union
from urllib.parse import urlsplit, urlunsplit

from app.errors import BlockedURLError, InvalidURLError, WebsiteUnreachableError

IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]

MAX_URL_LENGTH = 2048
ALLOWED_SCHEMES = ("http", "https")
ALLOWED_PORTS = (None, 80, 443)
_LABEL = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")
_TLD = re.compile(r"^(?:[a-z]{2,63}|xn--[a-z0-9-]{1,59})$")
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain", ".home.arpa", ".lan")


def is_public_ip(ip: IPAddress) -> bool:
    """True only for globally routable unicast addresses."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def normalize_url(raw: str) -> str:
    """Validate a user-supplied website URL and return a normalized form.

    Adds ``https://`` when no scheme is given, lowercases the host, removes the
    fragment. Raises InvalidURLError / BlockedURLError.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidURLError("URL must not be empty.")
    url = raw.strip()
    if len(url) > MAX_URL_LENGTH:
        raise InvalidURLError("URL is too long.")
    if any(ch.isspace() for ch in url):
        raise InvalidURLError("URL must not contain spaces.")
    if "://" not in url:
        url = f"https://{url}"

    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise InvalidURLError("URL is not valid.") from exc

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise InvalidURLError("Only http and https URLs are supported.")
    if parts.username or parts.password:
        raise InvalidURLError("URL must not contain a username or password.")
    if port not in ALLOWED_PORTS:
        raise BlockedURLError("Only the standard web ports 80 and 443 are allowed.")

    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise InvalidURLError("URL must contain a domain name.")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is not None:
        if not is_public_ip(ip):
            raise BlockedURLError("URL points to a private or internal network address.")
        netloc_host = f"[{host}]" if ip.version == 6 else host
    else:
        if host == "localhost" or host.endswith(_BLOCKED_SUFFIXES):
            raise BlockedURLError("URL points to a local or internal host name.")
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise InvalidURLError("URL has an invalid domain name.") from exc
        labels = ascii_host.split(".")
        if len(labels) < 2 or not _TLD.match(labels[-1]):
            raise InvalidURLError("URL must contain a valid domain, e.g. https://example.com.")
        if not all(_LABEL.match(label) for label in labels):
            raise InvalidURLError("URL has an invalid domain name.")
        netloc_host = ascii_host

    netloc = netloc_host if port is None else f"{netloc_host}:{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


async def resolve_public_ips(host: str, port: int) -> List[IPAddress]:
    """Resolve ``host`` and ensure that *every* address is public."""
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None:
        if not is_public_ip(literal):
            raise BlockedURLError("URL points to a private or internal network address.")
        return [literal]

    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise WebsiteUnreachableError(f"Domain name could not be resolved: {host}") from exc

    addresses: List[IPAddress] = []
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not is_public_ip(ip):
            raise BlockedURLError("URL resolves to a private or internal network address.")
        if ip not in addresses:
            addresses.append(ip)
    if not addresses:
        raise WebsiteUnreachableError(f"Domain name could not be resolved: {host}")
    return addresses
