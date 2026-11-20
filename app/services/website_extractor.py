"""Safe website fetching and content extraction.

Fetching (SSRF-safe):
  * every hop (initial URL and each redirect) is validated and DNS-resolved;
    all resolved IPs must be public;
  * the request is sent to the *validated IP* with the original Host header and
    TLS SNI, so certificate checks still use the domain name but a second DNS
    lookup cannot redirect us to an internal address (DNS rebinding);
  * redirects are followed manually (max N), responses are size-capped and
    only HTML is accepted, with connect/read timeouts.

Extraction: title, meta tags, Open Graph, JSON-LD organisation data, headings,
navigation, main text, footer (location/contact hints) - cleaned, de-duplicated
and trimmed to a token budget. A few key internal pages (about / services /
products) are fetched too, because homepages often say little about services.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Tag

from app.config import Settings
from app.errors import (
    AppError,
    BlockedURLError,
    InsufficientContentError,
    InvalidURLError,
    UnsupportedContentError,
    WebsiteTimeoutError,
    WebsiteUnreachableError,
)
from app.utils.validators import IPAddress, normalize_url, resolve_public_ips

logger = logging.getLogger(__name__)

Resolver = Callable[[str, int], Awaitable[List[IPAddress]]]

REDIRECT_CODES = (301, 302, 303, 307, 308)
HTML_TYPES = ("text/html", "application/xhtml+xml")
REMOVE_TAGS = ("script", "style", "noscript", "svg", "iframe", "template", "canvas", "form")
NOISE_PATTERN = re.compile(r"cookie|consent|gdpr|modal|popup|newsletter", re.IGNORECASE)
SOCIAL_HOSTS = {
    "facebook.com": "Facebook",
    "instagram.com": "Instagram",
    "linkedin.com": "LinkedIn",
    "twitter.com": "X",
    "x.com": "X",
    "youtube.com": "YouTube",
}
# Internal pages that usually describe the business, in priority order.
PAGE_KEYWORDS = (
    "service",
    "about",
    "product",
    "solution",
    "what-we-do",
    "offering",
    "industr",
    "pricing",
    "company",
)
SKIP_PATH_PATTERN = re.compile(
    r"login|signin|sign-in|signup|register|cart|checkout|account|privacy|terms|cookie|"
    r"careers?|jobs|\.(pdf|jpe?g|png|gif|webp|svg|zip|mp4|mp3|docx?|xlsx?)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ParsedPage:
    url: str
    title: str = ""
    meta_description: str = ""
    meta_keywords: str = ""
    og: Dict[str, str] = field(default_factory=dict)
    lang: str = ""
    h1: List[str] = field(default_factory=list)
    h2: List[str] = field(default_factory=list)
    h3: List[str] = field(default_factory=list)
    nav_items: List[str] = field(default_factory=list)
    text_lines: List[str] = field(default_factory=list)
    footer_lines: List[str] = field(default_factory=list)
    organizations: List[Dict[str, Any]] = field(default_factory=list)
    internal_links: List[Tuple[str, str]] = field(default_factory=list)  # (text, url)
    social_links: Dict[str, str] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        words = " ".join([self.title, self.meta_description, *self.h1, *self.h2, *self.text_lines])
        return len(words.split())


@dataclass
class WebsiteContent:
    requested_url: str
    final_url: str
    homepage: ParsedPage
    extra_pages: List[ParsedPage] = field(default_factory=list)

    @property
    def pages_analyzed(self) -> List[str]:
        return [self.homepage.url] + [p.url for p in self.extra_pages]

    def to_prompt_text(self, max_chars: int) -> str:
        """Compact, labelled plain-text view of the website for the LLM."""
        home = self.homepage
        header: List[str] = [f"WEBSITE URL: {self.final_url}"]
        _add(header, "PAGE TITLE", home.title)
        _add(header, "META DESCRIPTION", home.meta_description)
        _add(header, "META KEYWORDS", home.meta_keywords)
        for key in ("site_name", "title", "description"):
            _add(header, f"OPEN GRAPH {key.upper()}", home.og.get(key, ""))
        _add(header, "LANGUAGE", home.lang)
        for org in home.organizations[:3]:
            _add(header, "STRUCTURED DATA (JSON-LD)", json.dumps(org, ensure_ascii=False))
        if home.social_links:
            _add(header, "SOCIAL PROFILES", ", ".join(home.social_links.values()))
        _add(header, "NAVIGATION MENU", " | ".join(home.nav_items[:40]))
        _add(header, "H1", " | ".join(home.h1[:10]))
        _add(header, "H2", " | ".join(home.h2[:25]))
        _add(header, "H3", " | ".join(home.h3[:25]))
        header_text = "\n".join(header)

        remaining = max(max_chars - len(header_text), 1000)
        extra_budget = remaining // 2 if self.extra_pages else 0
        sections = [
            header_text,
            "\nHOMEPAGE TEXT:\n" + _clip_lines(home.text_lines, remaining - extra_budget),
        ]

        if self.extra_pages:
            per_page = extra_budget // len(self.extra_pages)
            for page in self.extra_pages:
                head = [f"\nPAGE: {page.url}"]
                _add(head, "TITLE", page.title)
                _add(head, "HEADINGS", " | ".join((page.h1 + page.h2)[:15]))
                body = _clip_lines(page.text_lines, max(per_page - 200, 300))
                sections.append("\n".join(head) + "\n" + body)

        if home.footer_lines:
            sections.append("\nFOOTER:\n" + _clip_lines(home.footer_lines, 800))
        return "\n".join(sections)[:max_chars]


def _add(lines: List[str], label: str, value: str) -> None:
    if value:
        lines.append(f"{label}: {value}")


def _clip_lines(lines: Iterable[str], max_chars: int) -> str:
    out: List[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > max_chars:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _unique(items: Iterable[str], min_len: int = 2) -> List[str]:
    seen, out = set(), []
    for item in items:
        item = _clean(item)
        key = item.lower()
        if len(item) >= min_len and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _bare_host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _meta(soup: BeautifulSoup, *, name: str = "", prop: str = "") -> str:
    tag = soup.find("meta", attrs={"name": name}) if name else None
    if tag is None and prop:
        tag = soup.find("meta", attrs={"property": prop})
    if tag is None and name:  # case-insensitive fallback
        tag = soup.find("meta", attrs={"name": re.compile(f"^{name}$", re.IGNORECASE)})
    return _clean(tag.get("content", "")) if isinstance(tag, Tag) else ""


def _walk_json_ld(node: Any, found: List[Dict[str, Any]]) -> None:
    """Collect organisation-like objects from (possibly nested) JSON-LD."""
    if isinstance(node, list):
        for item in node:
            _walk_json_ld(item, found)
        return
    if not isinstance(node, dict):
        return
    types = node.get("@type", [])
    types = types if isinstance(types, list) else [types]
    wanted = {"Organization", "Corporation", "LocalBusiness", "Store", "ProfessionalService"}
    if any(isinstance(t, str) and (t in wanted or t.endswith("Business")) for t in types):
        org: Dict[str, Any] = {}
        for key in ("name", "legalName", "description", "slogan", "url", "areaServed"):
            if isinstance(node.get(key), (str, list)):
                org[key] = node[key]
        address = node.get("address")
        if isinstance(address, dict):
            org["address"] = {
                k: address[k]
                for k in ("addressLocality", "addressRegion", "addressCountry")
                if isinstance(address.get(k), str)
            }
        if isinstance(node.get("sameAs"), list):
            org["sameAs"] = node["sameAs"][:8]
        if org:
            found.append(org)
    for key in ("@graph", "mainEntity", "publisher", "provider"):
        if key in node:
            _walk_json_ld(node[key], found)


def parse_html(html: bytes, url: str) -> ParsedPage:
    """Extract structured, cleaned content from an HTML document."""
    soup = BeautifulSoup(html, "html.parser")
    page = ParsedPage(url=url)

    page.title = _clean(soup.title.get_text()) if soup.title else ""
    page.meta_description = _meta(soup, name="description", prop="og:description")
    page.meta_keywords = _meta(soup, name="keywords")
    for key in ("site_name", "title", "description"):
        value = _meta(soup, prop=f"og:{key}")
        if value:
            page.og[key] = value
    html_tag = soup.find("html")
    if isinstance(html_tag, Tag):
        page.lang = _clean(html_tag.get("lang", ""))

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            _walk_json_ld(json.loads(script.string or ""), page.organizations)
        except (ValueError, TypeError):
            continue

    page_host = _bare_host(url)
    internal: Dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(url, href).split("#", 1)[0]
        host = _bare_host(absolute)
        for social_host, label in SOCIAL_HOSTS.items():
            if host == social_host or host.endswith("." + social_host):
                page.social_links.setdefault(label, absolute)
        if host == page_host and absolute.startswith(("http://", "https://")):
            internal.setdefault(absolute, _clean(anchor.get_text()))
    page.internal_links = [(text, link) for link, text in internal.items()]

    page.h1 = _unique(h.get_text(" ") for h in soup.find_all("h1"))
    page.h2 = _unique(h.get_text(" ") for h in soup.find_all("h2"))
    page.h3 = _unique(h.get_text(" ") for h in soup.find_all("h3"))

    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()
    noisy = [
        tag
        for tag in soup.find_all(True)
        if isinstance(tag, Tag)
        and tag.attrs is not None
        and NOISE_PATTERN.search(" ".join([tag.get("id", "") or "", *(tag.get("class") or [])]))
    ]
    for tag in noisy:
        if not tag.decomposed:
            tag.decompose()

    page.nav_items = _unique(
        a.get_text(" ") for nav in soup.find_all("nav") for a in nav.find_all("a")
    )
    footer = soup.find("footer")
    if isinstance(footer, Tag):
        page.footer_lines = _unique(footer.get_text("\n").split("\n"), min_len=3)[:25]
        footer.decompose()
    for nav in soup.find_all("nav"):
        nav.decompose()

    region = soup.find("main") or soup.find("article") or soup.body or soup
    page.text_lines = [
        line for line in _unique(region.get_text("\n").split("\n"), min_len=3) if not line.isdigit()
    ]
    return page


def pick_extra_pages(page: ParsedPage, limit: int) -> List[str]:
    """Choose internal pages most likely to describe the business."""
    if limit <= 0:
        return []
    home_path = urlsplit(page.url).path.rstrip("/")
    chosen: List[str] = []
    for keyword in PAGE_KEYWORDS:
        for text, link in page.internal_links:
            path = urlsplit(link).path
            if path.rstrip("/") == home_path or SKIP_PATH_PATTERN.search(path):
                continue
            if path.count("/") > 3:  # deep pages are usually individual posts
                continue
            if keyword in path.lower() or keyword in text.lower():
                if link not in chosen:
                    chosen.append(link)
                break
        if len(chosen) >= limit:
            break
    return chosen


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
class WebsiteExtractor:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        resolver: Resolver = resolve_public_ips,
    ) -> None:
        self.settings = settings
        self.client = client
        self.resolver = resolver

    async def extract(self, url: str) -> WebsiteContent:
        final_url, html = await self.fetch_html(url)
        homepage = parse_html(html, final_url)

        extra_urls = pick_extra_pages(homepage, self.settings.max_extra_pages)
        results = await asyncio.gather(
            *(self._fetch_extra(link) for link in extra_urls), return_exceptions=False
        )
        extra_pages = [page for page in results if page is not None]

        content = WebsiteContent(url, final_url, homepage, extra_pages)
        total_words = homepage.word_count + sum(p.word_count for p in extra_pages)
        if total_words < 20:
            raise InsufficientContentError(
                "The website returned too little readable text to analyse. It may require "
                "JavaScript to render its content or block automated access."
            )
        logger.info(
            "Extracted %s: %d page(s), %d words",
            final_url,
            len(content.pages_analyzed),
            total_words,
        )
        return content

    async def _fetch_extra(self, url: str) -> Optional[ParsedPage]:
        try:
            final_url, html = await self.fetch_html(url)
            return parse_html(html, final_url)
        except AppError as exc:
            logger.info("Skipping extra page %s: %s", url, exc.message)
            return None

    async def fetch_html(self, url: str) -> Tuple[str, bytes]:
        """Fetch ``url`` safely, following redirects manually. Returns (final_url, html)."""
        current = url
        for _ in range(self.settings.max_redirects + 1):
            parts = urlsplit(current)
            host = parts.hostname or ""
            port = parts.port or (443 if parts.scheme == "https" else 80)
            ip = (await self.resolver(host, port))[0]

            ip_host = f"[{ip}]" if ip.version == 6 else str(ip)
            netloc = ip_host if parts.port is None else f"{ip_host}:{parts.port}"
            target = urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))
            headers = {
                "Host": parts.netloc,
                "User-Agent": self.settings.user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                "Accept-Language": "en-US,en;q=0.8",
            }
            extensions = {"sni_hostname": host} if parts.scheme == "https" else {}

            try:
                async with self.client.stream(
                    "GET", target, headers=headers, extensions=extensions
                ) as response:
                    if response.status_code in REDIRECT_CODES:
                        location = response.headers.get("location")
                        if not location:
                            raise WebsiteUnreachableError(
                                "Website sent a redirect without a target."
                            )
                        try:
                            current = normalize_url(urljoin(current, location))
                        except InvalidURLError as exc:
                            raise BlockedURLError(
                                f"Redirect target rejected: {exc.message}"
                            ) from exc
                        continue
                    self._check_response(response)
                    return current, await self._read_limited(response)
            except httpx.TimeoutException as exc:
                raise WebsiteTimeoutError(
                    "The website did not respond within "
                    f"{self.settings.fetch_timeout_seconds:g} seconds."
                ) from exc
            except httpx.ConnectError as exc:
                if "CERTIFICATE" in str(exc).upper() or "SSL" in str(exc).upper():
                    raise WebsiteUnreachableError(
                        "The website's SSL/TLS certificate is invalid or could not be verified."
                    ) from exc
                raise WebsiteUnreachableError(
                    "Could not connect to the website (connection refused or network error)."
                ) from exc
            except httpx.HTTPError as exc:
                raise WebsiteUnreachableError(
                    f"Could not connect to the website ({type(exc).__name__})."
                ) from exc
        raise WebsiteUnreachableError("The website redirected too many times.")

    @staticmethod
    def _check_response(response: httpx.Response) -> None:
        status = response.status_code
        if status >= 400:
            hints = {
                401: "the page requires a login",
                403: "the website blocks automated access",
                404: "the page was not found",
                429: "the website is rate limiting requests",
            }
            hint = hints.get(status, "server error" if status >= 500 else "request rejected")
            raise WebsiteUnreachableError(f"Website returned HTTP {status} ({hint}).")
        if status != 200:
            raise WebsiteUnreachableError(f"Unexpected HTTP status {status} from website.")
        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(t in content_type for t in HTML_TYPES):
            raise UnsupportedContentError(
                "URL does not point to an HTML web page "
                f"(content type: {content_type.split(';')[0]})."
            )

    async def _read_limited(self, response: httpx.Response) -> bytes:
        """Read at most ``max_page_bytes``; larger pages are truncated, not loaded fully."""
        limit = self.settings.max_page_bytes
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) >= limit:
                logger.info("Page larger than %d bytes, truncated", limit)
                break
        return bytes(body[:limit])
