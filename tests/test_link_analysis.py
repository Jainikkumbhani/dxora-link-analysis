"""Tests for the /link-analysis API.

Websites, DNS and the Gemini API are all simulated (httpx.MockTransport and a
fake resolver): no network access and no API key are needed.
"""

import copy
import ipaddress
import json
import socket
from typing import Dict, List, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import BlockedURLError, InvalidURLError
from app.main import app, get_analysis_service
from app.schemas import SearchQuery
from app.services.analysis_service import LLM_SCHEMA, AnalysisService
from app.services.llm_service import GeminiService
from app.services.website_extractor import (
    WebsiteExtractor,
    parse_html,
    pick_extra_pages,
)
from app.utils.cache import RateLimiter, TTLCache
from app.utils.validators import normalize_url, resolve_public_ips

PUBLIC_IP = "93.184.215.14"
API_KEY = "test-gemini-key-123"
GEMINI_HOST = "generativelanguage.googleapis.com"

HOME_HTML = f"""<!doctype html>
<html lang="en"><head>
<title>Acme Web Studio | Web Development Agency in Ahmedabad</title>
<meta name="description"
  content="Acme Web Studio builds fast websites and e-commerce stores for small businesses.">
<meta name="keywords" content="web development, ecommerce">
<meta property="og:site_name" content="Acme Web Studio">
<script type="application/ld+json">{
    json.dumps(
        {
            "@context": "https://schema.org",
            "@graph": [
                {
                    "@type": "LocalBusiness",
                    "name": "Acme Web Studio",
                    "address": {"addressLocality": "Ahmedabad", "addressCountry": "IN"},
                    "sameAs": ["https://www.linkedin.com/company/acme"],
                }
            ],
        }
    )
}</script>
<script>var tracking = "SHOULD_NOT_APPEAR";</script>
<style>.x {{ color: red }}</style>
</head><body>
<nav><a href="/">Home</a><a href="/services">Services</a><a href="/about-us">About Us</a>
<a href="/login">Login</a><a href="/brochure.pdf">Brochure</a></nav>
<div id="cookie-banner">We use cookies to improve your experience</div>
<main>
  <h1>Websites that grow small businesses</h1>
  <h2>Web Development</h2><p>Custom websites built with modern frameworks.</p>
  <h2>E-commerce Stores</h2><p>Online stores with secure payments and inventory.</p>
  <h2>SEO Services</h2><p>Rank higher on Google with technical SEO.</p>
  <p>Custom websites built with modern frameworks.</p>
  <p>Trusted by 120 local businesses across Gujarat since 2015.</p>
</main>
<footer>Acme Web Studio, SG Highway, Ahmedabad, Gujarat
<a href="https://www.instagram.com/acmewebstudio">Instagram</a></footer>
</body></html>"""

SERVICES_HTML = """<html><head><title>Services - Acme</title></head><body><main>
<h1>Our Services</h1><p>Website maintenance and hosting plans for growing teams.</p>
<p>Mobile-friendly redesigns of existing websites.</p></main></body></html>"""

ABOUT_HTML = """<html><head><title>About - Acme</title></head><body><main>
<h1>About Acme</h1><p>We are a team of twelve developers and designers.</p></main></body></html>"""


def llm_payload(**overrides) -> Dict:
    """A valid analysis exactly as the LLM would return it."""
    data = {
        "brand": {
            "name": "Acme Web Studio",
            "category": "Web Development Agency",
            "summary": "Ahmedabad agency building websites and online stores for small businesses.",
        },
        "competitor_analysis": {
            "brand_name": "Acme Web Studio",
            "brand_overview": "Small-business web agency in Ahmedabad.",
            "business_category": "Web Development Agency",
            "main_products_services": ["Web Development", "E-commerce Stores", "SEO Services"],
            "brand_target_market": "Small businesses in Gujarat",
            "brand_key_strength": "Local, small-business focus",
            "competitors": [
                {
                    "name": "WebPixel Solutions",
                    "website": "https://webpixel.example.com",
                    "main_service": "Web Development",
                    "target_market": "SMEs in India",
                    "strengths": ["Large portfolio"],
                    "weaknesses_or_gaps": ["Generic templates"],
                    "confidence": "Medium",
                },
                {
                    "name": "Acme Web Studio",  # the brand itself -> must be removed
                    "website": None,
                    "main_service": "Web Development",
                    "target_market": "-",
                    "strengths": [],
                    "weaknesses_or_gaps": [],
                    "confidence": "high",
                },
                {
                    "name": "webpixel solutions",  # duplicate -> must be removed
                    "website": None,
                    "main_service": "Web Development",
                    "target_market": "-",
                    "strengths": [],
                    "weaknesses_or_gaps": [],
                    "confidence": "low",
                },
                {
                    "name": "Shopify",
                    "website": "not a url at all",  # malformed -> set to null
                    "main_service": "E-commerce platform",
                    "target_market": "Global merchants",
                    "strengths": ["Brand recognition"],
                    "weaknesses_or_gaps": ["Less personal service"],
                    "confidence": "high",
                },
                {
                    "name": "Digital Gujarat Agency",
                    "website": None,
                    "main_service": "Digital marketing",
                    "target_market": "Gujarat SMEs",
                    "strengths": ["Marketing bundles"],
                    "weaknesses_or_gaps": ["Limited custom development"],
                    "confidence": "low",
                },
            ],
            "strengths": ["Local presence", "Small-business pricing"],
            "opportunities": ["Publish case studies", "Target e-commerce SEO"],
        },
        "target_search_queries": [
            {
                "query": f"web development company {i}",
                "platforms": platforms,
                "intent": "Service Search",
                "business_relevance": "Core service",
                "target_audience": "Small business owners",
            }
            for i, platforms in enumerate(
                [["Google"], ["Facebook"], ["Instagram"], ["LinkedIn"], ["Twitter", "google"]]
            )
        ],
        "seo_improvements": {
            "primary_keyword": "web development company Ahmedabad",
            "secondary_keywords": ["ecommerce website development"],
            "long_tail_keywords": ["affordable website for small business Ahmedabad"],
            "business_keywords": ["web agency Gujarat"],
            "recommended_title": "Web Development Company in Ahmedabad | Acme Web Studio",
            "title_primary_keyword": "web development company Ahmedabad",
            "title_structure": "Primary Keyword | Brand",
            "meta_description": "Acme Web Studio builds fast websites and online stores for "
            "small businesses in Ahmedabad. Get a free quote today.",
            "meta_summary": "Websites and online stores for small businesses.",
            "meta_keywords": ["web development", "ecommerce"],
            "supporting_content": [
                {
                    "content_type": "FAQ",
                    "title": "Website cost FAQ",
                    "description": "Answer pricing questions",
                    "target_keyword": "website cost Ahmedabad",
                }
            ],
            "issues_found": ["Title is longer than 60 characters"],
        },
        "top_services": [
            {
                "service_name": name,
                "description": "desc",
                "target_industry": "Retail",
                "target_customer": "Small businesses",
                "main_benefit": "More customers",
                "source": source,
            }
            for name, source in [
                ("Web Development", "website"),
                ("E-commerce Stores", "website"),
                ("SEO Services", "website"),
                ("Website Maintenance", "WEBSITE"),
                ("Google Ads Management", "inferred"),
            ]
        ],
        "target_audience": {
            "primary_audience": ["Small business owners"],
            "secondary_audience": ["Startups"],
            "industries": ["Retail"],
            "business_types": ["Small Businesses"],
            "customer_profiles": [{"profile": "Shop owner", "description": "Wants online sales"}],
            "geographic_focus": ["Ahmedabad", "Gujarat"],
            "geographic_focus_source": "website",
            "customer_needs": ["Online presence"],
            "customer_challenges": ["Limited budget"],
            "customer_interests": ["Online selling"],
            "recommended_segments": [
                {
                    "audience": "Audience 1",
                    "business_type": "Retail Businesses",
                    "requirement": "E-commerce Solutions",
                    "recommended_channels": ["Instagram"],
                }
            ],
        },
    }
    data.update(overrides)
    return data


def gemini_ok(data: Dict, model: str = "gemini-3.8-flash") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "thinking...", "thought": True},
                            {"text": json.dumps(data)},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 4000,
                "candidatesTokenCount": 2500,
                "thoughtsTokenCount": 500,
            },
            "modelVersion": model,
        },
    )


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
class FakeWorld:
    """Simulated websites + Gemini API behind one MockTransport."""

    def __init__(self) -> None:
        self.pages: Dict[str, httpx.Response] = {
            "/": httpx.Response(200, html=HOME_HTML),
            "/services": httpx.Response(200, html=SERVICES_HTML),
            "/about-us": httpx.Response(200, html=ABOUT_HTML),
        }
        self.gemini: List[httpx.Response] = []
        self.requests: List[httpx.Request] = []
        self.dns: Dict[str, List[str]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == GEMINI_HOST:
            assert request.headers["x-goog-api-key"] == API_KEY
            if not self.gemini:
                return gemini_ok(llm_payload())
            return self.gemini.pop(0)
        # Website requests must go to the validated IP with the real Host header.
        assert request.url.host == PUBLIC_IP, "request was not pinned to the resolved IP"
        assert request.headers["host"].endswith("acme.example.com")
        response = self.pages.get(request.url.path)
        return response if response is not None else httpx.Response(404, html="missing")

    async def resolver(self, host: str, port: int):
        return [ipaddress.ip_address(ip) for ip in self.dns.get(host, [PUBLIC_IP])]

    def gemini_requests(self) -> List[httpx.Request]:
        return [r for r in self.requests if r.url.host == GEMINI_HOST]

    def site_requests(self) -> List[httpx.Request]:
        return [r for r in self.requests if r.url.host != GEMINI_HOST]


def make_settings(**overrides) -> Settings:
    values = dict(
        gemini_api_key=API_KEY,
        gemini_model="gemini-3.8-flash",
        gemini_fallback_model="gemini-3.5-flash-lite",
        max_extra_pages=2,
        rate_limit_per_minute=0,
        llm_retry_delay_seconds=0,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def world() -> FakeWorld:
    return FakeWorld()


@pytest.fixture
def make_client(world: FakeWorld):
    clients = []

    def _make(settings: Optional[Settings] = None, rate_limit: int = 0) -> TestClient:
        settings = settings or make_settings()
        http = httpx.AsyncClient(transport=httpx.MockTransport(world.handler))
        service = AnalysisService(
            settings,
            WebsiteExtractor(settings, http, resolver=world.resolver),
            GeminiService(settings, http),
            TTLCache(settings.cache_ttl_seconds, settings.cache_max_entries),
        )
        app.dependency_overrides[get_analysis_service] = lambda: service
        client = TestClient(app)
        client.__enter__()
        app.state.rate_limiter = RateLimiter(rate_limit)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)
    app.dependency_overrides.clear()


def analyse(client: TestClient, url: str = "https://acme.example.com") -> httpx.Response:
    return client.post("/link-analysis", json={"url": url})


# ---------------------------------------------------------------------------
# URL validation / SSRF
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("example.com", "https://example.com/"),
        ("HTTPS://Example.COM/Path?q=1#frag", "https://example.com/Path?q=1"),
        ("http://example.com:80/", "http://example.com:80/"),
        ("https://münchen.de", "https://xn--mnchen-3ya.de/"),
        ("https://8.8.8.8", "https://8.8.8.8/"),
    ],
)
def test_normalize_url_valid(raw, expected):
    assert normalize_url(raw) == expected


@pytest.mark.parametrize(
    "raw, error",
    [
        ("", InvalidURLError),
        ("not a url", InvalidURLError),
        ("ftp://example.com", InvalidURLError),
        ("javascript:alert(1)", InvalidURLError),
        ("https://user:pass@example.com", InvalidURLError),
        ("https://example", InvalidURLError),
        ("https://exa_mple.com", InvalidURLError),
        ("http://localhost", BlockedURLError),
        ("http://printer.local", BlockedURLError),
        ("http://127.0.0.1", BlockedURLError),
        ("http://10.0.0.5", BlockedURLError),
        ("http://192.168.1.1", BlockedURLError),
        ("http://169.254.169.254/latest/meta-data", BlockedURLError),
        ("http://[::1]/", BlockedURLError),
        ("http://[::ffff:127.0.0.1]/", BlockedURLError),
        ("http://0.0.0.0", BlockedURLError),
        ("https://example.com:8080", BlockedURLError),
        ("http://example.com:22", BlockedURLError),
    ],
)
def test_normalize_url_rejects(raw, error):
    with pytest.raises(error):
        normalize_url(raw)


def test_dns_resolving_to_private_ip_is_blocked(monkeypatch):
    import asyncio

    async def fake_getaddrinfo(host, port, type=0):
        ips = ["93.184.215.14", "10.0.0.7"]  # one private address is enough to block
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]

    async def run():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
        await resolve_public_ips("evil.example.com", 443)

    with pytest.raises(BlockedURLError):
        asyncio.run(run())


def test_api_rejects_invalid_and_blocked_urls(make_client, world):
    client = make_client()
    bad = analyse(client, "not a url")
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_url"
    blocked = analyse(client, "http://169.254.169.254/")
    assert blocked.status_code == 400 and blocked.json()["error"]["code"] == "blocked_url"
    missing = client.post("/link-analysis", json={})
    assert missing.status_code == 422 and missing.json()["error"]["code"] == "validation_error"
    assert world.requests == []  # nothing was fetched


def test_redirect_to_private_address_is_blocked(make_client, world):
    world.pages["/"] = httpx.Response(302, headers={"location": "http://192.168.0.1/admin"})
    response = analyse(make_client())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "blocked_url"


def test_redirect_to_domain_resolving_privately_is_blocked(make_client, world):
    world.pages["/"] = httpx.Response(301, headers={"location": "https://internal.example.net/"})
    world.dns["internal.example.net"] = ["10.1.2.3"]

    async def resolver(host, port):
        ips = world.dns.get(host, [PUBLIC_IP])
        for ip in ips:
            if not ipaddress.ip_address(ip).is_global:
                raise BlockedURLError("URL resolves to a private or internal network address.")
        return [ipaddress.ip_address(ip) for ip in ips]

    world.resolver = resolver  # type: ignore[assignment]
    response = analyse(make_client())
    assert response.status_code == 400


def test_too_many_redirects(make_client, world):
    world.pages["/"] = httpx.Response(302, headers={"location": "/"})
    response = analyse(make_client())
    assert response.status_code == 502
    assert "redirected too many times" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Website fetching errors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "page, status, code, text",
    [
        (httpx.Response(403, html="no"), 502, "website_unreachable", "blocks automated access"),
        (httpx.Response(500, html="err"), 502, "website_unreachable", "HTTP 500"),
        (
            httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}),
            422,
            "unsupported_content",
            "application/pdf",
        ),
        (
            httpx.Response(200, html="<html><body><div id='app'></div></body></html>"),
            422,
            "insufficient_content",
            "JavaScript",
        ),
    ],
)
def test_website_errors_are_handled(make_client, world, page, status, code, text):
    world.pages["/"] = page
    response = analyse(make_client())
    assert response.status_code == status
    body = response.json()
    assert body["success"] is False and body["error"]["code"] == code
    assert text in body["error"]["message"]
    assert world.gemini_requests() == []  # no LLM cost for unusable sites


def test_website_timeout(make_client, world):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    world.handler = handler  # type: ignore[assignment]
    response = analyse(make_client())
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "website_timeout"


def test_large_page_is_truncated_not_loaded_fully(make_client, world):
    world.pages["/"] = httpx.Response(200, html=HOME_HTML + "<p>x</p>" * 200_000)
    response = analyse(make_client(make_settings(max_page_bytes=50_000)))
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------
def test_parse_html_extracts_business_information():
    page = parse_html(HOME_HTML.encode(), "https://acme.example.com/")
    assert page.title.startswith("Acme Web Studio")
    assert "small businesses" in page.meta_description
    assert page.og["site_name"] == "Acme Web Studio"
    assert page.lang == "en"
    assert page.organizations[0]["name"] == "Acme Web Studio"
    assert page.organizations[0]["address"]["addressLocality"] == "Ahmedabad"
    assert page.h1 == ["Websites that grow small businesses"]
    assert "E-commerce Stores" in page.h2
    assert "Services" in page.nav_items
    assert any("Ahmedabad" in line for line in page.footer_lines)
    assert page.social_links["Instagram"].startswith("https://www.instagram.com/")

    text = "\n".join(page.text_lines)
    assert "SHOULD_NOT_APPEAR" not in text and "color: red" not in text  # scripts/styles
    assert "cookies" not in text  # cookie banner removed
    assert text.count("Custom websites built with modern frameworks.") == 1  # de-duplicated


def test_pick_extra_pages_prefers_business_pages():
    page = parse_html(HOME_HTML.encode(), "https://acme.example.com/")
    picked = pick_extra_pages(page, 3)
    assert picked == ["https://acme.example.com/services", "https://acme.example.com/about-us"]


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------
def test_successful_analysis_returns_all_five_sections(make_client, world):
    response = analyse(make_client())
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["success"] is True
    assert body["website"] == "https://acme.example.com/"
    assert body["brand"]["name"] == "Acme Web Studio"
    assert set(body["analysis"]) == {
        "competitor_analysis",
        "target_search_queries",
        "seo_improvements",
        "top_services",
        "target_audience",
    }

    competitors = body["analysis"]["competitor_analysis"]["competitors"]
    names = [c["name"] for c in competitors]
    assert names == ["WebPixel Solutions", "Shopify", "Digital Gujarat Agency"]
    assert all(c["source"] == "ai_suggestion" for c in competitors)
    assert competitors[1]["website"] is None  # malformed URL dropped
    assert competitors[0]["confidence"] == "medium"
    table = body["analysis"]["competitor_analysis"]["comparison_table"]
    assert len(table) == 3
    assert [r["comparison"] for r in table[0]["rows"]] == [
        "Brand Name",
        "Main Service",
        "Target Market",
        "Key Strength",
        "Opportunity / Gap",
    ]

    queries = body["analysis"]["target_search_queries"]
    assert len(queries) == 5
    assert queries[4]["platforms"] == ["X", "Google"]  # aliases normalised
    assert not any("volume" in json.dumps(q).lower() for q in queries)

    seo = body["analysis"]["seo_improvements"]
    assert seo["current_title"].startswith("Acme Web Studio")
    assert seo["recommended_title_length"] == len(seo["recommended_title"])

    services = body["analysis"]["top_services"]
    assert len(services) == 5 and services[3]["source"] == "website"
    assert any("Google Ads Management" in n for n in body["data_notes"])
    assert any("No search volume" in n for n in body["data_notes"])
    assert any("AI-generated suggestions" in n for n in body["data_notes"])

    meta = body["meta"]
    assert meta["model"] == "gemini-3.8-flash"
    assert meta["input_tokens"] == 4000 and meta["output_tokens"] == 3000
    assert meta["estimated_cost_usd"] == pytest.approx(4000 / 1e6 * 0.75 + 3000 / 1e6 * 3.75)
    assert meta["cached"] is False
    assert len(meta["pages_analyzed"]) == 3  # homepage + services + about


def test_llm_receives_website_content_and_schema(make_client, world):
    analyse(make_client())
    body = json.loads(world.gemini_requests()[0].content)
    prompt = body["contents"][0]["parts"][0]["text"]
    assert "Websites that grow small businesses" in prompt  # homepage text
    assert "Website maintenance and hosting plans" in prompt  # extra page text
    assert "Ahmedabad" in prompt  # JSON-LD / footer location
    assert "SHOULD_NOT_APPEAR" not in prompt
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == LLM_SCHEMA
    assert "comparison_table" not in json.dumps(LLM_SCHEMA)  # server-only field


def test_llm_schema_keeps_fields_named_title():
    """Schema keywords 'title' are stripped, but a *field* called 'title' must stay."""
    item = LLM_SCHEMA["$defs"]["SupportingContent"]
    assert "title" in item["properties"] and "title" in item["required"]
    assert "title" not in item  # the keyword itself was removed
    for name, definition in LLM_SCHEMA["$defs"].items():
        assert set(definition.get("required", [])) <= set(definition["properties"]), name


def test_invalid_llm_output_is_retried_with_feedback(make_client, world):
    broken = llm_payload()
    broken["top_services"] = broken["top_services"][:3]  # only 3 services
    world.gemini = [gemini_ok(broken), gemini_ok(llm_payload())]
    response = analyse(make_client())
    assert response.status_code == 200
    calls = world.gemini_requests()
    assert len(calls) == 2
    retry_prompt = json.loads(calls[1].content)["contents"][0]["parts"][0]["text"]
    assert "previous answer was rejected" in retry_prompt and "top_services" in retry_prompt
    assert response.json()["meta"]["input_tokens"] == 8000  # usage of both calls


def test_llm_output_invalid_twice_returns_502(make_client, world):
    not_json = httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": "sorry, no"}]}}]}
    )
    world.gemini = [not_json, copy.deepcopy(not_json)]
    response = analyse(make_client())
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_error"


def test_rate_limited_primary_model_falls_back(make_client, world):
    world.gemini = [
        httpx.Response(429, json={"error": {"message": "Quota exceeded"}}),
        gemini_ok(llm_payload(), model="gemini-3.5-flash-lite"),
    ]
    response = analyse(make_client())
    assert response.status_code == 200
    urls = [str(r.url) for r in world.gemini_requests()]
    assert "gemini-3.8-flash:generateContent" in urls[0]
    assert "gemini-3.5-flash-lite:generateContent" in urls[1]
    assert response.json()["meta"]["model"] == "gemini-3.5-flash-lite"


def test_quota_exceeded_on_all_models(make_client, world):
    quota = httpx.Response(429, json={"error": {"message": "Quota exceeded"}})
    world.gemini = [quota, copy.deepcopy(quota)]
    response = analyse(make_client())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_rate_limited"


def test_invalid_api_key_is_reported_without_leaking_it(make_client, world):
    world.gemini = [httpx.Response(403, json={"error": {"message": "API key not valid"}})]
    response = analyse(make_client())
    assert response.status_code == 502
    assert "GEMINI_API_KEY" in response.json()["error"]["message"]
    assert API_KEY not in response.text
    assert len(world.gemini_requests()) == 1  # auth errors do not try the fallback


def test_overloaded_primary_is_retried_before_fallback(make_client, world):
    world.gemini = [httpx.Response(503, json={"error": {"message": "overloaded"}})]
    response = analyse(make_client())
    assert response.status_code == 200
    urls = [str(r.url) for r in world.gemini_requests()]
    assert len(urls) == 2 and all("gemini-3.8-flash:" in u for u in urls)


def test_primary_overloaded_twice_uses_fallback(make_client, world):
    overloaded = httpx.Response(503, json={"error": {"message": "overloaded"}})
    world.gemini = [overloaded, copy.deepcopy(overloaded)]
    response = analyse(make_client())
    assert response.status_code == 200
    urls = [str(r.url) for r in world.gemini_requests()]
    assert len(urls) == 3 and "gemini-3.5-flash-lite" in urls[2]


def test_server_error_retried_once_without_fallback(make_client, world):
    world.gemini = [httpx.Response(503, json={"error": {"message": "overloaded"}})]
    response = analyse(make_client(make_settings(gemini_fallback_model=None)))
    assert response.status_code == 200
    urls = [str(r.url) for r in world.gemini_requests()]
    assert len(urls) == 2 and all("gemini-3.8-flash" in u for u in urls)


def test_schema_rejection_falls_back_to_prompt_schema(make_client, world):
    world.gemini = [httpx.Response(400, json={"error": {"message": "Unknown field"}})]
    response = analyse(make_client())
    assert response.status_code == 200
    retry = json.loads(world.gemini_requests()[1].content)
    assert "responseJsonSchema" not in retry["generationConfig"]
    assert "thinkingConfig" not in retry["generationConfig"]
    assert "matches this JSON Schema" in retry["contents"][0]["parts"][0]["text"]


def test_llm_timeout(make_client, world):
    def handler(request):
        if request.url.host == GEMINI_HOST:
            raise httpx.ReadTimeout("slow", request=request)
        return FakeWorld.handler(world, request)

    world.handler = handler  # type: ignore[assignment]
    response = analyse(make_client())
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "llm_timeout"


def test_missing_api_key_fails_before_fetching(make_client, world):
    response = analyse(make_client(make_settings(gemini_api_key=None)))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"
    assert world.requests == []


def test_results_are_cached(make_client, world):
    client = make_client()
    first = analyse(client)
    count = len(world.requests)
    second = analyse(client)
    assert second.status_code == 200
    assert len(world.requests) == count  # no new website or LLM requests
    assert second.json()["meta"]["cached"] is True
    assert second.json()["analysis"] == first.json()["analysis"]


def test_rate_limit(make_client):
    client = make_client(rate_limit=2)
    assert analyse(client).status_code == 200
    assert analyse(client).status_code == 200
    third = analyse(client)
    assert third.status_code == 429
    assert third.json()["error"]["code"] == "rate_limited"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def test_platform_aliases():
    query = SearchQuery(
        query="q",
        platforms=["twitter", "Google Search", "X (Twitter)"],
        intent="i",
        business_relevance="b",
        target_audience="t",
    )
    assert query.platforms == ["X", "Google"]


def test_price_lookup_handles_versioned_model_names():
    settings = make_settings()
    assert settings.price_for("gemini-3.8-flash-001") == (0.75, 3.75)
    assert settings.price_for("models/gemini-3.5-flash-lite") == (0.30, 2.50)
    assert settings.price_for("unknown-model") is None


def test_health_root_and_unknown_routes(make_client):
    client = make_client()
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["llm_configured"] is True
    assert client.get("/", follow_redirects=False).headers["location"] == "/docs"
    assert client.get("/nope").json()["error"]["code"] == "not_found"
    assert client.get("/link-analysis").status_code == 405


def test_all_models_overloaded_returns_503_unavailable(make_client, world):
    overloaded = httpx.Response(503, json={"error": {"message": "high demand"}})
    world.gemini = [overloaded, copy.deepcopy(overloaded), copy.deepcopy(overloaded)]
    response = analyse(make_client())
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "llm_unavailable"
    assert "temporarily unavailable" in body["error"]["message"]


def test_default_models_are_free_tier():
    settings = Settings(_env_file=None)
    assert settings.gemini_model == "gemini-3.5-flash-lite"
    assert settings.gemini_fallback_model == "gemini-3.8-flash"


def test_client_ip_prefers_cloudflare_headers(monkeypatch):
    from starlette.requests import Request

    from app import main

    def request(headers):
        raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        return Request({"type": "http", "headers": raw, "client": ("10.0.0.9", 1234)})

    monkeypatch.setattr(main.settings, "trust_proxy_headers", True)
    assert main.client_ip(request({"True-Client-IP": "1.1.1.1", "X-Forwarded-For": "9.9.9.9"})) == (
        "1.1.1.1"
    )
    assert main.client_ip(request({"CF-Connecting-IP": "2.2.2.2"})) == "2.2.2.2"
    assert main.client_ip(request({"X-Forwarded-For": "3.3.3.3, 10.1.1.1"})) == "3.3.3.3"
    assert main.client_ip(request({})) == "10.0.0.9"
    monkeypatch.setattr(main.settings, "trust_proxy_headers", False)
    assert main.client_ip(request({"True-Client-IP": "1.1.1.1"})) == "10.0.0.9"
