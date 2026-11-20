"""FastAPI entry point.

Run locally:  uvicorn app.main:app --host 0.0.0.0 --port 4000
Docs:         http://localhost:4000/docs
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.errors import AppError, RateLimitedError
from app.schemas import ErrorResponse, LinkAnalysisRequest, LinkAnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.llm_service import GeminiService
from app.services.website_extractor import WebsiteExtractor
from app.utils.cache import RateLimiter, TTLCache

settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    fetch_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.fetch_timeout_seconds, connect=5.0),
        follow_redirects=False,  # redirects are validated one by one (SSRF protection)
        limits=httpx.Limits(max_connections=20),
    )
    llm_client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)
    async with fetch_client, llm_client:
        llm = GeminiService(settings, llm_client)
        app.state.analysis_service = AnalysisService(
            settings,
            WebsiteExtractor(settings, fetch_client),
            llm,
            TTLCache(settings.cache_ttl_seconds, settings.cache_max_entries),
        )
        app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)
        logger.info(
            "Started: LLM %s (fallback %s), configured=%s",
            settings.gemini_model,
            settings.gemini_fallback_model,
            llm.is_configured,
        )
        yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Analyses a website URL with an LLM (Google Gemini) and returns five sections: "
        "competitor analysis, top 5 target search queries, SEO improvements, top 5 services "
        "and target audience."
    ),
    lifespan=lifespan,
)


def get_analysis_service(request: Request) -> AnalysisService:
    return request.app.state.analysis_service


def client_ip(request: Request) -> str:
    """Client IP used for rate limiting.

    Behind Render, Cloudflare sets True-Client-IP / CF-Connecting-IP to the real client
    address. X-Forwarded-For is only a fallback (Render appends to it, so the first entry
    is the client but the header can be pre-filled by the caller).
    """
    if settings.trust_proxy_headers:
        for header in ("true-client-ip", "cf-connecting-ip"):
            value = request.headers.get(header, "").strip()
            if value:
                return value
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    if not request.app.state.rate_limiter.allow(client_ip(request)):
        raise RateLimitedError(
            f"Too many requests: limit is {settings.rate_limit_per_minute} analyses per minute."
        )


ServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = ErrorResponse(error={"code": code, "message": message, "details": details})
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    log = logger.warning if exc.status_code >= 500 else logger.info
    log("%s %s -> %s: %s", request.method, request.url.path, exc.code, exc.message)
    return _error(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err.get("loc", ()) if p != "body") or "body"
        details.append({"field": field, "message": str(err.get("msg", "Invalid value"))})
    message = "; ".join(f"{d['field']}: {d['message']}" for d in details)
    return _error(422, "validation_error", message, details)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    codes = {404: "not_found", 405: "method_not_allowed"}
    return _error(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "internal_error", "Internal server error")


_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "URL blocked (private/internal address)"},
    422: {"model": ErrorResponse, "description": "Invalid URL or website content not usable"},
    429: {"model": ErrorResponse, "description": "Too many requests"},
    502: {"model": ErrorResponse, "description": "Website unreachable or LLM error"},
    503: {"model": ErrorResponse, "description": "LLM not configured or quota exceeded"},
    504: {"model": ErrorResponse, "description": "Website or LLM timeout"},
}


@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse,
    responses=_ERROR_RESPONSES,
    summary="Analyse a website with an LLM",
    tags=["Link Analysis"],
    dependencies=[Depends(enforce_rate_limit)],
)
async def link_analysis(payload: LinkAnalysisRequest, service: ServiceDep) -> LinkAnalysisResponse:
    logger.info("Analysis requested for %s", payload.url)
    return await service.analyze(payload.url)


@app.get("/health", tags=["Health"], summary="Health check")
async def health(service: ServiceDep) -> dict:
    return {
        "status": "ok",
        "llm_configured": service.llm.is_configured,
        "model": settings.gemini_model,
    }


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
