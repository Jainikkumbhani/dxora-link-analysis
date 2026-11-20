"""Application configuration, loaded from environment variables / a local .env file."""

from functools import lru_cache
from typing import Dict, Optional, Tuple

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Official Gemini API prices, USD per 1M tokens (input, output), standard paid tier.
# Source: https://ai.google.dev/gemini-api/docs/pricing (checked 2026-11-20).
# Thinking tokens are billed as output tokens. On the FREE tier the actual charge is $0;
# these prices are used to report the paid-tier equivalent cost.
GEMINI_PRICING_USD_PER_1M: Dict[str, Tuple[float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75),  # promotional until 2026-12-31, then 1.50 / 7.50
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-4-hydrogen": (1.50, 7.50),
    "gemini-4-carbon": (2.00, 10.00),  # until 2026-11-30, then 4.00 / 20.00
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Website Link Analysis API"
    log_level: str = "INFO"

    # --- LLM (Google Gemini) ---------------------------------------------------
    gemini_api_key: Optional[SecretStr] = None
    # Free-tier models. Flash-Lite was the only reliable free model in our benchmark.
    gemini_model: str = "gemini-3.5-flash-lite"
    # Used automatically when the primary model is rate limited / unavailable.
    gemini_fallback_model: Optional[str] = "gemini-3.8-flash"
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    # "low" keeps latency and (billed) thinking tokens down. Empty = model default.
    gemini_thinking_level: str = "low"
    llm_temperature: float = 0.4
    llm_max_output_tokens: int = 8192
    llm_timeout_seconds: float = 45.0

    # --- Website fetching --------------------------------------------------------
    fetch_timeout_seconds: float = 15.0
    max_page_bytes: int = 3_000_000
    max_redirects: int = 5
    # Extra internal pages (about / services / products) fetched besides the homepage.
    max_extra_pages: int = 3
    # Maximum characters of cleaned website text sent to the LLM (token budget).
    max_content_chars: int = 14_000
    user_agent: str = (
        "Mozilla/5.0 (compatible; LinkAnalysisBot/1.0; +https://github.com/Jainikkumbhani)"
    )

    # --- Cost / abuse protection -------------------------------------------------
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 200
    rate_limit_per_minute: int = 10  # per client IP, 0 disables
    # Use X-Forwarded-For for the client IP. Enable only behind a proxy (Render does).
    trust_proxy_headers: bool = False

    def price_for(self, model: str) -> Optional[Tuple[float, float]]:
        """Price for a model id; versioned ids like 'gemini-3.8-flash-001' match their base."""
        name = model.split("/")[-1]
        matches = [key for key in GEMINI_PRICING_USD_PER_1M if name.startswith(key)]
        return GEMINI_PRICING_USD_PER_1M[max(matches, key=len)] if matches else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
