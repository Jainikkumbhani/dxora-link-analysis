"""Google Gemini integration (REST API via httpx - no SDK dependency).

Docs: https://ai.google.dev/api/generate-content
      https://ai.google.dev/gemini-api/docs/structured-output

Features:
  * JSON mode + JSON Schema (``responseJsonSchema``) for structured output;
  * temporary 503 on the primary model: one retry after a short pause, then the
    fallback model; quota (429), missing model (404) or timeout: fallback directly;
  * if the API rejects optional settings (schema / thinking level) the call is
    repeated once without them, with the schema embedded in the prompt instead;
  * measured token usage (thinking tokens counted as billed output tokens).
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from app.config import Settings
from app.errors import (
    LLMError,
    LLMNotConfiguredError,
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)


class LLMOutputError(LLMError):
    """The model answered, but not with usable JSON."""


class _ModelUnavailable(LLMError):
    """Model not found / server error - try the fallback model."""


@dataclass
class LLMResult:
    data: Dict[str, Any]
    model: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    seconds: float


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_text(text: str) -> Dict[str, Any]:
    """Parse model text as a JSON object (tolerates ```json fences)."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        data = json.loads(cleaned)
    except ValueError as exc:
        raise LLMOutputError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMOutputError("Model returned JSON that is not an object.")
    return data


class GeminiService:
    provider = "Google Gemini"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    @property
    def is_configured(self) -> bool:
        key = self.settings.gemini_api_key
        return bool(key and key.get_secret_value().strip())

    @property
    def models(self) -> List[str]:
        primary = self.settings.gemini_model
        fallback = self.settings.gemini_fallback_model
        return [primary] + ([fallback] if fallback and fallback != primary else [])

    async def generate_json(
        self, system: str, prompt: str, schema: Optional[Dict[str, Any]] = None
    ) -> LLMResult:
        if not self.is_configured:
            raise LLMNotConfiguredError(
                "LLM is not configured: set GEMINI_API_KEY in the environment."
            )
        last_error: Optional[LLMError] = None
        for model in self.models:
            try:
                return await self._call_with_retry(model, system, prompt, schema)
            except (LLMRateLimitedError, LLMTimeoutError, _ModelUnavailable) as exc:
                logger.warning("Gemini model %s failed (%s); trying fallback", model, exc.message)
                last_error = exc
        assert last_error is not None
        if isinstance(last_error, _ModelUnavailable):
            raise LLMUnavailableError(
                "The AI model is temporarily unavailable, please retry later "
                f"({last_error.message})"
            )
        raise last_error

    async def _call_with_retry(
        self, model: str, system: str, prompt: str, schema: Optional[Dict[str, Any]]
    ) -> LLMResult:
        try:
            return await self._call(model, system, prompt, schema, optional=True)
        except _ModelUnavailable as exc:
            # "High demand" 503s on the primary model are usually short spikes, so the
            # primary is retried once after a short pause before switching to the
            # fallback model. Missing models (404) are never retried.
            if "HTTP 404" in exc.message or model != self.models[0]:
                raise
            logger.info("Gemini %s unavailable, retrying once: %s", model, exc.message)
            await asyncio.sleep(self.settings.llm_retry_delay_seconds)
            return await self._call(model, system, prompt, schema, optional=True)

    async def _call(
        self,
        model: str,
        system: str,
        prompt: str,
        schema: Optional[Dict[str, Any]],
        optional: bool,
    ) -> LLMResult:
        generation_config: Dict[str, Any] = {
            "responseMimeType": "application/json",
            "temperature": self.settings.llm_temperature,
            "maxOutputTokens": self.settings.llm_max_output_tokens,
        }
        user_text = prompt
        if schema is not None:
            if optional:
                generation_config["responseJsonSchema"] = schema
            else:
                user_text += (
                    "\n\nReturn JSON that matches this JSON Schema exactly:\n"
                    + json.dumps(schema, separators=(",", ":"))
                )
        if optional and self.settings.gemini_thinking_level:
            generation_config["thinkingConfig"] = {
                "thinkingLevel": self.settings.gemini_thinking_level
            }

        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": generation_config,
        }
        key = self.settings.gemini_api_key
        started = time.perf_counter()
        try:
            response = await self.client.post(
                f"{self.settings.gemini_api_base}/models/{model}:generateContent",
                headers={"x-goog-api-key": key.get_secret_value() if key else ""},
                json=body,
                timeout=self.settings.llm_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"The LLM did not respond within {self.settings.llm_timeout_seconds:g} seconds."
            ) from exc
        except httpx.HTTPError as exc:
            raise _ModelUnavailable(f"Could not reach the LLM API ({type(exc).__name__}).") from exc
        seconds = time.perf_counter() - started

        if response.status_code == 400 and optional:
            logger.warning(
                "Gemini rejected optional settings (%s); retrying without them",
                _api_error_message(response),
            )
            return await self._call(model, system, prompt, schema, optional=False)
        self._raise_for_status(response, model)

        payload = response.json()
        candidates = payload.get("candidates") or []
        if not candidates:
            reason = (payload.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise LLMError(f"The LLM returned no answer ({reason}).")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise LLMOutputError("The LLM response was cut off (output token limit reached).")
        data = parse_json_text(text)

        usage = payload.get("usageMetadata") or {}
        output_tokens = None
        if "candidatesTokenCount" in usage or "thoughtsTokenCount" in usage:
            output_tokens = usage.get("candidatesTokenCount", 0) + usage.get(
                "thoughtsTokenCount", 0
            )
        logger.info(
            "Gemini %s: %.1fs, input=%s output=%s tokens",
            model,
            seconds,
            usage.get("promptTokenCount"),
            output_tokens,
        )
        return LLMResult(
            data=data,
            model=payload.get("modelVersion") or model,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=output_tokens,
            seconds=seconds,
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response, model: str) -> None:
        status = response.status_code
        if status < 400:
            return
        message = _api_error_message(response)
        if status in (401, 403):
            raise LLMError(f"LLM authentication failed - check GEMINI_API_KEY ({message}).")
        if status == 429:
            raise LLMRateLimitedError(f"LLM quota or rate limit exceeded ({message}).")
        if status == 404:
            raise _ModelUnavailable(f"Model '{model}' not available: HTTP 404 ({message}).")
        if status >= 500:
            raise _ModelUnavailable(f"LLM service error HTTP {status} ({message}).")
        raise LLMError(f"LLM request rejected: HTTP {status} ({message}).")


def _api_error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        return str(error.get("message") or error.get("status") or response.reason_phrase)[:300]
    except ValueError:
        return response.reason_phrase or "unknown error"
