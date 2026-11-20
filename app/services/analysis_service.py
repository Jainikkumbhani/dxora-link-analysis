"""Orchestrates the full flow: validate -> fetch -> extract -> LLM -> validate -> respond."""

import copy
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from app.config import Settings
from app.errors import LLMError, LLMNotConfiguredError
from app.schemas import (
    Analysis,
    AnalysisMeta,
    LinkAnalysisResponse,
    LLMAnalysis,
)
from app.services.competitor_service import finalize_competitors
from app.services.llm_service import GeminiService, LLMOutputError, LLMResult
from app.services.website_extractor import WebsiteContent, WebsiteExtractor
from app.utils.cache import TTLCache

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior digital-marketing, SEO and competitive-intelligence analyst.
You receive text extracted from a company's website and return a structured JSON analysis.

Rules:
1. Brand facts, services, SEO findings and audience MUST be based on the WEBSITE CONTENT
   provided. Do not invent facts (prices, clients, awards, years, locations, numbers) that
   are not in the content.
2. Competitors: name 3-5 real companies that compete in the same category and, when known,
   the same market or geography, using your general knowledge. Use confidence "low" when
   unsure. Set "website" only when you are confident of the exact domain, otherwise null.
   Describe strengths and gaps qualitatively - no invented statistics.
3. Search queries: exactly 5 realistic queries a potential customer would use to find this
   business. Each query lists every platform where it fits (usually 2-3), using only:
   "Google", "Facebook", "Instagram", "LinkedIn", "X". Together the 5 queries must cover
   all five platforms. NEVER include search volumes, rankings, traffic or any numeric estimates.
4. SEO: recommended_title 50-60 characters containing the primary keyword and the brand;
   meta_description 140-160 characters; 3-6 long-tail keywords; 5-8 supporting_content
   items covering at least four different content types; issues_found lists only issues
   evident from the SEO FACTS and content.
5. Services: exactly 5. Prefer services explicitly stated on the website (source "website").
   If fewer than 5 are stated, add closely related recommended services with
   source "inferred".
6. Audience: geographic_focus only if stated or clearly implied (address, phone code,
   currency, language); set geographic_focus_source to "website", "inferred" or
   "not_available". Give 4-5 recommended_segments.
7. Write in English. Be specific to THIS business - generic text is a failure.
   Lists should normally have 3-6 items.
"""

# Fields the server fills in; they are removed from the schema sent to the LLM.
_SERVER_FIELDS = {
    "CompetitorAnalysis": ["comparison_table"],
    "Competitor": ["source"],
    "SEOImprovements": [
        "current_title",
        "current_meta_description",
        "recommended_title_length",
        "meta_description_length",
    ],
}


def _strip_schema(node: Any) -> Any:
    """Remove schema keywords that add tokens but no constraints (title, default, examples).

    Keys inside a "properties" mapping are field names (a field may be called
    "title"), so they are kept.
    """
    if isinstance(node, list):
        return [_strip_schema(v) for v in node]
    if not isinstance(node, dict):
        return node
    out: Dict[str, Any] = {}
    for key, value in node.items():
        if key in ("properties", "$defs"):
            out[key] = {name: _strip_schema(sub) for name, sub in value.items()}
        elif key not in ("title", "default", "examples"):
            out[key] = _strip_schema(value)
    return out


def build_llm_schema() -> Dict[str, Any]:
    schema = copy.deepcopy(LLMAnalysis.model_json_schema())
    for def_name, fields in _SERVER_FIELDS.items():
        definition = schema.get("$defs", {}).get(def_name, {})
        for name in fields:
            definition.get("properties", {}).pop(name, None)
            if name in definition.get("required", []):
                definition["required"].remove(name)
    return _strip_schema(schema)


LLM_SCHEMA = build_llm_schema()


def build_user_prompt(content: WebsiteContent, content_text: str) -> str:
    home = content.homepage
    seo_facts = [
        f"Current title: {home.title or '(missing)'} ({len(home.title)} characters)",
        f"Current meta description: {home.meta_description or '(missing)'} "
        f"({len(home.meta_description)} characters)",
        f"Number of H1 headings on homepage: {len(home.h1)}",
        f"Pages analysed: {', '.join(content.pages_analyzed)}",
    ]
    return (
        "Analyse this website and return the JSON analysis.\n\n"
        "SEO FACTS (measured by the crawler):\n- " + "\n- ".join(seo_facts) + "\n\n"
        "WEBSITE CONTENT:\n<<<\n" + content_text + "\n>>>"
    )


def _short_errors(exc: ValidationError, limit: int = 8) -> str:
    parts = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)


class AnalysisService:
    def __init__(
        self,
        settings: Settings,
        extractor: WebsiteExtractor,
        llm: GeminiService,
        cache: TTLCache,
    ) -> None:
        self.settings = settings
        self.extractor = extractor
        self.llm = llm
        self.cache = cache

    async def analyze(self, url: str) -> LinkAnalysisResponse:
        cached = self.cache.get(url)
        if cached is not None:
            logger.info("Cache hit for %s", url)
            response = cached.model_copy(deep=True)
            response.meta.cached = True
            return response

        if not self.llm.is_configured:
            raise LLMNotConfiguredError(
                "LLM is not configured: set GEMINI_API_KEY in the environment."
            )
        started = time.perf_counter()
        content = await self.extractor.extract(url)
        fetch_seconds = time.perf_counter() - started

        content_text = content.to_prompt_text(self.settings.max_content_chars)
        prompt = build_user_prompt(content, content_text)
        llm_started = time.perf_counter()
        analysis, model, input_tokens, output_tokens = await self._run_llm(prompt)
        llm_seconds = time.perf_counter() - llm_started

        notes = self._post_process(analysis, content, content_text)
        cost = self._estimate_cost(model, input_tokens, output_tokens)
        response = LinkAnalysisResponse(
            website=content.final_url,
            brand=analysis.brand,
            analysis=Analysis(
                competitor_analysis=analysis.competitor_analysis,
                target_search_queries=analysis.target_search_queries,
                seo_improvements=analysis.seo_improvements,
                top_services=analysis.top_services,
                target_audience=analysis.target_audience,
            ),
            data_notes=[
                f"Brand, services, SEO and audience insights were generated by {model} from "
                f"content extracted from {len(content.pages_analyzed)} page(s) of the website.",
                *notes,
            ],
            meta=AnalysisMeta(
                provider=self.llm.provider,
                model=model,
                pages_analyzed=content.pages_analyzed,
                content_characters=len(content_text),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                fetch_seconds=round(fetch_seconds, 2),
                llm_seconds=round(llm_seconds, 2),
                total_seconds=round(time.perf_counter() - started, 2),
                analyzed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        self.cache.set(url, response)
        return response

    async def _run_llm(self, prompt: str) -> Tuple[LLMAnalysis, str, Optional[int], Optional[int]]:
        """Call the LLM; on invalid/incomplete output retry once with the errors as feedback."""
        results: List[LLMResult] = []
        feedback = ""
        for attempt in (1, 2):
            try:
                result = await self.llm.generate_json(SYSTEM_PROMPT, prompt + feedback, LLM_SCHEMA)
                results.append(result)
                analysis = LLMAnalysis.model_validate(result.data)
                return analysis, result.model, *self._sum_usage(results)
            except ValidationError as exc:
                problem = _short_errors(exc)
            except LLMOutputError as exc:
                problem = exc.message
            logger.warning("LLM output invalid (attempt %d): %s", attempt, problem)
            feedback = (
                "\n\nIMPORTANT: your previous answer was rejected because: "
                f"{problem}. Return the complete, corrected JSON object."
            )
        raise LLMError("The LLM did not return a valid analysis after 2 attempts.")

    @staticmethod
    def _sum_usage(results: List[LLMResult]) -> Tuple[Optional[int], Optional[int]]:
        def total(values: List[Optional[int]]) -> Optional[int]:
            known = [v for v in values if v is not None]
            return sum(known) if known else None

        return total([r.input_tokens for r in results]), total([r.output_tokens for r in results])

    def _post_process(
        self, analysis: LLMAnalysis, content: WebsiteContent, content_text: str
    ) -> List[str]:
        notes = finalize_competitors(analysis.competitor_analysis, content.final_url, content_text)

        seo = analysis.seo_improvements
        seo.current_title = content.homepage.title or None
        seo.current_meta_description = content.homepage.meta_description or None
        seo.recommended_title_length = len(seo.recommended_title)
        seo.meta_description_length = len(seo.meta_description)

        notes.append(
            "meta.estimated_cost_usd is the paid-tier equivalent cost; this deployment uses the "
            "Gemini free tier, so the actual charge is $0."
        )
        notes.append(
            "Search queries are AI suggestions based on the website content. No search volume, "
            "ranking or traffic figures are reported because no search-volume data source is "
            "connected."
        )
        inferred = [s.service_name for s in analysis.top_services if s.source == "inferred"]
        if inferred:
            notes.append(
                "Services labelled 'inferred' are recommendations, not explicitly offered on the "
                f"website: {', '.join(inferred)}."
            )
        if len(content_text) >= self.settings.max_content_chars:
            notes.append(
                f"Website text was trimmed to {self.settings.max_content_chars} characters to "
                "control LLM cost."
            )
        return notes

    def _estimate_cost(
        self, model: str, input_tokens: Optional[int], output_tokens: Optional[int]
    ) -> Optional[float]:
        price = self.settings.price_for(model)
        if price is None or input_tokens is None or output_tokens is None:
            return None
        cost = input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1]
        return round(cost, 6)
