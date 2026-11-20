"""Request / response models and the schema the LLM output must satisfy."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils.validators import normalize_url

Platform = Literal["Google", "Facebook", "Instagram", "LinkedIn", "X"]
PLATFORM_ALIASES = {
    "google": "Google",
    "google search": "Google",
    "facebook": "Facebook",
    "instagram": "Instagram",
    "linkedin": "LinkedIn",
    "x": "X",
    "twitter": "X",
    "x (twitter)": "X",
    "x/twitter": "X",
}


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class LinkAnalysisRequest(BaseModel):
    url: str = Field(
        ...,
        description="Public website URL to analyse (http/https). 'example.com' is also accepted.",
        examples=["https://example.com"],
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return normalize_url(value)  # raises InvalidURLError / BlockedURLError


# ---------------------------------------------------------------------------
# LLM output (validated before it is returned to the client)
# ---------------------------------------------------------------------------
class _LLMModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class Brand(_LLMModel):
    name: str = Field(..., description="Brand / company name as shown on the website.")
    category: str = Field(..., description="Main business category, e.g. 'Web Development Agency'.")
    summary: str = Field(..., description="One or two sentence business summary.")


class Competitor(_LLMModel):
    name: str
    website: Optional[str] = Field(None, description="Competitor homepage if known, else null.")
    main_service: str
    target_market: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses_or_gaps: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = Field(
        "medium", description="How confident the model is that this is a real, relevant competitor."
    )
    source: str = Field(
        "ai_suggestion",
        description="Always 'ai_suggestion' (set by the server after verification).",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.lower().strip() if isinstance(value, str) else value


class ComparisonRow(BaseModel):
    comparison: str
    your_brand: str
    competitor: str


class CompetitorComparison(BaseModel):
    competitor_name: str
    rows: List[ComparisonRow]


class CompetitorAnalysis(_LLMModel):
    brand_name: str
    brand_overview: str = Field(..., description="Business overview of the submitted brand.")
    business_category: str
    main_products_services: List[str]
    brand_target_market: str
    brand_key_strength: str
    competitors: List[Competitor] = Field(
        ..., description="At least 3 relevant competitors where reliable information exists."
    )
    strengths: List[str] = Field(..., description="The submitted brand's competitive advantages.")
    opportunities: List[str] = Field(..., description="Ways to improve market positioning.")
    # Filled by the server (competitor_service), not by the LLM.
    comparison_table: List[CompetitorComparison] = Field(default_factory=list)


class SearchQuery(_LLMModel):
    query: str
    platforms: List[Platform] = Field(..., min_length=1)
    intent: str = Field(..., description="Search intent, e.g. 'Service Search', 'B2B Search'.")
    business_relevance: str
    target_audience: str

    @field_validator("platforms", mode="before")
    @classmethod
    def _normalise_platforms(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        out: List[str] = []
        for item in value:
            name = PLATFORM_ALIASES.get(str(item).strip().lower(), item)
            if name not in out:
                out.append(name)
        return out


class SupportingContent(_LLMModel):
    content_type: str = Field(
        ...,
        description=(
            "One of: Service landing page, Industry-specific page, FAQ, Blog topic, "
            "Location-based landing page, Supporting business article."
        ),
    )
    title: str
    description: str
    target_keyword: str


class SEOImprovements(_LLMModel):
    primary_keyword: str
    secondary_keywords: List[str]
    long_tail_keywords: List[str]
    business_keywords: List[str]
    recommended_title: str = Field(..., description="SEO title, ideally 50-60 characters.")
    title_primary_keyword: str
    title_structure: str = Field(..., description="Pattern used, e.g. 'Primary Keyword | Brand'.")
    meta_description: str = Field(..., description="Recommended meta description, 140-160 chars.")
    meta_summary: str = Field(..., description="Short, human-friendly business summary.")
    meta_keywords: List[str]
    supporting_content: List[SupportingContent]
    issues_found: List[str] = Field(
        default_factory=list, description="Concrete SEO issues seen in the extracted content."
    )
    # Filled by the server from the real page (not by the LLM).
    current_title: Optional[str] = None
    current_meta_description: Optional[str] = None
    recommended_title_length: Optional[int] = None
    meta_description_length: Optional[int] = None


class Service(_LLMModel):
    service_name: str
    description: str
    target_industry: str
    target_customer: str
    main_benefit: str
    source: Literal["website", "inferred"] = Field(
        ...,
        description=(
            "'website' if stated in the website content, 'inferred' if it is a recommendation."
        ),
    )

    @field_validator("source", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.lower().strip() if isinstance(value, str) else value


class CustomerProfile(_LLMModel):
    profile: str
    description: str


class AudienceSegment(_LLMModel):
    audience: str
    business_type: str
    requirement: str
    recommended_channels: List[str] = Field(default_factory=list)


class TargetAudience(_LLMModel):
    primary_audience: List[str]
    secondary_audience: List[str]
    industries: List[str]
    business_types: List[str]
    customer_profiles: List[CustomerProfile]
    geographic_focus: List[str] = Field(
        default_factory=list, description="Locations stated or clearly implied; empty if unknown."
    )
    geographic_focus_source: Literal["website", "inferred", "not_available"] = "not_available"
    customer_needs: List[str]
    customer_challenges: List[str]
    customer_interests: List[str]
    recommended_segments: List[AudienceSegment]

    @field_validator("geographic_focus_source", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.lower().strip() if isinstance(value, str) else value


class LLMAnalysis(_LLMModel):
    """Exact structure the LLM must return."""

    brand: Brand
    competitor_analysis: CompetitorAnalysis
    target_search_queries: List[SearchQuery] = Field(..., min_length=5, max_length=5)
    seo_improvements: SEOImprovements
    top_services: List[Service] = Field(..., min_length=5, max_length=5)
    target_audience: TargetAudience


# ---------------------------------------------------------------------------
# API response
# ---------------------------------------------------------------------------
class Analysis(BaseModel):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: List[SearchQuery]
    seo_improvements: SEOImprovements
    top_services: List[Service]
    target_audience: TargetAudience


class AnalysisMeta(BaseModel):
    provider: str
    model: str
    pages_analyzed: List[str]
    content_characters: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = Field(None, description="Includes billed thinking tokens.")
    estimated_cost_usd: Optional[float] = Field(
        None,
        description=(
            "Paid-tier equivalent cost computed from measured token usage and official "
            "per-token prices. On the Gemini free tier the actual charge is $0."
        ),
    )
    fetch_seconds: float
    llm_seconds: float
    total_seconds: float
    cached: bool = False
    analyzed_at: str


class LinkAnalysisResponse(BaseModel):
    success: Literal[True] = True
    website: str
    brand: Brand
    analysis: Analysis
    data_notes: List[str] = Field(
        default_factory=list, description="How each part was produced and its limitations."
    )
    meta: AnalysisMeta


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[List[Dict[str, Any]]] = None


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail
