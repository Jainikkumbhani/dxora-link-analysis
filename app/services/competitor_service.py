"""Post-processing of LLM competitor output.

The LLM suggests competitors from its general knowledge. This module makes the
result trustworthy and transparent:
  * removes duplicates and the analysed brand itself (same name or domain);
  * drops malformed competitor URLs instead of returning them;
  * labels every competitor's source: ``mentioned_on_website`` when the name
    appears in the extracted website text, otherwise ``ai_suggestion``;
  * builds the "Your Brand vs Competitor" comparison table.
"""

import re
from typing import List, Optional
from urllib.parse import urlsplit

from app.errors import AppError
from app.schemas import ComparisonRow, CompetitorAnalysis, CompetitorComparison
from app.utils.validators import normalize_url

MIN_COMPETITORS = 3


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _domain(url: Optional[str]) -> str:
    if not url:
        return ""
    host = (urlsplit(url if "://" in url else f"https://{url}").hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _safe_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        return normalize_url(url)
    except AppError:
        return None


def _first(items: List[str], default: str = "-") -> str:
    return items[0] if items else default


def finalize_competitors(
    analysis: CompetitorAnalysis, website_url: str, website_text: str
) -> List[str]:
    """Clean competitors in place, build the comparison table, return data notes."""
    brand_key = _key(analysis.brand_name)
    site_domain = _domain(website_url)
    text_lower = website_text.lower()

    kept = []
    seen = set()
    for competitor in analysis.competitors:
        key = _key(competitor.name)
        competitor.website = _safe_url(competitor.website)
        if not key or key in seen or key == brand_key:
            continue
        if competitor.website and _domain(competitor.website) == site_domain:
            continue
        seen.add(key)
        mentioned = len(key) >= 4 and competitor.name.lower() in text_lower
        competitor.source = "mentioned_on_website" if mentioned else "ai_suggestion"
        kept.append(competitor)
    analysis.competitors = kept

    analysis.comparison_table = [
        CompetitorComparison(
            competitor_name=c.name,
            rows=[
                ComparisonRow(
                    comparison="Brand Name", your_brand=analysis.brand_name, competitor=c.name
                ),
                ComparisonRow(
                    comparison="Main Service",
                    your_brand=_first(analysis.main_products_services),
                    competitor=c.main_service,
                ),
                ComparisonRow(
                    comparison="Target Market",
                    your_brand=analysis.brand_target_market,
                    competitor=c.target_market,
                ),
                ComparisonRow(
                    comparison="Key Strength",
                    your_brand=analysis.brand_key_strength,
                    competitor=_first(c.strengths),
                ),
                ComparisonRow(
                    comparison="Opportunity / Gap",
                    your_brand=_first(analysis.opportunities),
                    competitor=_first(c.weaknesses_or_gaps),
                ),
            ],
        )
        for c in kept
    ]

    notes = [
        "Competitors are AI-generated suggestions based on the model's general knowledge "
        "(source 'ai_suggestion'); they are not taken from live market research and should be "
        "verified. Competitors named on the website itself are labelled 'mentioned_on_website'."
    ]
    if len(kept) < MIN_COMPETITORS:
        notes.append(
            f"Only {len(kept)} competitor(s) returned: reliable information about more "
            "competitors was not available."
        )
    return notes
