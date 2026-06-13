"""
═══════════════════════════════════════════════════════════════
Web Search Tool — DuckDuckGo-based research with summarization
═══════════════════════════════════════════════════════════════
Agent decides when web search is required.
"""

import structlog
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

from config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class WebSearchTool:
    """Web research tool with search + summarization."""

    def __init__(self):
        self.ddgs = DDGS()
        genai.configure(api_key=settings.gemini_api_key)
        self.summarizer = genai.GenerativeModel("gemini-1.5-flash")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Perform web search and return results."""
        try:
            results = list(self.ddgs.text(
                query,
                max_results=max_results,
                region="wt-wt",
                safesearch="moderate",
            ))
            return [
                {
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", ""),
                }
                for r in results
            ]
        except Exception as e:
            logger.error("web_search_error", query=query, error=str(e))
            return []

    async def search_and_summarize(
        self,
        query: str,
        context: str | None = None,
        max_results: int = 5,
    ) -> dict:
        """Search the web and summarize results relevant to the context."""
        results = self.search(query, max_results)

        if not results:
            return {
                "query": query,
                "summary": "No relevant web results found.",
                "sources": [],
                "raw_results": [],
            }

        # Build context for summarization
        search_text = "\n\n".join([
            f"Source: {r['title']}\nURL: {r['href']}\nContent: {r['body']}"
            for r in results
        ])

        context_hint = f"\nUser's context: {context}" if context else ""

        try:
            response = self.summarizer.generate_content(
                f"""You are a research assistant for RentPrompts, an AI app marketplace.
Summarize the following web search results into actionable insights for building an AI app.
Focus on:
- Latest capabilities and best practices
- Prompt engineering tips specific to the technology
- Any limitations or gotchas
- Pricing or availability changes

Search query: "{query}"{context_hint}

Search Results:
{search_text}

Provide a concise, structured summary (3-5 bullet points). No fluff.""",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3,
                ),
            )
            summary = response.text
        except Exception as e:
            logger.error("summarization_error", error=str(e))
            summary = "\n".join([f"- {r['title']}: {r['body'][:100]}" for r in results[:3]])

        return {
            "query": query,
            "summary": summary,
            "sources": [{"title": r["title"], "url": r["href"]} for r in results],
            "raw_results": results,
        }

    def should_search(self, query: str, app_type: str | None = None) -> bool:
        """Heuristic: decide if web search is beneficial for this query.

        Returns True when the query likely needs fresh/external information.
        """
        query_lower = query.lower()

        # Signals that need fresh web data
        freshness_signals = [
            "latest", "newest", "recent", "update", "2025", "2026",
            "new feature", "just released", "announcement",
            "documentation", "api change", "deprecated",
        ]

        # Model/provider research signals
        research_signals = [
            "flux", "imagen", "dalle", "midjourney", "stable diffusion",
            "openai", "anthropic", "google", "meta", "mistral",
            "best practices", "how to", "tutorial",
            "comparison", "vs", "benchmark",
        ]

        if any(signal in query_lower for signal in freshness_signals):
            return True
        if any(signal in query_lower for signal in research_signals):
            return True

        return False


# Singleton
_web_search_tool: WebSearchTool | None = None


def get_web_search_tool() -> WebSearchTool:
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool
