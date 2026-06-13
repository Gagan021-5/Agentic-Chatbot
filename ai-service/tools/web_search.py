import os
from typing import Dict, Any, List
from loguru import logger
from tavily import TavilyClient

class TavilySearchTool:
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("TAVILY_API_KEY environment variable is missing from your environment setup!")
            
    async def search_and_summarize(self, query: str) -> Dict[str, Any]:
        """Executes an asynchronous web search optimized for LLM ingestion profiles."""
        if not self.api_key:
            return {
                "summary": "Search currently unavailable: Missing active Tavily API Key.",
                "sources": []
            }
            
        try:
            client = TavilyClient(api_key=self.api_key)
            logger.info(f"[Tavily Search Engine] Executing lookups for query: '{query}'")
            
            response = client.search(
                query=query,
                search_depth="advanced",
                include_answer=True,
                max_results=5
            )
            
            summary = response.get("answer")
            sources: List[Dict[str, str]] = []
            
            for result in response.get("results", []):
                sources.append({
                    "title": result.get("title", "Marketplace Source Link"),
                    "url": result.get("url", "")
                })
                
            if not summary and response.get("results"):
                summary = " ".join([r.get("content", "") for r in response["results"][:3]])
                
            return {
                "summary": summary or "No analytical information gathered.",
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"[Tavily Search Error] Execution suspended: {e}")
            return {
                "summary": f"Web verification suspended due to a service request timeout: {str(e)}",
                "sources": []
            }

def get_web_search_tool():
    """Factory helper matching the legacy layout constructor."""
    return TavilySearchTool()
