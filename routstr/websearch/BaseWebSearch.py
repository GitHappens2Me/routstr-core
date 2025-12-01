"""
Web search and scraping module for AI context enhancement.

This module provides web search functionality to enhance AI responses with
current information from the web. It includes dummy implementations for
testing and development.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from ..core.logging import get_logger

from ..core.settings import settings

logger = get_logger(__name__)

# Conditional imports for web scraper
try:
    from .BaseWebScraper import BaseWebScraper, GenericWebScraper
    WEB_SCRAPER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Web scraper not available: {e}")
    BaseWebScraper = None
    GenericWebScraper = None
    WEB_SCRAPER_AVAILABLE = False


@dataclass
class WebSearchResult:
    """Represents a single web search result."""
    title: str
    url: str
    snippet: str
    published_date: Optional[str] = None
    relevance_score: float = 0.0
    content: Optional[str] = None


@dataclass
class WebSearchResponse:
    """Complete web search response with context for AI."""
    query: str
    results: List[WebSearchResult]
    summary: str
    total_results: int
    search_time_ms: int
    timestamp: str


class BaseWebSearch:
    """Base class for web search providers."""
    
    provider_name: str = "Base"
    def __init__(self, scraper: Optional[BaseWebScraper] = None):
        """
        Initialize the web search provider.
        """
        # Use the provided scraper or default to a GenericWebScraper (NOT BaseWebScraper)
        if not WEB_SCRAPER_AVAILABLE:
            raise ImportError("Web scraper functionality not available. Install websearch dependencies.")
        self.scraper = scraper or GenericWebScraper()  # Changed from BaseWebScraper()
        self.web_scraper = self.scraper
        logger.info(f"WebSearch initialized with scraper: {self.scraper.scraper_name}")
        
    
    async def search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """Perform web search and return results."""
        raise NotImplementedError("Subclasses must implement search method")
    
        
    async def inject_web_context(self, request_body: bytes, query: str = None) -> bytes:
        """
        Enhance AI request with web search context by injecting system message.
        
        Args:
            request_body: The original request body as bytes
            query: Optional search query. If None, will extract from user message
            
        Returns:
            Enhanced request body with web context injected as bytes
        """
        try:
            # Parse the request body
            data = json.loads(request_body)
            
            # Extract query from user message if not provided
            if query is None:
                query = _extract_query_from_messages(data.get("messages", []))
            
            if not query:
                logger.warning("No query found for web search enhancement")
                return request_body
            
            # Perform web search and scraping

            search_results = await self.search(query)

            #TODO, use the other fields as well
            search_results = search_results.results
            urls_to_scrape = [result.url for result in search_results]
        
            # 3. Scrape all URLs concurrently using the properly initialized scraper
            logger.info(f"Scraping {len(urls_to_scrape)} URLs")
            
            max_concurrent_scrapes = settings.web_scrape_max_concurrent_urls
            scraped_contents = await self.scraper.scrape_urls(urls_to_scrape, max_concurrent=max_concurrent_scrapes)
            
            content_map = {scraped.url: scraped.content for scraped in scraped_contents}
            
            # Iterate through the original results and add the content
            for result in search_results:
                result.content = content_map.get(result.url) # Returns None if URL was not scraped successfully

            
            context_parts = []
            for result in search_results:
                if result.content:
                    # Use the title, snippet, and the newly added full content
                    context_parts.append(
                        f"Source: {result.title} ({result.url})\n"
                        f"Snippet: {result.snippet}\n"
                        #TODO: More sensible truncation
                        f"Full Content: {result.content[:1500]}..." # Truncate for brevity
                    )
            web_context = "\n\n---\n\n".join(context_parts) if context_parts else "Web search was performed, but no content could be successfully scraped."
            web_message = {
                "role": "system",
                "content": f"Web search context for query '{query}':\n{web_context}"
            }
            if "messages" in data:
                data["messages"].insert(0, web_message)
            # Convert back to bytes
            enhanced_body = json.dumps(data).encode()

            logger.info(
                f"Request enhanced with web search context",
                extra={
                    "query": query,
                    "original_size": len(request_body),
                    "enhanced_size": len(enhanced_body),
                    "total_urls": len(urls_to_scrape),
                    "scraped_successfully": len(scraped_contents)
                }
            )
            print(f"original_size {len(request_body)},enhanced_size {len(enhanced_body)}")
            
            return enhanced_body
            
        except Exception as e:
            logger.error(
                f"Failed to enhance request with web context: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "query": query
                }
            )
            return request_body


def _extract_query_from_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Extract search query from user messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Extracted query string or empty string if not found
    """
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            # Simple extraction - use the full user message as query
            # Could be enhanced with NLP to extract the actual question
            return content.strip()
    return ""