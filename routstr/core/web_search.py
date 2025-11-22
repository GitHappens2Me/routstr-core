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

from .logging import get_logger
from .web_scraper import get_web_scraper_manager, ScrapedContent

logger = get_logger(__name__)


@dataclass
class WebSearchResult:
    """Represents a single web search result."""
    title: str
    url: str
    snippet: str
    published_date: Optional[str] = None
    relevance_score: float = 0.0


@dataclass
class WebSearchResponse:
    """Complete web search response with context for AI."""
    query: str
    results: List[WebSearchResult]
    summary: str
    total_results: int
    search_time_ms: int
    timestamp: str


class WebSearchProvider:
    """Base class for web search providers."""
    
    provider_name: str = "base"
    
    async def search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """Perform web search and return results."""
        raise NotImplementedError("Subclasses must implement search method")


class DummyWebSearchProvider(WebSearchProvider):
    """Dummy web search provider for testing and development."""
    
    provider_name = "dummy"
    
    def __init__(self):
        """Initialize dummy provider with sample data."""
        self.dummy_data = {
            "what time is it in new zealand": [
                WebSearchResult(
                    title="Time is",
                    url="https://time.is/de/NZST",
                    snippet="New Zealand Time now",
                    relevance_score=0.95
                ),
                WebSearchResult(
                    title="Time and Date",
                    url="https://www.timeanddate.com/worldclock/new-zealand/auckland",
                    snippet="Current Local Time in Auckland, New Zealand (Tāmaki Makaurau)",
                    relevance_score=0.88
                )
            ]
        }
    
    async def search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """Perform dummy web search."""
        start_time = datetime.now()
        
        logger.info(f"Performing dummy web search for: {query}")
        
        # Find matching results or generate generic ones
        query_lower = query.lower()
        results = []
        
        # Try to find exact matches in dummy data
        for key, dummy_results in self.dummy_data.items():
            if key in query_lower:
                results = dummy_results[:max_results]
                break
        

        if not results:
            return []
        
        # Create summary
        summary = self._generate_summary(query, results)
        
        # Calculate search time
        search_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return WebSearchResponse(
            query=query,
            results=results,
            summary=summary,
            total_results=len(results),
            search_time_ms=search_time,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    
    def _generate_summary(self, query: str, results: List[WebSearchResult]) -> str:
        """Generate a summary of search results."""
        if not results:
            return f"No search results found for '{query}'."
        
        summary_parts = [f"Found {len(results)} relevant results for '{query}':\n"]
        
        for i, result in enumerate(results[:3], 1):  # Summarize top 3 results
            summary_parts.append(f"{i}. {result.title}: {result.snippet[:100]}...")
        
        return "\n".join(summary_parts)


class WebSearchManager:
    """Manages web search operations and provider selection."""
    
    def __init__(self, provider: Optional[WebSearchProvider] = None):
        """Initialize web search manager."""
        self.provider = provider or DummyWebSearchProvider()
        logger.info(f"Web search initialized with provider: {self.provider.provider_name}")
    
    async def search_and_format_for_ai(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform web search and format results for AI context injection.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with formatted search results for AI consumption
        """
        try:
            search_response = await self.provider.search(query, max_results)
            
            # Format for AI consumption
            formatted_results = {
                "query": search_response.query,
                "summary": search_response.summary,
                "sources": [
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "relevance": result.relevance_score
                    }
                    for result in search_response.results
                ],
                "metadata": {
                    "total_results": search_response.total_results,
                    "search_time_ms": search_response.search_time_ms,
                    "timestamp": search_response.timestamp,
                    "provider": self.provider.provider_name
                }
            }
            
            logger.info(
                f"Web search completed",
                extra={
                    "query": query,
                    "results_count": len(search_response.results),
                    "search_time_ms": search_response.search_time_ms
                }
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(
                f"Web search failed",
                extra={
                    "query": query,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            # Return error response
            return {
                "query": query,
                "summary": f"Web search failed: {str(e)}",
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
    
    async def search_and_scrape_for_ai(self, query: str, max_results: int = 5, max_scrape_urls: int = 2) -> Dict[str, Any]:
        """
        Perform web search AND scrape content from top URLs for comprehensive AI context.
        
        Args:
            query: Search query
            max_results: Maximum number of search results to return
            max_scrape_urls: Maximum number of URLs to scrape content from
            
        Returns:
            Dictionary with search results + scraped content for AI consumption
        """
        try:
            start_time = datetime.now()
            
            # Step 1: Perform web search
            search_response = await self.provider.search(query, max_results)
            
            # Step 2: Extract top URLs for scraping
            top_urls = [result.url for result in search_response.results[:max_scrape_urls]]
            
            # Step 3: Scrape content from top URLs
            scraper_manager = get_web_scraper_manager()
            scraped_data = await scraper_manager.scrape_urls_for_ai(top_urls)
            
            # Step 4: Create comprehensive AI-ready response
            combined_summary = self._create_combined_summary(search_response, scraped_data)
            
            enhanced_results = {
                "query": search_response.query,
                "search_summary": search_response.summary,
                "scraped_summary": scraped_data.get("combined_content", "No scraped content available."),
                "combined_summary": combined_summary,
                "sources": [
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "relevance": result.relevance_score,
                        "scraped": result.url in [content["url"] for content in scraped_data.get("contents", [])]
                    }
                    for result in search_response.results
                ],
                "scraped_contents": scraped_data.get("contents", []),
                "metadata": {
                    "search_results": search_response.total_results,
                    "scraped_urls": scraped_data.get("scraped_urls", 0),
                    "total_time_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                    "search_time_ms": search_response.search_time_ms,
                    "scrape_time_ms": scraped_data.get("metadata", {}).get("total_words", 0),  # Placeholder
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "search_provider": self.provider.provider_name,
                    "scraper": scraped_data.get("metadata", {}).get("scraper", "unknown")
                }
            }
            
            logger.info(
                f"Web search and scraping completed",
                extra={
                    "query": query,
                    "search_results": len(search_response.results),
                    "scraped_urls": scraped_data.get("scraped_urls", 0),
                    "total_time_ms": enhanced_results["metadata"]["total_time_ms"]
                }
            )
            
            return enhanced_results
            
        except Exception as e:
            logger.error(
                f"Web search and scraping failed",
                extra={
                    "query": query,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            # Return error response
            return {
                "query": query,
                "search_summary": f"Web search failed: {str(e)}",
                "scraped_summary": "Scraping failed due to search error.",
                "combined_summary": f"Search and scraping failed: {str(e)}",
                "sources": [],
                "scraped_contents": [],
                "metadata": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
    
    def _create_combined_summary(self, search_response: WebSearchResponse, scraped_data: Dict[str, Any]) -> str:
        """Create a combined summary from search results and scraped content."""
        combined_parts = []
        
        # Add search summary
        combined_parts.append("Search Results:")
        combined_parts.append(search_response.summary)
        
        # Add scraped content summary if available
        if scraped_data.get("scraped_urls", 0) > 0:
            combined_parts.append("\n\nScraped Content:")
            combined_parts.append(scraped_data.get("combined_content", "No detailed content available."))
        
        return "\n".join(combined_parts)


# Global web search manager instance
_web_search_manager: Optional[WebSearchManager] = None


def get_web_search_manager() -> WebSearchManager:
    """Get or create the global web search manager."""
    global _web_search_manager
    if _web_search_manager is None:
        _web_search_manager = WebSearchManager()
    return _web_search_manager


async def search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Convenience function to perform web search.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        Formatted search results for AI consumption
    """
    manager = get_web_search_manager()
    return await manager.search_and_format_for_ai(query, max_results)


async def search_and_scrape_web(query: str, max_results: int = 5, max_scrape_urls: int = 2) -> Dict[str, Any]:
    """
    Convenience function to perform web search AND scrape content from top URLs.
    
    Args:
        query: Search query
        max_results: Maximum number of search results to return
        max_scrape_urls: Maximum number of URLs to scrape content from
        
    Returns:
        Formatted search results + scraped content for AI consumption
    """
    manager = get_web_search_manager()
    return await manager.search_and_scrape_for_ai(query, max_results, max_scrape_urls)


async def enhance_request_with_web_context(request_body: bytes, query: str = None) -> bytes:
    """
    Enhance AI request with web search context by injecting system message.
    
    This is the main public function that should be called from base.py
    to add web search context to AI requests.
    
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
        web_data = await search_and_scrape_web(query)
        
        # Create web context message
        web_context = web_data.get("combined_summary", "No web data available")
        web_message = {
            "role": "system",
            "content": f"Web search context: {web_context}"
        }
        
        # Insert system message at the beginning
        if "messages" in data:
            data["messages"].insert(0, web_message)
        
        # Convert back to bytes
        enhanced_body = json.dumps(data).encode()
        
        logger.info(
            f"Enhanced request with web context",
            extra={
                "query": query,
                "original_size": len(request_body),
                "enhanced_size": len(enhanced_body),
                "web_sources": web_data.get("metadata", {}).get("scraped_urls", 0)
            }
        )
        
        return enhanced_body
        
    except Exception as e:
        logger.error(
            f"Failed to enhance request with web context",
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