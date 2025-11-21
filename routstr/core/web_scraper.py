"""
Web scraping module for extracting content from URLs.

This module provides web scraping functionality to extract clean text content
from web pages found by the search module. It includes dummy implementations
for testing and development.
"""

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScrapedContent:
    """Represents scraped content from a URL."""
    url: str
    title: str
    content: str
    content_type: str  # "article", "documentation", "news", etc.
    word_count: int
    scrape_time_ms: int
    timestamp: str


class WebScraper:
    """Base class for web scrapers."""
    
    scraper_name: str = "base"
    
    async def scrape_url(self, url: str) -> ScrapedContent:
        """Scrape content from a single URL."""
        raise NotImplementedError("Subclasses must implement scrape_url method")
    
    async def scrape_multiple(self, urls: List[str], max_concurrent: int = 3) -> List[ScrapedContent]:
        """Scrape multiple URLs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url: str) -> ScrapedContent:
            async with semaphore:
                return await self.scrape_url(url)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return only successful scrapes
        successful_results = []
        for result in results:
            if isinstance(result, ScrapedContent):
                successful_results.append(result)
            else:
                logger.error(f"Scraping failed: {result}")
        
        return successful_results
    
    def clean_content(self, raw_html: str) -> str:
        """Extract clean text from raw HTML."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', raw_html)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()


class DummyWebScraper(WebScraper):
    """Dummy web scraper for testing and development."""
    
    scraper_name = "dummy"
    
    def __init__(self):
        """Initialize dummy scraper with sample content."""
        self.dummy_content = {
            "https://time.is/de/nzst/": {
                "title": "Time is",
                "content": """
                Time.is
                New Zealand Time now
                03:55:55
                Samstag, 22. November 2025, Woche 47
                New Zealand Time als Standard setzen - Zu Favoriten hinzufügen
                """,
                "content_type": "info"
            },
            "https://www.timeanddate.com/worldclock/new-zealand/auckland": {
                "title": "Current Local Time in Auckland",
                "content": """
                Current Local Time in Auckland, New Zealand (Tāmaki Makaurau)

                    Time/General
                    Weather
                    Time Zone
                    DST Changes
                    Sun & Moon 

                03:55:55 NZDT

                Samstag, 22. November 2025
                Fullscreen
                Country: 	New Zealand
                Region: 	Auckland (AUK)
                Lat/Long: 	36°51'S / 174°46'E
                Elevation: 	48 m
                Currency: 	NZ Dollar (NZD)
                Languages: 	English, Māori, New Zealand Sign Language
                Country Code: 	+64
                Location of AucklandLocation
                °C
                Weather
                15 °C

                Clear.
                24 / 14 °C
                So 23.	[Sprinkles late. Mostly cloudy.] 	23 / 15 °C
                Mo 24.	[Mostly sunny.] 	23 / 13 °C

                Weather by CustomWeather, © 2025
                """,
                "content_type": "info"
            },
        }
    
    async def scrape_url(self, url: str) -> ScrapedContent:
        """Scrape content from a single URL using dummy data."""
        start_time = datetime.now()
        
        logger.info(f"Scraping dummy content for: {url}")
        
        # Get dummy content or generate generic content
        dummy_data = self.dummy_content.get(url)
        
        if dummy_data:
            title = dummy_data["title"]
            content = dummy_data["content"].strip()
            content_type = dummy_data["content_type"]
        else:
            # Generate generic dummy content for unknown URLs
            title = f"Content from {url}"
            content = "No Content Found"
            content_type = "article"
        
        # Calculate metrics
        word_count = len(content.split())
        scrape_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        scraped_content = ScrapedContent(
            url=url,
            title=title,
            content=content,
            content_type=content_type,
            word_count=word_count,
            scrape_time_ms=scrape_time,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(
            f"Dummy scraping completed",
            extra={
                "url": url,
                "word_count": word_count,
                "scrape_time_ms": scrape_time,
                "content_type": content_type
            }
        )
        
        return scraped_content
   
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL for generic content generation."""
        try:
            # Simple domain extraction
            match = re.search(r'https?://([^/]+)', url)
            if match:
                domain = match.group(1)
                # Remove www. prefix if present
                return domain.replace('www.', '')
            return "unknown domain"
        except Exception:
            return "unknown domain"


class WebScraperManager:
    """Manages web scraping operations."""
    
    def __init__(self, scraper: Optional[WebScraper] = None):
        """Initialize web scraper manager."""
        self.scraper = scraper or DummyWebScraper()
        logger.info(f"Web scraper initialized with: {self.scraper.scraper_name}")
    
    async def scrape_urls_for_ai(self, urls: List[str], max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Scrape multiple URLs and format results for AI context injection.
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum number of concurrent scraping operations
            
        Returns:
            Dictionary with formatted scraped content for AI consumption
        """
        try:
            scraped_contents = await self.scraper.scrape_multiple(urls, max_concurrent)
            
            # Format for AI consumption
            formatted_results = {
                "scraped_urls": len(scraped_contents),
                "contents": [
                    {
                        "url": content.url,
                        "title": content.title,
                        "content": content.content[:1000] + "..." if len(content.content) > 1000 else content.content,
                        "content_type": content.content_type,
                        "word_count": content.word_count,
                        "relevance": self._calculate_relevance(content)
                    }
                    for content in scraped_contents
                ],
                "combined_content": self._create_combined_content(scraped_contents),
                "metadata": {
                    "total_urls": len(urls),
                    "successful_scrapes": len(scraped_contents),
                    "total_words": sum(content.word_count for content in scraped_contents),
                    "scraper": self.scraper.scraper_name,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            
            logger.info(
                f"Web scraping completed",
                extra={
                    "total_urls": len(urls),
                    "successful_scrapes": len(scraped_contents),
                    "total_words": formatted_results["metadata"]["total_words"]
                }
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(
                f"Web scraping failed",
                extra={
                    "urls": urls,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            # Return error response
            return {
                "scraped_urls": 0,
                "contents": [],
                "combined_content": f"Web scraping failed: {str(e)}",
                "metadata": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
    
    def _calculate_relevance(self, content: ScrapedContent) -> float:
        """Calculate relevance score for scraped content."""
        # Simple relevance calculation based on content length and type
        base_score = 0.5
        
        # Boost for longer content (more information)
        if content.word_count > 500:
            base_score += 0.2
        elif content.word_count > 200:
            base_score += 0.1
        
        # Boost for certain content types
        if content.content_type in ["documentation", "encyclopedia"]:
            base_score += 0.2
        elif content.content_type in ["tutorial", "article"]:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _create_combined_content(self, scraped_contents: List[ScrapedContent]) -> str:
        """Create combined content summary from all scraped sources."""
        if not scraped_contents:
            return "No content was successfully scraped."
        
        combined_parts = [f"Scraped content from {len(scraped_contents)} sources:\n"]
        
        for i, content in enumerate(scraped_contents, 1):
            # Truncate very long content for summary
            content_preview = content.content[:500] + "..." if len(content.content) > 500 else content.content
            combined_parts.append(f"{i}. {content.title} ({content.url}):\n{content_preview}\n")
        
        return "\n".join(combined_parts)


# Global web scraper manager instance
_web_scraper_manager: Optional[WebScraperManager] = None


def get_web_scraper_manager() -> WebScraperManager:
    """Get or create the global web scraper manager."""
    global _web_scraper_manager
    if _web_scraper_manager is None:
        _web_scraper_manager = WebScraperManager()
    return _web_scraper_manager


async def scrape_urls(urls: List[str], max_concurrent: int = 3) -> Dict[str, Any]:
    """
    Convenience function to scrape multiple URLs.
    
    Args:
        urls: List of URLs to scrape
        max_concurrent: Maximum number of concurrent scraping operations
        
    Returns:
        Formatted scraped content for AI consumption
    """
    manager = get_web_scraper_manager()
    return await manager.scrape_urls_for_ai(urls, max_concurrent)