"""
Web scraping module for extracting content from URLs.

This module provides web scraping functionality to extract clean text content
from web pages found by the search module. It includes dummy implementations
for testing and development.
"""

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from .types import SearchResult, WebPageContent
import httpx

from ..core.logging import get_logger



logger = get_logger(__name__)


class ScrapeFailureError(Exception):
    """Custom exception for controlled scraping failures."""
    pass


class BaseWebScraper:
    """Base class for web scrapers."""

    scraper_name: str = "base"

    def __init__(self, output_dir: str = "scraped_html"):
        self.output_dir = output_dir
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    async def scrape_url(self, url: str) -> Optional[str]:
        """Scrape content from a single URL."""
        raise NotImplementedError("Subclasses must implement scrape_url method")


    async def scrape_webpages(
        self, webpages: List[WebPageContent], max_concurrent: int = 3
    ) -> List[WebPageContent]:
        """Scrape multiple webpages concurrently."""
        raise NotImplementedError("Subclasses must implement scrape_webpages method")


    async def scrape_search_results(self, search_result: SearchResult) -> SearchResult:
        """
        Scrape content from URLs in a SearchResult object.
        
        Args:
            search_result: SearchResult object with URLs to scrape
            
        Returns:
            SearchResult with scraped content populated
        """
        if not search_result.results:
            logger.warning("No results to scrape")
            return search_result
            
        pages_to_scrape = search_result.results
        num_pages_to_scrape = len(pages_to_scrape)
        logger.info(f"Scraping {num_pages_to_scrape} URLs from search results")
        
        
        max_concurrent_scrapes = 10  # Default, could be configurable
        start_time = datetime.now()
        scraped_webpages = await self.scrape_webpages(
            pages_to_scrape, max_concurrent=max_concurrent_scrapes
        )
        scrape_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        search_result.results = scraped_webpages
        num_successful_scrapes = len([c for c in scraped_webpages if c.content])

        logger.info(
            f"Scraped {num_successful_scrapes}/{num_pages_to_scrape} successfully in {scrape_time_ms}ms",
            # Pass the list of failures as structured data
            extra={
                "scraping_summary": {
                    "successful_count": num_successful_scrapes,
                    "failed_count": num_pages_to_scrape - num_successful_scrapes,
                    "total": num_pages_to_scrape,
                    "milliseconds": scrape_time_ms,
                }
            },
        )
        


    def _sanitize_filename(self, url: str) -> str:
        """Create a safe filename from a URL."""
        # Remove protocol
        filename = url.replace("https://", "").replace("http://", "")
        # Replace path separators and other unsafe characters
        filename = re.sub(r'[\\/:*?"<>|]', "_", filename)
        # Limit length and add .html extension
        return f"{filename[:250]}.txt"

    async def _write_to_file(self, filename: str, content: str) -> None:
        """Asynchronously write raw HTML content to a file."""
        try:
            filename = self._sanitize_filename(filename)
            filepath = os.path.join(self.output_dir, filename)

            def write_file() -> None:
                with open(filepath, "w") as f:
                    f.write(filename)
                    f.write("\n")
                    f.write(content)

            await asyncio.to_thread(write_file)

        except Exception as e:
            logger.error(f"Failed to write HTML for {filename} to file: {e}")