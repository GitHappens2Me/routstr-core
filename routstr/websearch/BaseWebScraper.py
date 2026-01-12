"""
Web scraping module for extracting content from URLs.

This module provides web scraping functionality to extract clean text content
from web pages found by the search module. It includes dummy implementations
for testing and development.
"""

import asyncio
import os
import re
from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import datetime
from typing import List, Optional

import httpx

from ..core.logging import get_logger
from ..core.settings import settings
from .types import SearchResult, WebPageContent

logger = get_logger(__name__)


class ScrapeFailureError(Exception):
    """Custom exception for controlled scraping failures."""

    pass


class BaseWebScraper(ABC):
    """Base class for web scrapers."""

    scraper_name: str = "base"

    def __init__(self, output_dir: str = "scraped_html"):
        self.output_dir = output_dir
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        self.client_timeout: httpx.Timeout = httpx.Timeout(3.0, connect=3.0)
        self.client_headers: dict = {
            "Accept": "text/html, text/plain",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        }
        self.client_redirects: bool = True

    @abstractmethod
    async def scrape_url(self, url: str, client: httpx.AsyncClient) -> Optional[str]:
        """Scrape content from a single URL."""

    @abstractmethod
    async def scrape_webpages(
        self, webpages: List[WebPageContent], max_concurrent: int = 10
    ) -> List[WebPageContent]:
        """Scrape multiple webpages concurrently."""

    async def scrape_search_results(self, search_result: SearchResult) -> SearchResult:
        """
        Scrape content from URLs in a SearchResult object and return a new SearchResult.

        Args:
            search_result: SearchResult object with URLs to scrape

        Returns:
            A new SearchResult object with scraped content populated
        """
        if not search_result.results:
            # TODO: better logging
            
            logger.warning("No results to scrape")
            return search_result

        pages_to_scrape = search_result.results
        num_pages_to_scrape = len(pages_to_scrape)
        logger.info(f"Scraping {num_pages_to_scrape} URLs from search results")

        max_concurrent_scrapes = settings.web_scrape_max_concurrent_urls
        start_time = datetime.now()
        scraped_webpages = await self.scrape_webpages(
            pages_to_scrape, max_concurrent=max_concurrent_scrapes
        )
        scrape_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        num_successful_scrapes = len([c for c in scraped_webpages if c.content])

        logger.info(
            f"Scraped {num_successful_scrapes}/{num_pages_to_scrape} successfully in {scrape_time_ms}ms",
            extra={
                "scraping_summary": {
                    "successful_count": num_successful_scrapes,
                    "failed_count": num_pages_to_scrape - num_successful_scrapes,
                    "total": num_pages_to_scrape,
                    "milliseconds": scrape_time_ms,
                }
            },
        )

        return replace(search_result, results=scraped_webpages)

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
