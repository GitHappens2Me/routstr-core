import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
from .types import SearchResult, WebPageContent
from .BaseWebScraper import BaseWebScraper, ScrapeFailureError
import httpx

from ..core.logging import get_logger

logger = get_logger(__name__)

# Optional imports for web scraping
NEWSPAPER_AVAILABLE = False
TRAFILATURA_AVAILABLE = False
GOOSE_AVAILABLE = False

try:
    from newspaper import Article

    NEWSPAPER_AVAILABLE = True
    import logging

    newspaper_logger = logging.getLogger('newspaper')
    newspaper_logger.setLevel(logging.WARNING)
except ImportError as e :
    print(f"newspaper not imported: {e}")
    Article = None

try:
    import trafilatura
    trafilatura_logger = logging.getLogger('trafilatura')
    trafilatura_logger.setLevel(logging.WARNING)
    TRAFILATURA_AVAILABLE = True
except ImportError:
    print("trafilatura not imported")
    trafilatura = None

try:
    from goose3 import Goose
    goose3_logger = logging.getLogger('goose3')
    goose3_logger.setLevel(logging.WARNING)
    GOOSE_AVAILABLE = True
except ImportError as e:
    print(f"goose not imported: {e}")
    Goose = None

class DefaultWebScraper(BaseWebScraper):
    """Dummy web scraper for testing and development."""

    scraper_name = "default"

    def __init__(self):
        super().__init__() 

        self._client_config = {
            "timeout": httpx.Timeout(3.0, connect=3.0),
            "headers": {
                "Accept": "text/html, text/plain",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            },
            "follow_redirects": True,
        }
        self.client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        """Ensures the httpx client is created and open."""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(**self._client_config)
        return self.client


    async def scrape_url(self, url: str) -> Optional[str]:
        """Scrape content from a single URL."""
        start_time = datetime.now()
        client = await self.get_client()
        try:
            # 1. Reject URLs that are too long
            if len(url) >= 200:
                raise ScrapeFailureError(f"Rejected: URL too long ({len(url)} chars)")

            # 2. Make the HTTP GET request
            response = await client.get(url)
            # print(response)
            # 3. Validate MIME type from headers
            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith(("text/html", "text/plain")):
                raise ScrapeFailureError(f"Rejected non-text content: '{content_type}'")

            # 4. Reject oversized responses based on Content-Length header
            content_length = int(response.headers.get("content-length", 0))
            if content_length > 5_000_000:  # 5MB
                raise ScrapeFailureError("Rejected oversized response")

            # 5. Read content and reject binary content
            content_bytes = await response.aread()
            if b"\x00" in content_bytes:
                raise ScrapeFailureError("Rejected binary content (null byte found)")

            # 6. Decode content and clean it
            html = content_bytes.decode("utf-8", errors="replace")
            await self._write_to_file(f"{url}", html)

            content = await self.extract_content(html, url)
            return content

        except (
            httpx.TimeoutException,
            httpx.RequestError,
            ScrapeFailureError,
            Exception,
        ) as e:
            # This single block handles ALL failure paths.
            error_message = str(e)

            # Log the error as before
            if isinstance(e, httpx.TimeoutException):
                print(url)
                logger.error(f"Timeout exceeded for {url}: {e}, {error_message}")
            elif isinstance(e, httpx.RequestError):
                logger.error(f"Request failed for {url}: {error_message}")
            elif isinstance(e, ScrapeFailureError):
                logger.warning(f"Scrape rejected for {url}: {error_message}")
            else:
                logger.error(f"An unexpected error occurred for {url}: {error_message}")

            # 10. Return the error ScrapedContent object
            return None


    async def scrape_webpages(
        self, webpages: List[WebPageContent], max_concurrent: int = 10
    ) -> List[WebPageContent]:
        """
        Scrapes multiple URLs concurrently using asyncio.gather and a semaphore.
        """
        #TODO:  define this globally in init?
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_single_page(webpage: WebPageContent) -> WebPageContent:
            """Wraps the scraping of a single page with the semaphore."""
            async with semaphore:
                content = await self.scrape_url(webpage.url)
                webpage.content = content
                return webpage

        # Create a list of tasks to run concurrently
        tasks = [scrape_single_page(page) for page in webpages]
        
        # Wait for all tasks to complete, returning exceptions in the results list
        scraped_webpages = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = [
            page for page in scraped_webpages if page.content is not None
        ]
    
        return successful_results


    async def extract_content(self, raw_html, url): 
        start_time = datetime.now()

        # Define extractors in order of preference.
        # Each entry points to one of your dedicated extraction methods.
        extractors = []
        if TRAFILATURA_AVAILABLE:
            extractors.append(("Trafilatura", self.extract_with_trafilatura))
        if GOOSE_AVAILABLE:
            extractors.append(("Goose3", self.extract_with_goose3))
        if NEWSPAPER_AVAILABLE:
            extractors.append(("Newspaper", self.extract_with_newspaper))


        try:
            for name, extractor_func in extractors:
                try:
                    # Call the specific extraction function
                    result = extractor_func(raw_html, url)

                    # Check if the function returned a valid result (no 'error' key)
                    if result and "error" not in result and result.get("content"):
                        content = result["content"].strip()
                        title = result.get("title", "No title found")
                        
                        #logger.info(f"Successfully extracted content with {name} for {url}. Title: '{title}'")
                        
                        await self._write_to_file(f"{name.lower()}_{url}", content)
                        return content

                except Exception as e:
                    # This catches unexpected errors from the orchestrator itself,
                    # though the individual functions should handle their own.
                    logger.warning(f"Unexpected error during {name} extraction for {url}: {e}")
                    continue

            # If all extraction methods failed, return raw HTML as a fallback
            logger.warning(f"All extraction methods failed for {url}. Returning raw HTML as fallback.")
            return raw_html

        except (httpx.TimeoutException, httpx.RequestError, ScrapeFailureError, Exception) as e:
            # Centralized error handling for network or critical failures
            error_type = type(e).__name__
            logger.error(f"Scraping failed for {url} ({error_type}): {e}")
            return None

        finally:
            # Log timing for both success and failure
            scrape_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.debug(f"Scraping attempt for {url[:20]}.. completed in {scrape_time_ms}ms")



    def extract_with_goose3(self, html_content: str, url: str) -> dict:
        """
        Extracts title and content from HTML using the goose3 library.

        Args:
            html_content: The raw HTML string.
            url: The source URL of the content.

        Returns:
            A dictionary with 'title' and 'content' keys, or with 'error' on failure.
        """
        if not GOOSE_AVAILABLE:
            return {"error": "goose3 not available"}

        try:
            g = Goose()
            article = g.extract(raw_html=html_content, url=url)

            # goose3 can return empty strings, so we handle that
            if not article.cleaned_text:
                return {"error": "goose3 could not extract content."}

            return {"title": article.title, "content": article.cleaned_text}
        except Exception as e:
            return {"error": f"goose3 failed: {str(e)}"}

    def extract_with_newspaper(self, html_content: str, url: str) -> dict:
        """
        Extracts title and content from HTML using the newspaper3k library.
        This version is corrected to work with pre-downloaded HTML.
        """
        if not NEWSPAPER_AVAILABLE:
            return {"error": "newspaper not available"}

        try:
            from newspaper.configuration import Configuration

            # Create a config object to prevent newspaper from making its own network request
            config = Configuration()
            config.browser_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

            article = Article(url, config=config)
            # Use .set_html() to provide the content you already downloaded
            article.set_html(html_content)
            article.parse()

            if not article.text:
                return {"error": "newspaper could not extract content."}

            return {"title": article.title, "content": article.text}
        except Exception as e:
            return {"error": f"newspaper failed: {str(e)}"}

    def extract_with_trafilatura(self, html_content: str, url: str) -> dict:
        """
        Extracts title and content from HTML using the trafilatura library.
        This version is corrected to work with pre-downloaded HTML.
        """
        if not TRAFILATURA_AVAILABLE:
            return {"error": "trafilatura not available"}

        try:
            # Pass the HTML string directly to the extract function
            result = trafilatura.extract(
                html_content, include_comments=False, include_tables=False
            )

            if not result:
                return {"error": "trafilatura could not extract content."}

            # Pass the HTML string directly to the metadata function
            metadata = trafilatura.metadata.extract_metadata(html_content)

            return {
                "title": metadata.title
                if metadata and metadata.title
                else "No title found",
                "content": result,
            }
        except Exception as e:
            return {"error": f"trafilatura failed: {str(e)}"}

    async def close(self) -> None:
        """Close the httpx client."""
        await self.client.aclose()
