"""
Web scraping module for extracting content from URLs.

This module provides web scraping functionality to extract clean text content
from web pages found by the search module. It includes dummy implementations
for testing and development.
"""

import os
import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import httpx

# Optional imports for web scraping
NEWSPAPER_AVAILABLE = False
TRAFILATURA_AVAILABLE = False
GOOSE_AVAILABLE = False

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    print("newspaper not imported")
    Article = None

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    print("trafilatura not imported")
    trafilatura = None

try:
    from goose3 import Goose
    GOOSE_AVAILABLE = True
except ImportError:
    print("goose not imported")
    Goose = None

from ..core.logging import get_logger
logger = get_logger(__name__)



@dataclass
class ScrapedContent:
    """Represents scraped content from a URL."""
    url: str
    title: str
    content: str
    content_type: str
    word_count: int
    scrape_time_ms: int
    timestamp: str
    error: Optional[str] = None

class ScrapeFailureError(Exception):
    """Custom exception for controlled scraping failures."""
    pass

class BaseWebScraper:
    """Base class for web scrapers."""
    
    scraper_name: str = "base"
    
    
    async def scrape_url(self, url: str) -> ScrapedContent:
        """Scrape content from a single URL."""
        raise NotImplementedError("Subclasses must implement scrape_url method")
    
    async def scrape_urls(self, urls: List[str], max_concurrent: int = 3) -> List[ScrapedContent]:
        """Scrape multiple URLs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url: str) -> ScrapedContent:
            async with semaphore:
                return await self.scrape_url(url)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results: List[ScrapedContent] = []
        failed_results: List[ScrapedContent] = []
        
        for result in results:
            if result.error:
                failed_results.append(result)
            else:
                successful_results.append(result)

        num_successful = len(successful_results)
        num_failed = len(failed_results)
        
        logger.info(
            f"Scraping summary: {num_successful} URLs successfully scraped, {num_failed} URLs failed.",
            # Pass the list of failures as structured data
            extra={
                "scraping_summary": {
                    "successful_count": num_successful,
                    "failed_count": num_failed,
                    "failures": failed_results
                }
            }
        )
        return successful_results


class GenericWebScraper(BaseWebScraper):
    """Dummy web scraper for testing and development."""
    
    scraper_name = "Generic"
    
    def __init__(self, output_dir: str = "scraped_html"):
        # httpx.AsyncClient is more efficient for async operations
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0), # 10s total, 5s connect
            headers={"Accept": "text/html, text/plain",
                     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                    },
            follow_redirects=True
        )
        self.output_dir = output_dir
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def _sanitize_filename(self, url: str) -> str:
        """Create a safe filename from a URL."""
        # Remove protocol
        filename = url.replace("https://", "").replace("http://", "")
        # Replace path separators and other unsafe characters
        filename = re.sub(r'[\\/:*?"<>|]', '_', filename)
        # Limit length and add .html extension
        return f"{filename[:250]}.txt"

    async def _write_to_file(self, filename: str, content: str):
        """Asynchronously write raw HTML content to a file."""
        try:
            filename = self._sanitize_filename(filename)
            filepath = os.path.join(self.output_dir, filename)
            
            def write_file():
                with open(filepath, 'w') as f:
                    f.write(filename)
                    f.write("\n")
                    f.write(content)
            
            await asyncio.to_thread(write_file)

        except Exception as e:
            logger.error(f"Failed to write HTML for {filename} to file: {e}")
    
    async def scrape_url(self, url: str) -> Optional[ScrapedContent]:
        """Scrape content from a single URL."""
        start_time = datetime.now()

        try:
            # 1. Reject URLs that are too long
            if len(url) >= 200:
                raise ScrapeFailureError(f"Rejected: URL too long ({len(url)} chars)")

            # 2. Make the HTTP GET request
            response = await self.client.get(url)
            #print(response)
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
            if b'\x00' in content_bytes:
                raise ScrapeFailureError("Rejected binary content (null byte found)")



            # 6. Decode content and clean it
            content = content_bytes.decode("utf-8", errors="replace")
            await self._write_to_file(f"{url}", content)

            # Try different extraction methods based on availability
            goose_result = {"error": "goose3 not available"}
            newspaper_result = {"error": "newspaper not available"}
            trafilatura_result = {"error": "trafilatura not available"}

            #print(GOOSE_AVAILABLE, NEWSPAPER_AVAILABLE, TRAFILATURA_AVAILABLE)
            if GOOSE_AVAILABLE:
                goose_result = self.extract_with_goose3(content, url)
            if NEWSPAPER_AVAILABLE:
                newspaper_result = self.extract_with_newspaper(content, url)
            if TRAFILATURA_AVAILABLE:
                trafilatura_result = self.extract_with_trafilatura(content, url)

            # Log results and write files for successful extractions
            if "content" in goose_result:
                await self._write_to_file(f"goose_{url}", goose_result["content"])
            else:
                logger.debug(f"Goose3 extraction failed or not available for {url}", extra={"error": goose_result.get("error", "Not available")})

            if "content" in newspaper_result:
                await self._write_to_file(f"newspaper_{url}", newspaper_result["content"])
            else:
                logger.debug(f"Newspaper extraction failed or not available for {url}", extra={"error": newspaper_result.get("error", "Not available")})

            if "content" in trafilatura_result:
                await self._write_to_file(f"trafilatura_{url}", trafilatura_result["content"])
                # Use trafilatura as primary if available
                content = trafilatura_result["content"]
                title = trafilatura_result["title"]
            elif "content" in newspaper_result:
                # Fallback to newspaper
                content = newspaper_result["content"]
                title = newspaper_result["title"]
                logger.info(f"Using newspaper extraction as fallback for {url}")
            elif "content" in goose_result:
                # Fallback to goose3
                content = goose_result["content"]
                title = goose_result["title"]
                logger.info(f"Using goose3 extraction as fallback for {url}")
            else:
                # No extraction method worked
                raise ScrapeFailureError("All content extraction methods failed")


            await self._write_to_file(f"{title}", content)

            # 8. Calculate metrics
            word_count = len(content.split())
            scrape_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return ScrapedContent(
                url=url,
                title=title,
                content=content,
                content_type=content_type,
                word_count=word_count,
                scrape_time_ms=scrape_time_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=None # Explicitly None on success
            )

        except (httpx.TimeoutException, httpx.RequestError, ScrapeFailureError, Exception) as e:
            # This single block handles ALL failure paths.
            error_message = str(e)
            
            # Log the error as before
            if isinstance(e, httpx.TimeoutException):
                logger.error(f"Timeout exceeded for {url}: {error_message}")
            elif isinstance(e, httpx.RequestError):
                logger.error(f"Request failed for {url}: {error_message}")
            elif isinstance(e, ScrapeFailureError):
                logger.warning(f"Scrape rejected for {url}: {error_message}")
            else:
                logger.error(f"An unexpected error occurred for {url}: {error_message}")
                
            # Calculate metrics for the failed attempt
            scrape_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # 10. Return the error ScrapedContent object
            return ScrapedContent(
                url=url,
                title=f"Error: {type(e).__name__}",
                content=error_message,
                content_type="error",
                word_count=0,
                scrape_time_ms=scrape_time_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=error_message # The error attribute is populated
            )


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

            return {
                "title": article.title,
                "content": article.cleaned_text
            }
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
            config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
            
            article = Article(url, config=config)
            # Use .set_html() to provide the content you already downloaded
            article.set_html(html_content)
            article.parse()
            
            if not article.text:
                return {"error": "newspaper could not extract content."}

            return {
                "title": article.title,
                "content": article.text
            }
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
            result = trafilatura.extract(html_content, include_comments=False, include_tables=False)

            if not result:
                return {"error": "trafilatura could not extract content."}

            # Pass the HTML string directly to the metadata function
            metadata = trafilatura.metadata.extract_metadata(html_content)

            return {
                "title": metadata.title if metadata and metadata.title else "No title found",
                "content": result
            }
        except Exception as e:
            return {"error": f"trafilatura failed: {str(e)}"}


    async def close(self):
        """Close the httpx client."""
        await self.client.aclose()


    
            

        
