"""
Custom RAG Provider Module

This module provides a manual RAG implementation that combines separate components:
web search, content scraping, and text chunking into a unified RAG pipeline.
Offers flexibility to use different providers for each pipeline stage.
"""

from datetime import datetime, timezone
from typing import List

from ..core.logging import get_logger
from ..core.settings import settings
from .BaseWebChunker import BaseWebChunker
from .BaseWebRAG import BaseWebRAG
from .BaseWebScraper import BaseWebScraper
from .BaseWebSearch import BaseWebSearch
from .types import SearchResult, WebPageContent

logger = get_logger(__name__)


class CustomRAG(BaseWebRAG):
    """
    Manual RAG pipeline implementation using separate search, scraping, and chunking components.
    
    Combines any web search provider, web scraper, and chunker to provide a complete
    RAG solution. Offers maximum flexibility for customizing each pipeline stage.
    """

    provider_name = "Custom"

    def __init__(
        self,
        search_provider: BaseWebSearch,
        scraper_provider: BaseWebScraper,
        chunker_provider: BaseWebChunker,
    ):
        """
        Initialize CustomRAG with pipeline components.

        Args:
            search_provider: Web search provider (e.g., SerperWebSearch)
            scraper_provider: Web scraper for content extraction
            chunker_provider: Text chunker for content processing

        Raises:
            ValueError: If any provider is None
        """
        if not search_provider:
            raise ValueError("Search provider cannot be None")
        if not scraper_provider:
            raise ValueError("Scraper provider cannot be None")
        if not chunker_provider:
            raise ValueError("Chunker provider cannot be None")

        self.search_provider = search_provider
        self.scraper_provider = scraper_provider
        self.chunker_provider = chunker_provider

        logger.info(
            f"CustomRAG initialized with: {search_provider.__class__.__name__}, "
            f"{scraper_provider.__class__.__name__}, {chunker_provider.__class__.__name__}"
        )

    async def retrieve(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Execute complete manual RAG pipeline.

        Performs the full pipeline:
        1. Web search using configured search provider
        2. Content scraping using configured scraper
        3. Text chunking using configured chunker
        4. Returns unified SearchResult with processed content

        Args:
            query: The search query for retrieving relevant web content
            max_results: Maximum number of web sources to process

        Returns:
            SearchResult with processed content, chunks, and metadata

        Raises:
            Exception: If any pipeline stage fails
        """
        start_time = datetime.now()
        logger.info(f"Starting CustomRAG pipeline for query: '{query}'")

        try:
            # Step 1: Perform web search
            logger.debug("Step 1: Performing web search")
            search_response = await self.search_provider.search(query, max_results)

            if not search_response.results:
                logger.warning(f"No search results found for query: '{query}'")
                return SearchResult(
                    query=query,
                    results=[],
                    summary=None,
                    total_results=0,
                    search_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

            # Step 2: Scrape content from URLs
            logger.debug(f"Step 2: Scraping content from {len(search_response.results)} URLs")
            await self.scraper_provider.scrape_search_results(search_response)

            # Step 3: Chunk the scraped content
            if settings.enable_chunking and search_response.results:
                logger.debug("Step 3: Chunking scraped content")
                await self._chunk_search_results(search_response, query)

            # Calculate total pipeline time
            search_time = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.info(
                f"CustomRAG pipeline completed: {len(search_response.results)} results in {search_time}ms"
            )

            # Update timing metadata
            search_response.search_time_ms = search_time
            search_response.timestamp = datetime.now(timezone.utc).isoformat()

            return search_response

        except Exception as e:
            error_msg = f"CustomRAG pipeline failed for query '{query}': {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def _scrape_search_results(self, search_response: SearchResult) -> None:
        """
        Scrape content from URLs in search results.

        Args:
            search_response: SearchResult with URLs to scrape
        """
        urls_to_scrape = [result.url for result in search_response.results]
        logger.info(f"Scraping {len(urls_to_scrape)} URLs")

        max_concurrent_scrapes = settings.web_scrape_max_concurrent_urls
        scraped_contents = await self.scraper_provider.scrape_urls(
            urls_to_scrape, max_concurrent=max_concurrent_scrapes
        )

        # Map scraped content back to results
        content_map = {scraped.url: scraped.content for scraped in scraped_contents}
        for result in search_response.results:
            result.content = content_map.get(result.url)

        # Log scraping summary
        successful_scrapes = len([c for c in scraped_contents if c.content])
        logger.info(f"Successfully scraped {successful_scrapes}/{len(urls_to_scrape)} URLs")

    async def _chunk_search_results(self, search_result: SearchResult, query: str) -> None:
        """
        Chunk the content in search results using configured chunker.

        Args:
            search_result: SearchResult with content to chunk
            query: Original query for relevance ranking
        """
        logger.info(f"Chunking content for {len(search_result.results)} search results")

        for result in search_result.results:
            if result.content:
                chunks = await self.chunker_provider.chunk_text(result.content)
                # Rank chunks by relevance and limit chunks per source
                ranked_chunks = self.chunker_provider.rank_chunks(chunks, query)
                ranked_chunks = ranked_chunks[: settings.chunk_max_chunks_per_source]
                result.relevant_chunks = " [...] ".join(ranked_chunks)
                logger.debug(f"Created {len(ranked_chunks)} chunks for {result.url}")

    async def check_availability(self) -> bool:
        """
        Verify all pipeline components are available and functional.

        Checks availability of:
        - Web search provider
        - Web scraper provider
        - Chunker provider

        Returns:
            True if all components are available, False otherwise
        """
        try:
            # Check search provider availability
            if hasattr(self.search_provider, "check_availability"):
                search_available = await self.search_provider.check_availability()
            else:
                # Fallback: assume available if no check method
                search_available = True

            # Check scraper availability (most scrapers don't have availability checks)
            scraper_available = True

            # Check chunker availability (validate parameters)
            chunker_available = self.chunker_provider.validate_parameters()

            all_available = search_available and scraper_available and chunker_available

            if not all_available:
                logger.warning(
                    f"CustomRAG availability check failed: "
                    f"search={search_available}, scraper={scraper_available}, chunker={chunker_available}"
                )
            else:
                logger.info("CustomRAG all components available")

            return all_available

        except Exception as e:
            logger.error(f"CustomRAG availability check failed: {e}")
            return False