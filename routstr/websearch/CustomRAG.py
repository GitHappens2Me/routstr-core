"""
Custom RAG Provider Module

This module provides a manual RAG implementation that combines separate components:
web search, content scraping, and text chunking into a unified RAG pipeline.
Offers flexibility to use different providers for each pipeline stage.
"""

from dataclasses import replace
from datetime import datetime, timezone

from ..core.logging import get_logger
from ..core.settings import settings
from .BaseWebChunker import BaseWebChunker
from .BaseWebRAG import BaseWebRAG
from .BaseWebScraper import BaseWebScraper
from .BaseWebSearch import BaseWebSearch
from .types import SearchResult

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
        scrape_provider: BaseWebScraper,
        chunk_provider: BaseWebChunker,
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
        if not scrape_provider:
            raise ValueError("Scraper provider cannot be None")
        if not chunk_provider:
            raise ValueError("Chunker provider cannot be None")

        self.search_provider = search_provider
        self.scraper_provider = scrape_provider
        self.chunker_provider = chunk_provider

        logger.info(
            f"CustomRAG initialized with: {search_provider.__class__.__name__}, "
            f"{scrape_provider.__class__.__name__}, {chunk_provider.__class__.__name__}"
        )

    async def retrieve_context(self, query: str, max_results: int = 10) -> SearchResult:
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
            search_response = await self.search_provider.search(query, max_results)

            if not search_response.results:
                logger.warning(f"No search results found for query: '{query}'")
                return SearchResult(
                    query=query,
                    results=[],
                    summary=None,
                    search_time_ms=int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            #TODO: rename to scrapes_response or something more descripitve
            search_response = await self.scraper_provider.scrape_search_results(search_response)

            if settings.enable_chunking and search_response.results:
                search_response = await self.chunker_provider.chunk_search_results(search_response, query)

            # Calculate total pipeline time
            search_time = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.info(
                f"CustomRAG pipeline completed: {len(search_response.results)} results in {search_time}ms"
            )

            # Update timing metadata and return a new object
            return replace(
                search_response,
                search_time_ms=search_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        except Exception as e:
            error_msg = f"CustomRAG pipeline failed for query '{query}': {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

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
            #TODO: BaseWebSearch should enforce check_availability -> Does serper have a usage endpoint?
            if hasattr(self.search_provider, "check_availability"):
                search_available = await self.search_provider.check_availability()
            else:
                # Fallback: assume available if no check method
                search_available = True

            # Check scraper availability
            #TODO: BaseWebScraper should also have a check_availability function ?
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
                logger.debug("CustomRAG all components available")

            return all_available

        except Exception as e:
            logger.error(f"CustomRAG availability check failed: {e}")
            return False
