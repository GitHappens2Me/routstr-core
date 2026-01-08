"""
Exa API RAG Provider Module

This module provides a complete RAG implementation using the Exa API.
Exa delivers intelligent web search with embeddings-based ranking, content extraction,
and highlight generation optimized for AI context enhancement.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict

from ..core.logging import get_logger
from .BaseWebRAG import BaseWebRAG
from .types import SearchResult, WebPageContent

logger = get_logger(__name__)


class ExaWebRAG(BaseWebRAG):
    """
    All-in-one RAG provider using the Exa API for neural web search and content extraction.
    """

    provider_name = "Exa"

    def __init__(self, api_key: str):
        """
        Initialize the Exa RAG provider.

        Args:
            api_key: The Exa API key for authentication

        Raises:
            ValueError: If API key is empty or None
        """
        super().__init__()

        self.base_url = "https://api.exa.ai"

        if not api_key:
            raise ValueError("Exa API key cannot be empty.")
        self.api_key = api_key
        logger.info("ExaWebRAG initialized.")

    async def retrieve(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Perform RAG retrieval using Exa's neural search API.

        Args:
            query: The search query for retrieving relevant web content
            max_results: Maximum number of web sources to process (max 10 recommended)

        Returns:
            SearchResult with neural-ranked content, extracted highlights, and metadata

        Raises:
            Exception: If API call fails or response parsing fails
        """
        start_time = datetime.now()
        logger.info(f"Performing Exa API search for: '{query}'")

        try:
            # --- MOCK DATA FOR TESTING ---
            api_response = await self._load_mock_data(
                "exa_what_is_the_latest_news_about_the_Donald_Trump_peace_deal_Which_websites_did_you_search_be_brief_20251219_163302.json"
                # exa_what_is_the_state_of_the_US_jobmarket_currently_Which_websites_did_you_search_be_brief_20251223_145745.json
            )
            # ---------------------------------------------------------------
            # api_response = await self._call_exa_api(query, max_results)
            # await self._save_api_response(api_response, query, "exa")
            # ---------------------------------------------------------------

            # Parse the results from Exa response
            exa_results = api_response.get("results", [])
            parsed_results = []

            # print(api_response)
            for i, web_page in enumerate(exa_results):
                # Combine highlights to single string
                if highlights := web_page.get("highlights", None):
                    highlights = " [...] ".join(highlights)

                result = WebPageContent(
                    title=web_page.get("title", "No Title"),
                    url=web_page.get("url", "Unknown URL"),
                    summary=web_page.get("summary", None),
                    publication_date=web_page.get("publishedDate", None),
                    relevance_score=web_page.get(
                        "score", 1.0 - (i * 0.1)
                    ),  # Fallback assumes results in order of relevance
                    content=web_page.get("text", None),
                    relevant_chunks=highlights,
                )
                parsed_results.append(result)

            if not parsed_results:
                logger.warning(f"No results found for query: '{query}'")
                return SearchResult(  # TODO: just return None?
                    query=query,
                    results=[],
                    summary=None,
                    total_results=0,
                    search_time_ms=int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    ),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

            # Calculate search time
            search_time = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.info(
                f"Exa search completed successfully: {len(parsed_results)} results in {search_time}ms"
            )

            return SearchResult(
                query=query,
                results=parsed_results,
                summary=None,
                total_results=len(parsed_results),
                search_time_ms=search_time,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Exa API response for query '{query}': {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = (
                f"Failed to get or process Exa API response for query '{query}': {e}"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

    async def _call_exa_api(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Make live API call to Exa's neural search endpoint.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            Dictionary containing the complete Exa API response

        Raises:
            Exception: If API call fails or returns non-200 status

        """
        logger.debug(f"Making live Exa API call for: '{query}'")

        # Exa request payload configured for neural search with context
        payload = {
            "query": query,
            "type": "neural",
            "numResults": min(
                max_results, 10
            ),  # Exa's max is 100, but a maximum of 10 is requested
            "contents": {
                "text": False,  # Complete page content
                "summary": False,  # Short summary of page content
                "highlights": {  # Most relevant chunks
                    "numSentences": 3,  # Number of sentences per highlight
                    "highlightsPerUrl": 5,  # Chunks per URL
                },
            },
            "context": True,  # Enable context to get combined content for LLMs
            "includeText": None,
            "excludeText": None,
            "startCrawlDate": None,
            "endCrawlDate": None,
            "startPublishedDate": None,
            "endPublishedDate": None,
            "includeDomains": None,
            "excludeDomains": self.EXCLUDE_DOMAINS, #handle self.EXCLUDE_DOMAINS == None
            "category": None,
            "subpageTarget": None,
            "subpages": 0,  # Do not scrape subpages
            "livecrawl": "never",
            "extras": {"links": 0, "imageLinks": 0},
        }

        headers = {"Content-Type": "application/json", "x-api-key": self.api_key}

        # Make the API request
        return await self.make_request(
            method="POST",
            endpoint="/search",
            headers=headers,
            payload=payload,
        )

    async def check_availability(
        self,
    ) -> bool:
        """
        Verify Exa service availability and API key validity.

        Returns:
            True if basic validation passes, False otherwise

        """
        # Exa does not suppor any way of checking availbility without using API-tokens
        # Fallback: Assume availibity
        return True
