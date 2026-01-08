"""
Serper API Searcher Module

This module provides a WebSearchProvider implementation that uses the Serper API
to get search results and formats them into the standard WebSearchResponse.
"""

from datetime import datetime, timezone
from typing import Any, Dict

from ..core.logging import get_logger
from .BaseWebSearch import BaseWebSearch
from .types import SearchResult, WebPageContent

logger = get_logger(__name__)


class SerperWebSearch(BaseWebSearch):
    """
    A web search provider that uses the Serper API to get search results.
    """

    provider_name = "Serper"

    def __init__(self, api_key: str):
        """
        Initialize the Serper provider.

        Args:
            api_key: The Serper API key.
        """

        super().__init__()

        self.base_url = "https://google.serper.dev"
        # 2. Now, do the Serper-specific initialization.
        if not api_key:
            raise ValueError("Serper API key cannot be empty.")
        self.api_key = api_key

        # The logger from the parent will have already run, so you can add your own.
        logger.info("SerperWebSearch initialized with API key.")

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Perform web search using the Serper API and return a WebSearchResponse.
        """
        start_time = datetime.now()
        logger.info(f"Performing Serper API search for: '{query}'")

  
        try:
            # --- MOCK DATA FOR TESTING ---
            api_response = await self._load_mock_data(
                "serper_trump-peace-plan.json"
                # serper_what_is_the_state_of_the_US_jobmarket_currently_Which_websites_did_you_search_be_brief_20251223_150343.json
            )
            # ---------------------------------------------------------------
            # api_response = await self._call_serper_api(query, max_results)
            # await self._save_api_response(api_response, query, "serper")
            # ---------------------------------------------------------------

            # Parse the results from the API response
            serper_result = api_response.get("organic", [])
            parsed_results = []
            for i, item in enumerate(serper_result):
                result = WebPageContent(
                    title=item.get("title", "No Title"),
                    url=item.get("link", "Unknown URL"),
                    summary=item.get("snippet", None),
                    publication_date=item.get("date", None),
                    relevance_score=1.0
                    - (i * 0.1),  # Simple relevance based on position
                    content=None,
                    relevant_chunks=None,
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

            return SearchResult(
                query=query,
                results=parsed_results,
                summary=None,
                total_results=len(parsed_results),
                search_time_ms=search_time,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except FileNotFoundError:
            error_msg = "Dummy data file not found."
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = (
                f"Failed to get or process Serper API response for query '{query}': {e}"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

    async def _call_serper_api(
        self, query: str, max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Make a live API call to Serper search service.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            Dictionary containing the API response

        Raises:
            Exception: If API call fails or returns non-200 status
        """

        logger.debug(f"Making Serper API call for: '{query}'")
        # Prepare Serper API request

        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        payload = {"q": query, "num": max_results}

        return await self.make_request(
            method="POST",
            endpoint="/search",
            headers=headers,
            payload=payload,
        )
