"""
Tavily API Searcher Module

This module provides a WebSearchProvider implementation that uses the Tavily API
to get search results and formats them into the standard WebSearchResponse.
Tavily provides an all-in-one RAG solution with search, content extraction, and ranking.
"""

import http.client
import json
from datetime import datetime, timezone
from typing import Any, Dict

from ..core.logging import get_logger
from .BaseWebSearch import BaseWebSearch, SearchResult, WebPageContent

logger = get_logger(__name__)


class TavilyWebSearch(BaseWebSearch):
    """
    A web search provider that uses the Tavily API to get search results.
    """

    provider_name = "Tavily"

    def __init__(self, api_key: str):
        """
        Initialize the Tavily provider.

        Args:
            api_key: The Tavily API key.
        """

        if not api_key:
            raise ValueError("Tavily API key cannot be empty.")
        self.api_key = api_key

        logger.info("TavilyWebSearch initialized.")

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Perform web search using the Tavily API and return a SearchResult instance.
        """
        start_time = datetime.now()
        logger.info(f"Performing Tavily API search for: '{query}'")

        try:
            # --- MOCK DATA FOR TESTING  ---
            api_response = await self._load_mock_data(
                "tavily_what_is_the_latest_news_about_the_Donald_Trump_peace_deal_Which_websites_did_you_search_be_brief.json"
            )
            # ---------------------------------------------------------------
            # api_response = await self._call_tavily_api(query, max_results)
            # await self._save_api_response(api_response, query, "tavily")
            # ---------------------------------------------------------------

            # Parse the results from Tavily response
            tavily_results = api_response.get("results", [])
            parsed_results = []

            print(api_response)
            for i, web_page in enumerate(tavily_results):
                result = WebPageContent(
                    title=web_page.get("title", "No Title"),
                    url=web_page.get("url", "Unknown URL"),
                    summary=None,  # Summary not supported by tavily
                    publication_date=None,  # Tavily doesn't provide publish date in basic search
                    relevance_score=web_page.get(
                        "score", 1.0 - (i * 0.1)
                    ),  # Fallback assumes results in order of relevance
                    content=web_page.get(
                        "raw_content", None
                    ),  # Complete webpage content (usually unused)
                    relevant_chunks=web_page.get(
                        "content", None
                    ),  # Tavily's pre-chunked content
                )
                parsed_results.append(result)

            if not parsed_results:
                logger.warning(f"No results found for query: '{query}'")
                return SearchResult(
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
                f"Tavily search completed successfully: {len(parsed_results)} results in {search_time}ms"
            )

            print(f"{query=}\n{parsed_results=}")
            return SearchResult(
                query=query,
                results=parsed_results,
                summary=api_response.get("answer", None),
                total_results=len(parsed_results),
                search_time_ms=search_time,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Tavily API response for query '{query}': {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = (
                f"Failed to get or process Tavily API response for query '{query}': {e}"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

    async def _call_tavily_api(
        self, query: str, max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Make a live API call to Tavily search service.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            Dictionary containing the API response

        Raises:
            Exception: If API call fails or returns non-200 status
        """
        logger.debug(f"Making live Tavily API call for: '{query}'")

        # Prepare Tavily API request
        # TODO: Move this to an persistant connection on startup?
        conn = http.client.HTTPSConnection("api.tavily.com")

        # Tavily request payload with all-in-one RAG parameters
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",  # Use advanced to get chunks functionality
            "include_images": False,  # We don't need images for RAG
            "include_raw_content": False,  # We'll use chunks instead of raw content
            "max_results": min(max_results, 10),  # Tavily max is 10
            "include_domains": None,
            "exclude_domains": None,
            "days": None,  # No time limit by default
            "chunks_per_source": 5,  # Get 3 chunks per source
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Make the API request
        conn.request("POST", "/search", json.dumps(payload), headers)
        res = conn.getresponse()
        data = res.read()
        conn.close()

        # Parse the response
        api_response = json.loads(data.decode("utf-8"))

        if res.status != 200:
            error_msg = api_response.get(
                "error", f"Tavily API returned status {res.status}"
            )
            logger.error(f"Tavily API error: {error_msg}")
            raise Exception(f"Tavily API error: {error_msg}")

        return api_response
