"""
Tavily API RAG Provider Module

This module provides a complete RAG implementation using the Tavily API.
Tavily offers an all-in-one solution combining web search, content extraction,
and intelligent chunking specifically designed for AI context enhancement.
"""

import http.client
import json
from datetime import datetime, timezone
from typing import Any, Dict

from ..core.logging import get_logger

from .BaseWebRAG import BaseWebRAG
from .types import SearchResult, WebPageContent
logger = get_logger(__name__)


class TavilyWebRAG(BaseWebRAG):
    """
    All-in-one RAG provider using the Tavily API for intelligent web content retrieval.
    
    Tavily handles the complete pipeline: web search, content extraction,
    relevance ranking, and chunking in a single API call, making it ideal
    for Retrieval Augmented Generation use cases.
    """

    provider_name = "Tavily"

    def __init__(self, api_key: str):
        """
        Initialize the Tavily RAG provider.

        Args:
            api_key: The Tavily API key for authentication
            
        Raises:
            ValueError: If API key is empty or None
        """

        if not api_key:
            raise ValueError("Tavily API key cannot be empty.")
        self.api_key = api_key

        logger.info("TavilyWebRAG initialized.")

    async def retrieve(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Perform complete RAG pipeline using Tavily's all-in-one API.
        
        Executes web search, content extraction, and chunking in a single call
        to Tavily's advanced search endpoint with RAG-optimized parameters.

        Args:
            query: The search query for retrieving relevant web content
            max_results: Maximum number of web sources to process (max 10 for Tavily)
            
        Returns:
            SearchResult with processed content, pre-chunked highlights, and metadata
            
        Raises:
            Exception: If API call fails or response parsing fails
        """
        start_time = datetime.now()
        logger.debug(f"Performing Tavily API search for: '{query}'")

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

            logger.debug(f"Tavily API response: {api_response}")
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

            logger.debug(f"Query: {query}, Results: {len(parsed_results)} items")
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
        

    # TODO: http.client.HTTPSConnection is not asynchronous
    async def _call_tavily_api(
        self, query: str, max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Make live API call to Tavily's advanced search endpoint.

        Args:
            query: The search query
            max_results: Maximum number of results to return (max 10)
            
        Returns:
            Dictionary containing the complete Tavily API response
            
        Raises:
            Exception: If API call fails or returns non-200 status
            
        Note:
            Uses advanced search depth with chunking enabled for optimal RAG performance
        """
        logger.debug(f"Making live Tavily API call for: '{query}'")

        # Prepare Tavily API request
        # TODO: Move this to an persistant connection on startup?
        conn = http.client.HTTPSConnection("api.tavily.com")

        # Tavily request payload with all-in-one RAG parameters
        payload = {
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

    # TODO: http.client.HTTPSConnection is not asynchronous
    async def check_availability(
        self,
    ) -> bool:
        """
        Verify Tavily service availability and API key validity.
        
        Makes a lightweight call to Tavily's usage endpoint to confirm:
        - Service is accessible and operational
        - API key is valid and authorized
        - Rate limits and quotas are available

        Returns:
            True if Tavily service is available and API key is valid, False otherwise
        """
        logger.info("Checking Tavily API availability")

        try:
            # Prepare Tavily API request
            # TODO: Move this to an persistant connection on startup?
            conn = http.client.HTTPSConnection("api.tavily.com")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            # Make the API request
            conn.request("GET", "/usage", body=None, headers=headers)
            res = conn.getresponse()
            data = res.read()
            conn.close()

            # Parse the response
            api_response = json.loads(data.decode("utf-8"))
                        
            logger.debug(f"Tavily usage endpoint response: {api_response}")
            
            if res.status != 200:
                error_msg = api_response.get(
                    "error", f"Tavily API not available: {res.status}"
                )
                logger.error(f"Tavily availability check failed: {error_msg}")
                return False

            logger.info("Tavily API availability check completed succesfully")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Tavily availability check failed - JSON decode error: {e}")
            return False
        except Exception as e:
            logger.error(f"Tavily availability check failed - unexpected error: {e}")
            return False