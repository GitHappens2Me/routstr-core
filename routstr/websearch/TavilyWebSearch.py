"""
Tavily API Searcher Module

This module provides a WebSearchProvider implementation that uses the Tavily API
to get search results and formats them into the standard WebSearchResponse.
Tavily provides an all-in-one RAG solution with search, content extraction, and ranking.
"""

import http.client
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .BaseWebSearch import BaseWebSearch, WebSearchResponse, WebSearchResult
from .BaseWebScraper import BaseWebScraper
from ..core.logging import get_logger

logger = get_logger(__name__)


class TavilyWebSearch(BaseWebSearch):
    """
    A web search provider that uses the Tavily API to get search results.
    Tavily provides an all-in-one RAG solution with built-in content extraction
    and relevance ranking, eliminating the need for separate scraping and chunking.
    """
    provider_name = "Tavily"
    
    def __init__(self, api_key: str, scraper: Optional[BaseWebScraper] = None):
        """
        Initialize the Tavily provider.
        
        Args:
            api_key: The Tavily API key.
            scraper: Not used for Tavily as it provides content directly, but kept for interface compatibility.
        """
        # Call parent init but we won't use the scraper for Tavily
        super().__init__(scraper=None)
        
        if not api_key:
            raise ValueError("Tavily API key cannot be empty.")
        self.api_key = api_key
        
        logger.info(f"TavilyWebSearch initialized with API key. Tavily provides all-in-one RAG functionality.")

    async def search(self, query: str, max_results: int = 10) -> WebSearchResponse:
        """
        Perform web search using the Tavily API and return a WebSearchResponse.
        Tavily provides search results with pre-extracted and ranked content.
        """
        start_time = datetime.now()
        logger.info(f"Performing Tavily API search for: '{query}'")

        try:
            # --- MOCK DATA FOR TESTING (saves credits) ---
            # This will use the local file instead of making live API calls
            logger.debug("Using mock data from 'tavily_trump-peace-deal.json'.")
            from pathlib import Path
            script_dir = Path(__file__).parent
            json_file_path = script_dir / 'tavily_trump-peace-deal.json'
            with open(json_file_path, 'r', encoding='utf-8') as file:
                api_response = json.load(file)
            # ---------------------------------------------------------------
            
            # --- UNCOMMENT THE BLOCK BELOW FOR LIVE API CALLS ---
            # # Prepare Tavily API request
            # conn = http.client.HTTPSConnection("api.tavily.com")
            #
            # # Tavily request payload with all-in-one RAG parameters
            # payload = {
            #     "api_key": self.api_key,
            #     "query": query,
            #     "search_depth": "advanced",  # Use advanced to get chunks functionality
            #     "include_images": False,  # We don't need images for RAG
            #     "include_raw_content": False,  # We'll use chunks instead of raw content
            #     "max_results": min(max_results, 10),  # Tavily max is 10
            #     "include_domains": None,
            #     "exclude_domains": None,
            #     "days": None,  # No time limit by default
            #     "chunks_per_source": 3  # Get 3 chunks per source for focused content
            # }
            #
            # headers = {
            #     'Content-Type': 'application/json',
            #     'Authorization': f'Bearer {self.api_key}'
            # }
            #
            # # Make the API request
            # conn.request("POST", "/search", json.dumps(payload), headers)
            # res = conn.getresponse()
            # data = res.read()
            # conn.close()
            #
            # # Parse the response
            # api_response = json.loads(data.decode("utf-8"))
            #
            # if res.status != 200:
            #     error_msg = api_response.get('error', f'Tavily API returned status {res.status}')
            #     logger.error(f"Tavily API error: {error_msg}")
            #     raise Exception(f"Tavily API error: {error_msg}")
            # -----------------------------------------------------------------

            # Parse the results from Tavily response
            tavily_results = api_response.get('results', [])
            parsed_results = []
            
            print(api_response)
            for i, item in enumerate(tavily_results):
                content = item.get('content', '')
                result = WebSearchResult(
                    title=item.get('title', 'No Title'),
                    url=item.get('url', 'Unknown URL'),
                    snippet=None,  # Snippet not supported by tavily
                    published_date=None,  # Tavily doesn't provide publish date in basic search
                    relevance_score=item.get('score', 1.0 - (i * 0.1)),  # Use Tavily's score or fallback
                    content=None, 
                    chunks=content  # Store Tavily's pre-chunked content
                )
                parsed_results.append(result)

            if not parsed_results:
                logger.warning(f"No results found for query: '{query}'")
                return WebSearchResponse(
                    query=query,
                    results=[],
                    summary=f"No search results found for '{query}'.",
                    total_results=0,
                    search_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )

            # Create summary using Tavily's answer if available
            tavily_answer = api_response.get('answer', '')
            summary = tavily_answer if tavily_answer else f"Found {len(parsed_results)} relevant results for '{query}'."
            
            # Calculate search time
            search_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"Tavily search completed successfully: {len(parsed_results)} results in {search_time}ms")
            
            return WebSearchResponse(
                query=query,
                results=parsed_results,
                summary=summary,
                total_results=len(parsed_results),
                search_time_ms=search_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Tavily API response for query '{query}': {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Failed to get or process Tavily API response for query '{query}': {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
