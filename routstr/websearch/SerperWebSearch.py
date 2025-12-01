"""
Serper API Searcher Module

This module provides a WebSearchProvider implementation that uses the Serper API
to get search results and formats them into the standard WebSearchResponse.
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


class SerperWebSearch(BaseWebSearch):
    """
    A web search provider that uses the Serper API to get search results.
    """
    provider_name = "Serper"
    
    
    def __init__(self, api_key: str, scraper: Optional[BaseWebScraper] = None):
        """
        Initialize the Serper provider.
        
        Args:
            api_key: The Serper API key.
            scraper: An optional web scraper instance. If None, a DummyWebScraper is used.
        """
        # 1. Call the parent's __init__ method FIRST.
        # This is what sets up self.scraper.
        super().__init__(scraper=scraper)
        
        # 2. Now, do the Serper-specific initialization.
        if not api_key:
            raise ValueError("Serper API key cannot be empty.")
        self.api_key = api_key
        
        # The logger from the parent will have already run, so you can add your own.
        logger.info(f"SerperWebSearch initialized with API key.")

    async def search(self, query: str, max_results: int = 10) -> WebSearchResponse:
        """
        Perform web search using the Serper API and return a WebSearchResponse.
        """
        start_time = datetime.now()
        logger.info(f"Performing Serper API search for: '{query}'")

        try:
            from pathlib import Path 
            # --- THIS IS THE PART YOU WANTED TO KEEP UNREACHABLE FOR NOW ---
            # It will use a local file instead of making a live API call.
            logger.debug("Using dummy data from 'serper_trump-peace-plan.json'.")
            script_dir = Path(__file__).parent
            json_file_path = script_dir / 'serper_trump-peace-plan.json'
            with open(json_file_path, 'r', encoding='utf-8') as file:
                api_response = json.load(file)
            # ---------------------------------------------------------------

            # --- UNCOMMENT THE BLOCK BELOW FOR LIVE API CALLS ---
            #
            # conn = http.client.HTTPSConnection("google.serper.dev")
            # payload = json.dumps({"q": query, "num": max_results})
            # headers = {
            #     f'X-API-KEY': self.api_key,
            #     'Content-Type': 'application/json'
            # }
            # conn.request("POST", "/search", payload, headers)
            # res = conn.getresponse()
            # data = res.read()
            # api_response = json.loads(data.decode("utf-8"))
            # conn.close()
            #
            # -----------------------------------------------------------------

            # Parse the organic results from the API response
            organic_results = api_response.get('organic', [])
            parsed_results = []
            for i, item in enumerate(organic_results):
                result = WebSearchResult(
                    title=item.get('title', 'No Title'),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', 'No Snippet Available.'),
                    published_date=item.get('date'),
                    relevance_score=1.0 - (i * 0.1) # Simple relevance based on position
                )
                parsed_results.append(result)

            if not parsed_results:
                logger.warning(f"No organic results found for query: '{query}'")
                # Return an empty response, not an error
                return WebSearchResponse(
                    query=query,
                    results=[],
                    summary=f"No search results found for '{query}'.",
                    total_results=0,
                    search_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )

            # Create summary using the inherited method from the base class
            summary = ""
            
            # Calculate search time
            search_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return WebSearchResponse(
                query=query,
                results=parsed_results,
                summary=summary,
                total_results=len(parsed_results),
                search_time_ms=search_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        except FileNotFoundError:
            error_msg = "Dummy data file 'serper_trump-peace-plan.json' not found."
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Failed to get or process Serper API response for query '{query}': {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
