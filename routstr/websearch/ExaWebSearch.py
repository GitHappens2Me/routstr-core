""" Exa API Searcher Module This module provides a WebSearchProvider implementation that uses the Exa API to get search results and formats them into the standard SearchResult. Exa provides intelligent web search with embeddings-based models and content extraction. """
import http.client
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from .BaseWebSearch import BaseWebSearch, SearchResult, WebPageContent
from .BaseWebScraper import BaseWebScraper
from ..core.logging import get_logger

logger = get_logger(__name__)

class ExaWebSearch(BaseWebSearch):
    """
    A web search provider that uses the Exa API to get search results.
    """
    provider_name = "Exa"

    def __init__(self, api_key: str, scraper: Optional[BaseWebScraper] = None):
        """
        Initialize the Exa provider.
        
        Args:
            api_key: The Exa API key.
            scraper: Unused (TODO: Can i remove it?)
        """
        super().__init__(scraper=None)
        if not api_key:
            raise ValueError("Exa API key cannot be empty.")
        self.api_key = api_key
        logger.info(f"ExaWebSearch initialized.")

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        """
        Perform web search using the Exa API and return a SearchResult instance.
        """
        start_time = datetime.now()
        logger.info(f"Performing Exa API search for: '{query}'")
        
        try:
            # --- MOCK DATA FOR TESTING ---
            api_response = await self._load_mock_data()
            # ---------------------------------------------------------------
            #api_response = await self._call_exa_api(query, max_results)
            #await self._save_api_response(api_response, query)
            # ---------------------------------------------------------------
            
            # Parse the results from Exa response
            exa_results = api_response.get('results', [])
            parsed_results = []
            
            print(api_response)
            for i, web_page in enumerate(exa_results):
                # Combine highlights to single string
                if highlights := web_page.get('highlights', None):
                    highlights = " [...] ".join(highlights)

                result = WebPageContent(
                    title = web_page.get('title', 'No Title'),
                    url = web_page.get('url', 'Unknown URL'),
                    snippet = web_page.get('summary', None), 
                    published_date = web_page.get('publishedDate', None),
                    relevance_score = web_page.get('score', 1.0 - (i * 0.1)), # Fallback assumes results in order of relevance
                    content = web_page.get('text', None),
                    chunks = highlights 
                )
                parsed_results.append(result)

            if not parsed_results:
                logger.warning(f"No results found for query: '{query}'")
                return SearchResult( #TODO: just return None?
                    query=query,
                    results=[],
                    summary=None,
                    total_results=0,
                    search_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )

            # Calculate search time
            search_time = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.info(f"Exa search completed successfully: {len(parsed_results)} results in {search_time}ms")
            
            print(f"{query=}\n{parsed_results=}" )
            return SearchResult(
                query=query,
                results=parsed_results,
                summary=None,
                total_results=len(parsed_results),
                search_time_ms=search_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Exa API response for query '{query}': {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Failed to get or process Exa API response for query '{query}': {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def _call_exa_api(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Make a live API call to Exa search service using neural search.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing the API response
            
        Raises:
            Exception: If API call fails or returns non-200 status
        """
        logger.debug(f"Making live Exa API call for: '{query}'")
        print("maximum results: ", max_results)
        # Prepare Exa API request
        #TODO: Move this to an persistant connection on startup?
        conn = http.client.HTTPSConnection("api.exa.ai")
        
        # Exa request payload configured for neural search with context
        payload = {
            "query": query,
            "type": "neural",
            "numResults": min(max_results, 10),  # Exa's max is 100, but a maximum of 10 is requested
            "contents": {
                "text": False,   # Complete page content
                "summary": False, # Short summary of page content
                "highlights": {  # Most relevant chunks
                    "numSentences": 3, # Number of sentences per highlight
                    "highlightsPerUrl": 5 # Chunks per URL
                } 
            },
            "context": True,  # Enable context to get combined content for LLMs
            "includeText": None,
            "excludeText": None,
            "startCrawlDate": None,
            "endCrawlDate": None,
            "startPublishedDate": None,
            "endPublishedDate": None,
            "includeDomains": None,
            "excludeDomains": None,
            "category": None,
            "subpageTarget": None,
            "subpages": 0, # Do not scrape subpages
            "livecrawl": "never",
            "extras": {
                "links": 0, 
                "imageLinks": 0
            }
        }
        
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key
        }
        
        # Make the API request
        conn.request("POST", "/search", json.dumps(payload), headers)
        res = conn.getresponse()
        data = res.read()
        conn.close()
        
        # Parse the response
        api_response = json.loads(data.decode("utf-8"))
        
        if res.status != 200:
            error_msg = api_response.get('error', f'Exa API returned status {res.status}')
            logger.error(f"Exa API error: {error_msg}")
            raise Exception(f"Exa API error: {error_msg}")
            
        return api_response

    async def _load_mock_data(self) -> Dict[str, Any]:
        """
        Load mock data from local JSON file for testing purposes.
        
        Returns:
            Dictionary containing mock API response
        """
        logger.debug("Using mock data from file.")
        from pathlib import Path
        script_dir = Path(__file__).parent
        json_file_path = script_dir / 'api_responses/exa_what_is_the_latest_news_about_the_Donald_Trump_peace_deal_Which_websites_did_you_search_be_brief_20251219_143125.json'
        with open(json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    async def _save_api_response(self, response_data: Dict[str, Any], query: str) -> None:
        """
        Save API response to a timestamped JSON file.
        
        Args:
            response_data: The API response dictionary to save
            query: The search query (used in filename)
        """
        from pathlib import Path
        import json
        
        # Create responses directory if it doesn't exist
        responses_dir = Path(__file__).parent / 'api_responses'
        responses_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp and sanitized query
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(" ", "_")
        filename = f"exa_{safe_query}_{timestamp}.json"
        file_path = responses_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            logger.info(f"API response saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save API response: {e}")