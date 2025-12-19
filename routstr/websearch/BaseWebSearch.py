"""
Web search module for AI context enhancement.

This module provides web search functionality to enhance AI responses with current information from the web. 
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from ..core.logging import get_logger


logger = get_logger(__name__)

# TODO: Maybe move to WebManager??
@dataclass
class WebPageContent:
    """Retrieved content from a single URL"""
    title: str
    url: str
    snippet: str # Can contain a content summery #TODO: rename to summery (and make optional?)
    published_date: Optional[str] = None
    relevance_score: float = 0.0 #TODO: Is this necessary (Should i be made optional)
    content: Optional[str] = None
    chunks: Optional[str] = None # Relevant Chunks as one LLM-readable string. #TODO: Rename?

# TODO: Maybe move to WebManager??
@dataclass
class SearchResult:
    """Complete web search result with context for AI."""
    query: str
    results: List[WebPageContent]
    summary: str
    total_results: int
    search_time_ms: int
    timestamp: str


class BaseWebSearch:
    """Base class for web search providers."""
    
    provider_name: str = "Base"

    async def search(self, query: str, max_results: int = 5) -> SearchResult:
        """Perform web search and return results."""
        raise NotImplementedError("Subclasses must implement search method")
    
    async def _load_mock_data(self, file_name: str) -> Dict[str, Any]:
        """
        Load mock data from local JSON file for testing purposes.
        
        Returns:
            Dictionary containing mock API response
        """
        logger.debug("Using mock data from file.")
        from pathlib import Path
        script_dir = Path(__file__).parent
        json_file_path = script_dir / f'api_responses/{file_name}'
        with open(json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    async def _save_api_response(self, response_data: Dict[str, Any], query: str, provider: str) -> None:
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
        filename = f"{provider}_{safe_query}_{timestamp}.json"
        file_path = responses_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            logger.info(f"API response saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save API response: {e}")