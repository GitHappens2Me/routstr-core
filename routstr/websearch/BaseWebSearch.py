"""
Base class for Web search for AI context enhancement using Retrieval Augmented Generation (RAG).

"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .types import SearchResult
from ..core.logging import get_logger

logger = get_logger(__name__)


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
        json_file_path = script_dir / f"api_responses/{file_name}"
        with open(json_file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    async def _save_api_response(
        self, response_data: Dict[str, Any], query: str, provider: str
    ) -> None:
        """
        Save API response to a timestamped JSON file.

        Args:
            response_data: The API response dictionary to save
            query: The search query (used in filename)
        """
        import json
        from pathlib import Path

        # Create responses directory if it doesn't exist
        responses_dir = Path(__file__).parent / "api_responses"
        responses_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp and sanitized query
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = (
            "".join(c for c in query if c.isalnum() or c in (" ", "-", "_"))
            .rstrip()
            .replace(" ", "_")
        )
        filename = f"{provider}_{safe_query}_{timestamp}.json"
        file_path = responses_dir / filename

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            logger.info(f"API response saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save API response: {e}")
