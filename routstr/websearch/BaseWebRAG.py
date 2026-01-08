"""
Base classes for Retrieval Augmented Generation (RAG) providers.

This module defines the core interfaces and data structures for all-in-one RAG providers
that combine web search, content extraction, and chunking into a unified solution for
AI context enhancement.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from ..core.logging import get_logger
from .types import SearchResult

logger = get_logger(__name__)


class BaseWebRAG:
    """Base class for RAG providers.

    Defines the interface for providers that handle the complete RAG pipeline in single unified API call.
    """

    provider_name: str = "Base"

    def __init__(self) -> None:
        # httpx.AsyncClient configuration:
        self.client_timeout: httpx.Timeout = httpx.Timeout(3.0, connect=3.0)
        self.client_headers: dict = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        }
        self.client_redirects: bool = True

    async def make_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request using the httpx client.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative or absolute URL)
            headers: Additional headers (merged with client headers)
            payload: Request payload for POST/PUT requests

        Returns:
            Parsed JSON response

        Raises:
            Exception: If the request fails or returns non-200 status
        """
        if not hasattr(self, "base_url"):
            raise NotImplementedError("Subclass must define a 'base_url' attribute.")

        url = f"{self.base_url}{endpoint}"

        request_headers = self.client_headers.copy()
        if headers:
            request_headers.update(headers)

        async with httpx.AsyncClient(
            timeout=self.client_timeout,
            headers=self.client_headers,
            follow_redirects=True,
        ) as client:
            try:
                if method.upper() == "GET":
                    response = await client.get(url, headers=request_headers)
                elif method.upper() == "POST":
                    response = await client.post(
                        url, json=payload, headers=request_headers
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()  # Raise for non-200 status codes
                return response.json()

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP error {e.response.status_code} for {method} {url}"
                logger.error(error_msg)
                raise Exception(error_msg) from e
            except httpx.RequestError as e:
                error_msg = f"Request failed for {method} {url}: {e}"
                logger.error(error_msg)
                raise Exception(error_msg) from e

    async def retrieve(self, query: str, max_results: int = 5) -> SearchResult:
        """Perform web retrieval

        Args:
            query: The search query for retrieving relevant web content
            max_results: Maximum number of web sources to process

        Returns:
            SearchResult with processed content, chunks, and metadata

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement retrieve method")

    async def _load_mock_data(self, file_name: str) -> Dict[str, Any]:
        """Load mock API response data from local JSON file for testing purposes.

        Args:
            file_name: Name of the JSON file containing mock response data

        Returns:
            Dictionary containing mock API response data

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
        """Save live API response to timestamped JSON file for debugging and testing.

        Args:
            response_data: The API response dictionary to save
            query: The search query (used in filename generation)
            provider: Provider name (used in filename generation)

        Note:
            Creates files in api_responses/ directory with timestamp and sanitized query
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
        )[:60]
        filename = f"{provider}_{safe_query}_{timestamp}.json"
        file_path = responses_dir / filename

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            logger.info(f"API response saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save API response: {e}")

    async def check_availability(self) -> bool:
        """Check if the RAG provider service is available and API key is valid.

        Performs a lightweight API call to verify:
        - Service accessibility
        - API key validity
        - Service operational status

        Returns:
            True if provider is available and functional, False otherwise

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement check_availability method")
