"""
Base class for Web search for AI context enhancement using Retrieval Augmented Generation (RAG).

"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from urllib.parse import urlparse

from ..core.logging import get_logger
from .types import SearchResult

logger = get_logger(__name__)


class BaseWebSearch:
    """Base class for web search providers."""

    provider_name: str = "Base"

    # TODO: Only for typechecking?
    def __init__(self) -> None:
        """Initialize the base web search provider."""
        self.client_timeout: httpx.Timeout = httpx.Timeout(3.0, connect=3.0)
        self.client_headers: dict = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        }
        self.client_redirects: bool = True

        # Domain Blocklist will be used by Search provider if domain exclusion is supported
        # TODO: Unify this with BaseRAG.BLOCKED_DOMAINS?
        self.BLOCKED_DOMAINS = {
            "youtube.com", "youtu.be",
            "vimeo.com",
            "tiktok.com",
            "instagram.com", 
            "facebook.com",
            }

    async def search(self, query: str, max_results: int = 5) -> SearchResult:
        """Perform web search and return results."""
        raise NotImplementedError("Subclasses must implement search method")

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

    
    def is_blocked(self, url: str) -> bool:
        """Check if a URL's domain is in the blocklist.

        Args:
            url: The URL to check.

        Returns:
            True if the domain is blocked, False otherwise.
        """
        # Extract domain from URL
        domain = urlparse(url).netloc
        # Remove 'www.' if present to ensure matching
        if domain.startswith("www."):
            domain = domain[4:]
        if domain in self.BLOCKED_DOMAINS:
            print(f"blocked: {url}")
        return domain in self.BLOCKED_DOMAINS


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
