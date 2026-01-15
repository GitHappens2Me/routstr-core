import json
from typing import Optional, Dict, Any, Tuple

from ..core.logging import get_logger
from ..core.settings import settings
from .BaseWebChunker import BaseWebChunker
from .BaseWebRanker import BaseWebRanker
from .BaseWebRAG import BaseWebRAG
from .BaseWebScraper import BaseWebScraper
from .BaseWebSearch import BaseWebSearch
from .CustomRAG import CustomRAG
from .types import SearchResult

logger = get_logger(__name__)


class WebManager:
    """
    Manager class for web search and RAG functionality.
    Handles provider initialization, caching, and request enhancement.
    """

    def __init__(self):
        self._rag_provider: Optional[BaseWebRAG] = None
        self._search_provider: Optional[BaseWebSearch] = None
        self._scraper_provider: Optional[BaseWebScraper] = None
        self._chunker_provider: Optional[BaseWebChunker] = None
        self._rank_provider: Optional[BaseWebRanker] = None 

    async def get_rag_provider(self) -> Optional[BaseWebRAG]:
        """
        Get RAG provider based on RAG_PROVIDER configuration.
        This only returns true all-in-one RAG providers (like Tavily).

        Returns:
            BaseWebRAG instance or None if not available
        """
        if self._rag_provider:
            return self._rag_provider

        if not settings.web_rag_provider:
            logger.debug("No RAG_PROVIDER configured")
            return None

        provider_name = settings.web_rag_provider.lower()

        match provider_name:
            case "tavily":
                try:
                    from .TavilyWebRAG import TavilyWebRAG

                    if not settings.tavily_api_key:
                        logger.warning(
                            "Tavily selected as RAG provider but no API key configured"
                        )
                        return None
                    tavily = TavilyWebRAG(api_key=settings.tavily_api_key)
                    if not await tavily.check_availability():
                        logger.warning(
                            "Tavily availability check failed - service may be unavailable or API key invalid"
                        )
                        return None

                    logger.info("Using Tavily as RAG provider")
                    self._rag_provider = tavily
                    return self._rag_provider
                except Exception as e:
                    logger.error(f"Failed to initialize TavilyWebRAG: {e}")
                    return None

            case "exa":
                try:
                    from .ExaWebRAG import ExaWebRAG

                    if not settings.exa_api_key:
                        logger.warning(
                            "Exa selected as RAG provider but no API key configured"
                        )
                        return None
                    exa = ExaWebRAG(api_key=settings.exa_api_key)
                    if not await exa.check_availability():
                        logger.warning(
                            "Exa availability check failed - service may be unavailable or API key invalid"
                        )
                        return None
                    logger.info("Using Exa as RAG provider")
                    self._rag_provider = exa
                    return self._rag_provider
                except Exception as e:
                    logger.error(f"Failed to initialize ExaWebRAG: {e}")
                    return None

            case "custom":
                try:
                    # Get individual providers for custom pipeline
                    search_provider = await self.get_web_search_provider()
                    scraper_provider = await self.get_web_scraper_provider()
                    chunker_provider = await self.get_web_chunker_provider()
                    ranker_provider = await self.get_web_ranker_provider()

                    if not search_provider:
                        logger.warning(
                            "Custom RAG provider selected but no web search provider available"
                        )
                        return None
                    if not scraper_provider:
                        logger.warning(
                            "Custom RAG provider selected but no web scraper provider available"
                        )
                        return None
                    if not chunker_provider:
                        logger.warning(
                            "Custom RAG provider selected but no chunker provider available"
                        )
                        return None
                    if not ranker_provider: # Add validation
                        logger.warning("Custom RAG provider selected but no ranker provider available")
                        return None

                    custom_rag = CustomRAG(
                        search_provider, scraper_provider, chunker_provider, ranker_provider 
                    )
                    if not await custom_rag.check_availability():
                        logger.warning(
                            "Custom RAG availability check failed - some components may be unavailable"
                        )
                        return None

                    logger.info("Using Custom RAG provider")
                    self._rag_provider = custom_rag
                    return self._rag_provider

                except Exception as e:
                    logger.error(f"Failed to initialize CustomRAG: {e}")
                    return None

            case _:
                logger.error(f"Unknown RAG provider: {provider_name}")
                return None

    async def get_web_search_provider(self) -> Optional[BaseWebSearch]:
        """
        Get web search provider based on WEB_SEARCH_PROVIDER configuration.
        This only returns web search providers (like Serper)

        Returns:
            Web search provider instance or None if not available
        """
        if self._search_provider:
            return self._search_provider

        if not settings.web_search_provider:
            logger.debug("No WEB_SEARCH_PROVIDER configured")
            return None

        provider_name = settings.web_search_provider.lower()
        if provider_name == "serper":
            try:
                from .SerperWebSearch import SerperWebSearch

                if not settings.serper_api_key:
                    logger.warning("Serper provider selected but no API key configured")
                    return None
                self._search_provider = SerperWebSearch(api_key=settings.serper_api_key)
                return self._search_provider
            except Exception as e:
                logger.error(f"Failed to initialize SerperWebSearch: {e}")
                return None
        elif provider_name == "none":
            logger.info("No web search provider configured")
            return None
        else:
            logger.error(f"Unknown web search provider: {provider_name}")
            return None

    async def get_web_scraper_provider(self) -> Optional[BaseWebScraper]:
        """
        Get web scraper provider based on configuration.

        Returns:
            Web scraper provider instance or None if not available
        """
        if self._scraper_provider:
            return self._scraper_provider

        scraper_name = settings.web_scraper_provider.lower()

        if scraper_name == "default":
            try:
                from .DefaultWebScraper import DefaultWebScraper

                self._scraper_provider = DefaultWebScraper()
                return self._scraper_provider
            except Exception as e:
                logger.error(f"Failed to initialize DefaultWebScraper: {e}")
                return None

        else:
            logger.error(
                f"Unknown web scraper provider: {scraper_name}. Web functionality disabled"
            )
            return None

    async def get_web_chunker_provider(self) -> Optional[BaseWebChunker]:
        """
        Get chunker provider based on configuration.

        Returns:
            Chunker provider instance or None if not available
        """
        if self._chunker_provider:
            return self._chunker_provider

        chunker_name = settings.chunker_provider.lower()

        if chunker_name == "fixed":
            try:
                from .FixedSizeChunker import FixedSizeChunker

                self._chunker_provider = FixedSizeChunker(
                    chunk_size=settings.chunk_max_size, chunk_overlap=settings.chunk_overlap
                )
                return self._chunker_provider
            except Exception as e:
                logger.error(f"Failed to initialize FixedSizeChunker: {e}")
                return None
        elif chunker_name == "recursive":
            try:
                from .RecursiveChunker import RecursiveChunker
                self._chunker_provider = RecursiveChunker(
                    chunk_size=settings.chunk_max_size,
                    chunk_overlap=settings.chunk_overlap
                )
                return self._chunker_provider
            except Exception as e:
                logger.error(f"Failed to initialize RecursiveChunker: {e}")
                return None
        elif chunker_name == "none":
            logger.info("No chunker configured. Chunking functionality disabled")
            return None
        else:
            logger.error(
                f"Unknown chunker provider: {chunker_name}. Chunking functionality disabled"
            )
            return None


    async def get_web_ranker_provider(self) -> Optional[BaseWebRanker]:
        """
        Get ranker provider based on configuration.
        Returns: Ranker provider instance or None if not available
        """
        if self._rank_provider:
            return self._rank_provider

        # If setting is missing, we'll default to 'bm25'
        ranker_name = settings.web_ranking_provider.lower()

        if ranker_name == "bm25":
            try:
                from .BM25WebRanker import BM25WebRanker
                self._rank_provider = BM25WebRanker()
                return self._rank_provider
            except Exception as e:
                logger.error(f"Failed to initialize BM25WebRanker: {e}")
                return None
        elif ranker_name == "none":
            return None
        else:
            logger.error(f"Unknown ranker provider: {ranker_name}")
            return None


    async def enhance_request_with_web_context(
        self, request_body: bytes
    ) -> dict[str, Any]:
        """
        Enhance AI request with web search context using configured RAG provider.

        This method uses the unified RAG interface that handles the complete pipeline:
        - All-in-one providers (Tavily, Exa): Search + extract + chunk in one call
        - Custom provider: Manual pipeline with separate search, scrape, and chunk components

        Args:
            request_body: The original request body as bytes

        Returns:
            Dict containing:
            - 'body': Enhanced request body as bytes
            - 'sources': List of source strings
            - 'success': Boolean indicating if RAG was successful and yielded results
        """
        try:
            # Get configured RAG provider (all-in-one or custom)
            rag_provider = await self.get_rag_provider()

            if not rag_provider:
                logger.warning("No RAG provider available, cannot enhance request")
                return {"body": request_body, "sources": [], "success": False}

            # Extract query from request
            extracted_query = self._extract_query_from_request_body(request_body)

            if not extracted_query:
                return {"body": request_body, "sources": [], "success": False}

            # Perform complete RAG pipeline (handles all complexity internally)
            max_web_searches = settings.web_search_max_results
            search_result = await rag_provider.retrieve_context(
                extracted_query, max_web_searches
            )

            # Inject context into request
            enhanced_body, sources = await self._inject_web_context_into_request(
                request_body, search_result, extracted_query
            )

            # If we have results, it's a success
            success = bool(search_result and search_result.results)
            return {"body": enhanced_body, "sources": sources, "success": success}

        except Exception as e:
            logger.error(
                f"Failed to enhance request with web context: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "rag_provider": settings.web_rag_provider,
                },
            )
            return {"body": request_body, "sources": [], "success": False}


    @staticmethod
    def extract_web_search_parameter(body: bytes | None) -> Tuple[bytes | None, bool]:
        """
        Extracts the 'enable_web_search' parameter from a JSON request body.

        This function parses the body, extracts the boolean value of the
        'enable_web_search' key, and returns the body bytes with the key removed
        to prevent it from being forwarded to the upstream provider.

        Args:
            body: The raw request body as bytes

        Returns:
            A tuple containing:
            - bytes | None: The modified body as bytes, with 'enable_web_search' removed.
                            Returns the original body if parsing fails or it's not JSON.
            - bool: The extracted value of 'enable_web_search', defaulting to False.
        """
        if not body:
            return None, False

        try:
            body_dict = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to decode or parse request body as JSON for web search extraction.",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            return body, False

        enable_web_search = bool(body_dict.pop("enable_web_search", False))

        # Serialize the modified dictionary back to bytes
        try:
            cleaned_body = json.dumps(body_dict).encode("utf-8")
            return cleaned_body, enable_web_search
        except (TypeError, ValueError) as e:
            # Log the error and return the original body
            logger.error(
                "Failed to re-serialize request body after removing web search parameter.",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            return body, enable_web_search  # Still return the flag


    def _extract_query_from_request_body(self, request_body: bytes) -> str:
        """
        Extract search query from user messages in request body.

        Args:
            request_body: The request body as bytes

        Returns:
            Extracted query string or empty string if not found
        """
        try:
            data = json.loads(request_body)
            messages = data.get("messages", [])
            
            # Iterate in reverse (from end to start)
            for message in reversed(messages):
                if message.get("role") == "user":
                    content = message.get("content", "")
                    # print(f"Extracted Query: {content}")
                    return content.strip()

            return ""

        except Exception as e:
            logger.warning(f"Failed to extract query from request body: {e}")
            return ""

    async def _inject_web_context_into_request(
        self, request_body: bytes, search_result: SearchResult, query: str
    ) -> tuple[bytes, list[str]]:
        """
        Inject web search context into AI request, filtering out None values and empty content.

        Args:
            request_body: The original request body as bytes
            search_result: Either SearchResult object or None, if websearch failed
            query: The search query used

        Returns:
            Tuple of (Enhanced request body as bytes, List of source strings)
        """

        if not search_result or not search_result.results:
            return request_body, []

        # Prepare sources list for the response body
        sources = []

        # Build structured XML context
        context_parts = ["<search_results>"]
        context_parts.append(f"Websearch yielded {len(search_result.results)} relevant results for query '{query}'.\n")

        if search_result.summary:
            context_parts.append(f"Summary: {search_result.summary}\n")

        for i, web_page in enumerate(search_result.results, 1):

            sources.append(web_page.url)
            
            result_block = [f"<result id=\"{i}\">"]
            if web_page.title:
                result_block.append(f"<title>{web_page.title}</title>")
            result_block.append(f"<url>{web_page.url}</url>")
            
            if web_page.publication_date:
                result_block.append(f"<date>{web_page.publication_date}</date>")
            
            if web_page.relevance_score:
                result_block.append(f"<relevance_score>{web_page.relevance_score}</relevance_score>")
            
            if web_page.summary:
                result_block.append(f"<summary>{web_page.summary}</summary>")
            
            if web_page.relevant_chunks:
                joined_chunks = "\n\n".join(web_page.relevant_chunks)
                result_block.append(f"<content>\n{joined_chunks}\n    </content>")
            
            result_block.append("</result>")
            context_parts.append("\n".join(result_block))

        context_parts.append("</search_results>\n")
        context_parts.append("Please use the provided search results to answer the user's request. If the information is not available in the results, state that you don't know based on the web search.")
        
        web_context = "\n".join(context_parts)
        #print(web_context) #TODO: DEBUG PRINT
        # Parse and enhance the request
        try:
            request_data = json.loads(request_body.decode("utf-8"))
            messages = request_data.get("messages", [])

            web_context_message = {
                "role": "system",
                "content": web_context,
            }

            # Inject context just before the last user message
            # Find the index of the last user message
            last_user_index = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_index = i
                    break
            
            if last_user_index != -1:
                messages.insert(last_user_index, web_context_message)
            else:
                # Fallback: append if no user message found (shouldn't happen in normal chat)
                messages.append(web_context_message)

            request_data["messages"] = messages

            enhanced_request_body = json.dumps(request_data).encode("utf-8")
            logger.info(f"Successfully injected web context for query: '{query}'")
            return enhanced_request_body, sources

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse request body for context injection: {e}")
            return request_body, sources
        except Exception as e:
            logger.error(f"Unexpected error during context injection: {e}")
            return request_body, sources


# Singleton instance
web_manager = WebManager()
