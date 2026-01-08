import json
from typing import Optional

from ..core.logging import get_logger
from ..core.settings import settings
from .BaseWebChunker import BaseWebChunker
from .BaseWebRAG import BaseWebRAG
from .BaseWebScraper import BaseWebScraper
from .BaseWebSearch import BaseWebSearch, SearchResult
from .CustomRAG import CustomRAG

logger = get_logger(__name__)


# TODO: Add availibity check!
async def get_rag_provider() -> Optional[BaseWebRAG]:
    """
    Get RAG provider based on RAG_PROVIDER configuration.
    This only returns true all-in-one RAG providers (like Tavily).

    Returns:
        BaseWebRAG instance or None if not available
    """
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
                return tavily
            except ImportError as e:  # TODO: is this even necessary?
                logger.error(f"Failed to import TavilyWebRAG: {e}")
                return None
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
                logger.info("Using Exa as RAG provider")
                exa = ExaWebRAG(api_key=settings.exa_api_key)
                if not await exa.check_availability():
                    logger.warning(
                        "Exa availability check failed - service may be unavailable or API key invalid"
                    )
                    return None
                logger.info("Using Exa as RAG provider")
                return exa
            except ImportError as e:
                logger.error(f"Failed to initialize ExaWebRAG: {e}")
                return None

        case "custom":
            try:
                # Get individual providers for custom pipeline
                search_provider = get_web_search_provider()
                scraper_provider = get_web_scraper_provider()
                chunker_provider = get_web_chunker_provider()

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

                custom_rag = CustomRAG(
                    search_provider, scraper_provider, chunker_provider
                )
                if not await custom_rag.check_availability():
                    logger.warning(
                        "Custom RAG availability check failed - some components may be unavailable"
                    )
                    return None

                logger.info("Using Custom RAG provider")
                return custom_rag

            except Exception as e:
                logger.error(f"Failed to initialize CustomRAG: {e}")
                return None

        case _:
            logger.error(f"Unknown RAG provider: {provider_name}")
            return None


def get_web_search_provider() -> Optional[BaseWebSearch]:
    """
    Get web search provider based on WEB_SEARCH_PROVIDER configuration.
    This only returns web search providers (like Serper), not RAG providers.

    Returns:
        Web search provider instance or None if not available
    """
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
            # logger.info("Using Serper web search provider")
            return SerperWebSearch(api_key=settings.serper_api_key)
        except ImportError as e:
            logger.error(f"Failed to import SerperWebSearch: {e}")
            return None
    elif provider_name == "none":
        logger.info("No web search provider configured")
        return None
    else:
        logger.error(f"Unknown web search provider: {provider_name}")
        return None


def get_web_scraper_provider() -> Optional[BaseWebScraper]:
    """
    Get web scraper provider based on configuration.

    Returns:
        Web scraper provider instance or None if not available
    """
    scraper_name = settings.web_scraper_provider.lower()

    if scraper_name == "default":
        try:
            from .DefaultWebScraper import DefaultWebScraper

            return DefaultWebScraper()
        except ImportError as e:
            logger.error(f"Failed to import DefaultWebScraper: {e}")
            return None

    else:
        logger.error(
            f"Unknown web scraper provider: {scraper_name}. Web functionality disabled"
        )
        return None


def get_web_chunker_provider() -> Optional[BaseWebChunker]:
    """
    Get chunker provider based on configuration.

    Returns:
        Chunker provider instance or None if not available
    """
    chunker_name = settings.chunker_provider.lower()

    if chunker_name == "fixed":
        try:
            from .FixedSizeChunker import FixedSizeChunker

            return FixedSizeChunker(
                chunk_size=settings.chunk_max_size, chunk_overlap=settings.chunk_overlap
            )
        except ImportError as e:
            logger.error(f"Failed to import FixedSizeChunker: {e}")
            return None
    elif chunker_name == "recursive":
        try:
            from .RecursiveChunker import RecursiveChunker
            return RecursiveChunker(
                chunk_size=settings.chunk_max_size,
                chunk_overlap=settings.chunk_overlap
            )
        except ImportError as e:
            logger.error(f"Failed to import RecursiveCharacterChunker: {e}")
        return None
    elif chunker_name == "none":
        logger.info("No chunker configured. Chunking functionality disabled")
        return None
    else:
        logger.error(
            f"Unknown chunker provider: {chunker_name}. Chunking functionality disabled"
        )
        return None


async def enhance_request_with_web_context(request_body: bytes) -> tuple[bytes, list[dict]]:
    """
    Enhance AI request with web search context using configured RAG provider.

    This method uses the unified RAG interface that handles the complete pipeline:
    - All-in-one providers (Tavily, Exa): Search + extract + chunk in one call
    - Custom provider: Manual pipeline with separate search, scrape, and chunk components

    Args:
        request_body: The original request body as bytes

    Returns:
        Tuple of (Enhanced request body as bytes, List of source dictionaries)
    """
    try:
        # Get configured RAG provider (all-in-one or custom)
        rag_provider = await get_rag_provider()

        if not rag_provider:
            logger.warning("No RAG provider available, cannot enhance request")
            return request_body, []

        # Extract query from request
        extracted_query = _extract_query_from_request_body(request_body)

        # Perform complete RAG pipeline (handles all complexity internally)
        max_web_searches = settings.web_search_max_results
        search_result = await rag_provider.retrieve(extracted_query, max_web_searches)

        # Inject context into request
        return await _inject_web_context_into_request(
            request_body, search_result, extracted_query
        )

    except Exception as e:
        logger.error(
            f"Failed to enhance request with web context: {e}",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "rag_provider": settings.web_rag_provider,
            },
        )
        return request_body, []


def _extract_query_from_request_body(request_body: bytes) -> str:
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
                print(f"Extracted Query: {content}")
                return content.strip()

        return ""

    except Exception as e:
        logger.warning(f"Failed to extract query from request body: {e}")
        return ""

# TODO: If search_response is None: inject a short message telling the model that                                                            # TODO: Add type (move Dataclasses to webmanager)
async def _inject_web_context_into_request(
    request_body: bytes, search_result: SearchResult, query: str
) -> tuple[bytes, list[dict]]:
    """
    Inject web search context into AI request, filtering out None values and empty content.

    Args:
        request_body: The original request body as bytes
        search_response: Either SearchResult object or None, if websearch failed
        query: The search query used

    Returns:
        Tuple of (Enhanced request body as bytes, List of source dictionaries)
    """

    if not search_result:
        return request_body, []

    # Add number of results
    web_context = f"Websearch yielded {len(search_result.results)} relevant results.\n"

    # Add optional summary
    if search_result.summary:
        web_context += f"Summary: '{search_result.summary}'\n"

    # Prepare sources list for the response body
    sources = []

    # Add results
    for i, web_page in enumerate(search_result.results, 1):
        web_context += f"Result {i}: [Title: '{web_page.title}', "
        web_context += f"URL: '{web_page.url}', "
        if web_page.summary:
            web_context += f"Summary: '{web_page.summary}', "

        if web_page.publication_date:
            web_context += f"Publishing Date: '{web_page.publication_date}', "

        if web_page.relevance_score:
            web_context += f"Relevance Score: '{web_page.relevance_score}', "

        web_context += f"Relevant Sections: '{web_page.relevant_chunks}']\n"

         
        # We can also add the title:
        #sources.append({
        #   "title": web_page.title,
        #   "url": web_page.url
        #)

        sources.append(web_page.url)

    # Parse and enhance the request
    try:
        request_data = json.loads(request_body.decode("utf-8"))

        web_context_message = {
            "role": "system",
            "content": web_context,
        }

        messages = request_data.get("messages", [])
        messages.append(web_context_message)
        request_data["messages"] = messages

        enhanced_request_body = json.dumps(request_data).encode("utf-8")
        logger.info(f"Successfully injected web context for query: '{query}'")
        #logger.debug(f"Enhanced Body: '{request_data}'")
        return enhanced_request_body, sources

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse request body for context injection: {e}")
        return request_body, sources
    except Exception as e:
        logger.error(f"Unexpected error during context injection: {e}")
        return request_body, sources
