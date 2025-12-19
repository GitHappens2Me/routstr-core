import json

from typing import Optional, List
from ..core.logging import get_logger
from ..core.settings import settings

logger = get_logger(__name__)



def get_rag_provider():
    """
    Get RAG provider based on RAG_PROVIDER configuration.
    This only returns true all-in-one RAG providers (like Tavily).
    
    Returns:
        BaseWebSearch instance or None if not available
    """
    if not settings.rag_provider:
        logger.debug("No RAG_PROVIDER configured")
        return None
    
    provider_name = settings.rag_provider.lower()
    logger.info(f"Checking configured RAG provider: {provider_name}")
    
    if provider_name == "tavily":
        try:
            from .TavilyWebSearch import TavilyWebSearch
            if not settings.tavily_api_key:
                logger.warning("Tavily selected as search provider but no API key configured")
                return None
            logger.info("Using Tavily RAG provider")
            return TavilyWebSearch(api_key=settings.tavily_api_key)
        except ImportError as e: #TODO: is this even necessary?
            logger.error(f"Failed to import TavilyWebSearch: {e}")
            return None
        
    if provider_name == "exa":
        try:
            from .ExaWebSearch import ExaWebSearch
            if not settings.exa_api_key:
                logger.warning("Exa selected as search provider but no API key configured")
                return None
            logger.info("Using Exa RAG provider")
            return ExaWebSearch(api_key=settings.exa_api_key)
        except ImportError as e:
            logger.error(f"Failed to import ExaWebSearch: {e}")
            return None
        
    else:
        logger.error(f"Unknown RAG provider: {provider_name}")
        return None


def get_web_search_provider():
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
    logger.info(f"Checking configured web search provider: {provider_name}")
    
    if provider_name == "serper":
        try:
            from .SerperWebSearch import SerperWebSearch
            if not settings.serper_api_key:
                logger.warning("Serper provider selected but no API key configured")
                return None
            logger.info("Using Serper web search provider")
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


def get_web_scraper_provider():
    """
    Get web scraper provider based on configuration.
    
    Returns:
        Web scraper provider instance or None if not available
    """
    scraper_name = settings.web_scraper_provider.lower()
    
    if scraper_name == "generic":
        try:
            from .BaseWebScraper import GenericWebScraper
            return GenericWebScraper()
        except ImportError as e:
            logger.error(f"Failed to import GenericWebScraper: {e}")
            return None
    
    elif scraper_name == "none":
        logger.info("No scraper configured. Web functionality disabled")
        return None
    
    else:
        logger.error(f"Unknown web scraper provider: {scraper_name}. Web functionality disabled")
        return None


def get_chunker_provider():
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
                chunk_size=settings.chunk_max_size,
                chunk_overlap=settings.chunk_overlap
            )
        except ImportError as e:
            logger.error(f"Failed to import FixedSizeChunker: {e}")
            return None
    elif chunker_name == "recursive":
        #try:
        #    from .RecursiveCharacterChunker import RecursiveCharacterChunker
        #    return RecursiveCharacterChunker(
        #        chunk_size=settings.chunk_max_size,
        #        chunk_overlap=settings.chunk_overlap
        #    )
        #except ImportError as e:
        #    logger.error(f"Failed to import RecursiveCharacterChunker: {e}")
            return None
    elif chunker_name == "none":
        logger.info("No chunker configured. Chunking functionality disabled")
        return None
    else:
        logger.error(f"Unknown chunker provider: {chunker_name}. Chunking functionality disabled")
        return None


async def enhance_request_with_web_context(request_body: bytes, query: str = None) -> bytes:
    """
    Enhance AI request with web search context using configured providers.
    This method orchestrates the complete Web-RAG pipeline:
    1. Check for RAG provider -> 2. Check for web search provider -> 3. Execute pipeline
    
    For RAG providers (Tavily): All-in-one pipeline
    For web search providers (Serper): Manual pipeline with scraper + chunker
    
    Args:
        request_body: The original request body as bytes
        query: Optional search query. If None, will extract from user message
        
    Returns:
        Enhanced request body with web context injected as bytes
    """
    try:
        # Step 1: Try to get RAG provider first
        rag_provider = get_rag_provider()
        
        if rag_provider:
            logger.info("Using RAG provider - all-in-one pipeline")
            # Step 2: Extract query (use provided query or extract from request)
            extracted_query = query or _extract_query_from_request_body(request_body)
            
            # Step 3: Perform RAG search (includes content extraction and chunking)
            max_web_searches = settings.web_search_max_results
            search_result = await rag_provider.search(extracted_query, max_web_searches)
            
            # Step 4: Inject context into request
            return await _inject_web_context_into_request(request_body, search_result, extracted_query)
        
        # Step 5: Fall back to manual web search pipeline
        search_provider = get_web_search_provider()
        scraper_provider = get_web_scraper_provider()
        chunker_provider = get_chunker_provider()
        
        if not search_provider:
            logger.warning("No RAG or web search provider available, cannot enhance request")
            return request_body
        
        logger.info("Using manual web search pipeline")
        # Step 6: Extract query
        extracted_query = query or _extract_query_from_request_body(request_body)
        
        # Step 7: Perform web search and scraping
        max_web_searches = settings.web_search_max_results
        search_result = await _perform_web_search_and_scraping(search_provider, scraper_provider, extracted_query, max_web_searches)
        
        # Step 8: Chunk the scraped content
        if settings.enable_chunking and chunker_provider and search_result.results:
            await _chunk_search_results(search_result.results, chunker_provider)

        for result in search_result.results:
            print(result.chunks)
        # Step 9: Inject chunked context into request
        return await _inject_web_context_into_request(request_body, search_result, extracted_query)
            
    except Exception as e:
        logger.error(
            f"Failed to enhance request with web context: {e}",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "query": query,
                "rag_provider": settings.rag_provider,
                "web_search_provider": settings.web_search_provider,
                "scraper_provider": settings.web_scraper_provider,
                "chunker_provider": settings.chunker_provider,
            }
        )
        return request_body


async def _perform_web_search_and_scraping(search_provider, scraper_provider, query: str, max_web_searches: int):
    """Perform web search and scraping, returning search response with scraped content."""

    # Perform search
    search_response = await search_provider.search(query, max_web_searches)
    
    # Scrape URLs from search results
    if search_response.results:
        urls_to_scrape = [result.url for result in search_response.results]
        logger.info(f"Scraping {len(urls_to_scrape)} URLs")
        
        max_concurrent_scrapes = settings.web_scrape_max_concurrent_urls
        scraped_contents = await scraper_provider.scrape_urls(urls_to_scrape, max_concurrent=max_concurrent_scrapes)
        
        # Map scraped content back to results
        content_map = {scraped.url: scraped.content for scraped in scraped_contents}
        for result in search_response.results:
            result.content = content_map.get(result.url)
    
    return search_response


async def _chunk_search_results(results, chunker_provider, query: str = None):
    """Chunk the content in search results."""
    logger.info(f"Chunking content for {len(results)} search results")
    
    for result in results:
        if result.content:
            chunks = await chunker_provider.chunk_text(result.content)
            # Rank chunks by relevance and limit chunks per source
            ranked_chunks = chunker_provider.rank_chunks(chunks, query)
            result.chunks = ranked_chunks[:settings.chunk_max_chunks_per_source]
            logger.debug(f"Created {len(result.chunks)} chunks for {result.url}")

#TODO: If search_response is None: inject a short message telling the model that                                                            # TODO: Add type (move Dataclasses to webmanager)
async def _inject_web_context_into_request(request_body: bytes, search_result, query: str) -> bytes:
    """
    Inject web search context into AI request, filtering out None values and empty content.
    
    Args:
        request_body: The original request body as bytes
        search_response: Either SearchResult object or None, if websearch failed
        query: The search query used
    
    Returns:
        Enhanced request body with web context injected as bytes
    """

    if(not search_result):
        return request_body

    # Add number of results
    web_context = f"Websearch yielded {len(search_result.results)} relevant results.\n"

    # Add optional summary
    if search_result.summary:
        web_context += f"Summary: '{search_result.summary}'\n"

    # Add results
    for i, web_page in enumerate(search_result.results, 1):
        web_context += f"Result {i}: [Title: '{web_page.title}', "
        web_context += f"URL: '{web_page.url}', "
        
        if web_page.snippet:
            web_context += f"Summary: '{web_page.snippet}', "
        
        if web_page.published_date:
            web_context += f"Publishing Date: '{web_page.published_date}', "
        
        if web_page.relevance_score:
            web_context += f"Relevance Score: '{web_page.relevance_score}', "
        
        web_context += f"Relevant Sections: '{web_page.chunks}']\n"
        
    # Parse and enhance the request
    try:
        request_data = json.loads(request_body.decode('utf-8'))
        
        web_context_message = {
            "role": "system",
            "content": web_context,
        }
        
        messages = request_data.get('messages', [])
        messages.append(web_context_message)
        request_data['messages'] = messages  

        enhanced_request_body = json.dumps(request_data).encode('utf-8')
        logger.info(f"Successfully injected web context for query: '{query}'")
        print("Enhanced Body:", request_data)
        return enhanced_request_body
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse request body for context injection: {e}")
        return request_body
    except Exception as e:
        logger.error(f"Unexpected error during context injection: {e}")
        return request_body


def _extract_query_from_request_body(request_body: bytes) -> str:
    """
    Extract search query from user messages in request body.
    
    Args:
        request_body: The request body as bytes
        
    Returns:
        Extracted query string or empty string if not found
    """
    try:
        import json
        data = json.loads(request_body)
        
        for message in data.get("messages", []):
            if message.get("role") == "user":
                content = message.get("content", "")
                # Simple extraction - use the full user message as query
                return content.strip()
        return ""
    except Exception as e:
        logger.warning(f"Failed to extract query from request body: {e}")
        return ""