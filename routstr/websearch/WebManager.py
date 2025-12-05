"""
Factory module for creating web search and scraper instances based on configuration.
"""

from typing import Optional, List
from ..core.logging import get_logger
from ..core.settings import settings

logger = get_logger(__name__)



def get_web_search_provider():
    """
    Get web search provider based on configuration.
    
    Returns:
        Web search provider instance or None if not available
    """
    #TODO: Test what happens if env variable does not exist
    provider_name = settings.web_search_provider.lower()
    
    # Auto-detection: If only Tavily API key is set, default to Tavily
    if provider_name == "serper" and not settings.serper_api_key and settings.tavily_api_key:
        logger.info("Auto-detecting Tavily provider (API key found, Serper key missing)")
        provider_name = "tavily"
    
    if provider_name == "tavily":
        try:
            from .TavilyWebSearch import TavilyWebSearch
            if not settings.tavily_api_key:
                logger.warning("Tavily provider selected but no API key configured")
                return None
            logger.info("Using Tavily all-in-one RAG provider")
            return TavilyWebSearch(api_key=settings.tavily_api_key)
        except ImportError as e:
            logger.error(f"Failed to import TavilyWebSearch: {e}")
            return None
    elif provider_name == "serper":
        try:
            from .SerperWebSearch import SerperWebSearch
            if not settings.serper_api_key:
                logger.warning("Serper provider selected but no API key configured")
                return None
            return SerperWebSearch(api_key=settings.serper_api_key)
        except ImportError as e:
            logger.error(f"Failed to import SerperWebSearch: {e}")
            return None
    elif provider_name == "none":
        logger.info("No web search provider configured. Web functionality disabled")
        return None
    else:
        logger.error(f"Unknown web search provider: {provider_name}. Web functionality disabled")
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
    1. Query Extraction -> 2. Search -> 3. Scrape -> 4. Chunk -> 5. Context Assembly
    
    For Tavily: Steps 3-4 are skipped as Tavily provides pre-chunked content
    For other providers: Full pipeline is executed
    
    Args:
        request_body: The original request body as bytes
        query: Optional search query. If None, will extract from user message
        
    Returns:
        Enhanced request body with web context injected as bytes
    """
    try:
        # Get configured providers
        search_provider = get_web_search_provider()
        scraper_provider = get_web_scraper_provider()
        chunker_provider = get_chunker_provider()
        
        if not search_provider:
            logger.warning("No web search provider available, cannot enhance request")
            return request_body
        
        # Step 1: Extract query (use provided query or extract from request)
        extracted_query = query or _extract_query_from_request_body(request_body)
        
        # Check if using Tavily (all-in-one RAG)
        is_tavily = hasattr(search_provider, 'provider_name') and search_provider.provider_name == "Tavily"
        
        if is_tavily:
            logger.info("Using Tavily all-in-one RAG - skipping separate scraping and chunking")
            # Step 2: Perform Tavily search (includes content extraction and chunking)
            search_response = await search_provider.search(extracted_query)
        else:
            # Step 2: Perform web search and scraping for other providers
            search_response = await _perform_web_search_and_scraping(search_provider, scraper_provider, extracted_query)
            
            # Step 3: Chunk the scraped content
            if settings.enable_chunking and chunker_provider and search_response.results:
                await _chunk_search_results(search_response.results, chunker_provider)

        for result in search_response.results:
            print(result.chunks)
        # Step 4: Inject chunked context into request
        return await _inject_web_context_into_request(request_body, search_response, extracted_query)
            
    except Exception as e:
        logger.error(
            f"Failed to enhance request with web context: {e}",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "query": query,
                "search_provider": settings.web_search_provider,
                "scraper_provider": settings.web_scraper_provider,
                "chunker_provider": settings.chunker_provider,
            }
        )
        return request_body


async def _perform_web_search_and_scraping(search_provider, scraper_provider, query: str):
    """Perform web search and scraping, returning search response with scraped content."""
    from .BaseWebSearch import BaseWebSearch
    
    # Create web search instance with configured scraper
    web_search = BaseWebSearch(scraper=scraper_provider)
    
    # Perform search
    search_response = await search_provider.search(query)
    
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


async def _inject_web_context_into_request(request_body: bytes, search_response, query: str = None) -> bytes:
    """Inject web search context into the request body."""
    import json
    
    # Parse the request body
    data = json.loads(request_body)
    
    # Assemble context from chunks or full content
    context_parts = []
    
    # Add AI-generated answer if available
    if search_response.summary and not search_response.summary.startswith("No search results found"):
        context_parts.append(f"AI-Generated Answer: {search_response.summary}")
    
    for result in search_response.results:
        if result.chunks:
            # Use pre-chunked content if available (from Tavily or other providers)
            context_parts.append(
                f"Source: {result.title} ({result.url})\n"
                f"Relevant Content: {result.chunks}..."
            )
        elif result.content:
            # Fallback to full content
            context_parts.append(
                f"Source: {result.title} ({result.url})\n"
                f"Content: {result.content[:500]}..."
            )
    
    web_context = "\n\n---\n\n".join(context_parts) if context_parts else "Web search was performed, but no content could be successfully scraped."
    
    web_message = {
        "role": "system",
        "content": f"Web search context for query '{query}':\n{web_context}"
    }
    
    if "messages" in data:
        data["messages"].insert(0, web_message)
    
    # Convert back to bytes
    enhanced_body = json.dumps(data).encode()
    
    logger.info(
        f"Request enhanced with web search context",
        extra={
            "query": query,
            "original_size": len(request_body),
            "enhanced_size": len(enhanced_body),
            "total_results": len(search_response.results),
        }
    )
    print(enhanced_body)
    
    return enhanced_body


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