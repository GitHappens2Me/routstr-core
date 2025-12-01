"""
Factory module for creating web search and scraper instances based on configuration.
"""

from typing import Optional
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
    
    if provider_name == "serper":
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


async def enhance_request_with_web_context(request_body: bytes, query: str = None) -> bytes:
    """
    Enhance AI request with web search context using configured providers.
    
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
        
        if not search_provider:
            logger.warning("No web search provider available, cannot enhance request")
            return request_body
        
        # Create web search instance with configured scraper
        from .BaseWebSearch import BaseWebSearch
        web_search = BaseWebSearch(scraper=scraper_provider)
        
        # Use the search provider's method
        if hasattr(search_provider, 'inject_web_context'):
            return await search_provider.inject_web_context(request_body, query)
        else:
            # Fallback to BaseWebSearch method
            return await web_search.inject_web_context(request_body, query)
            
    except Exception as e:
        logger.error(
            f"Failed to enhance request with web context: {e}",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "query": query,
                "search_provider": settings.web_search_provider,
                "scraper_provider": settings.web_scraper_provider,
            }
        )
        return request_body