"""
Base chunker module for text content granulation.

This module provides the abstract base class for text chunking algorithms,
following the same modular pattern as the web search and scraping components.
"""

from abc import ABC, abstractmethod
from typing import List

from ..core.logging import get_logger

logger = get_logger(__name__)


class BaseWebChunker(ABC):
    """Base class for content chunkers."""

    chunker_name: str = "base"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Initialized {self.chunker_name} chunker with size={chunk_size}, overlap={chunk_overlap}"
        )

    @abstractmethod
    async def chunk_text(self, text: str, **kwargs) -> List[str]:
        """
        Chunk text into smaller pieces and return list of strings.

        Args:
            text: The text to chunk
            **kwargs: Additional parameters for chunking algorithms

        Returns:
            List of text chunks as strings
        """
        raise NotImplementedError("Subclasses must implement scrape_url method")

    def rank_chunks(self, chunks: List[str], query: str) -> List[str]:
        """
        Rank chunks by relevance to the query.

        Args:
            chunks: List of text chunks to rank
            query: Search query for relevance scoring (optional)

        Returns:
            List of chunks ranked by relevance (currently returns chunks as-is)
        """
        # TODO: Implement relevance scoring based on query
        # For now, return chunks in original order
        return chunks

    def chunk_texts_batch(self, texts: List[str], **kwargs) -> List[List[str]]:
        """
        Chunk multiple texts (synchronous version for simplicity).

        Args:
            texts: List of texts to chunk
            **kwargs: Additional parameters for chunking algorithms

        Returns:
            List of lists containing text chunks for each input text
        """
        logger.debug(f"Batch chunking {len(texts)} texts with {self.chunker_name}")
        return [self.chunk_text(text, **kwargs) for text in texts]

    def validate_parameters(self) -> bool:
        """
        Validate chunker parameters.

        Returns:
            True if parameters are valid, False otherwise
        """
        if self.chunk_size <= 0:
            logger.error(f"Invalid chunk_size: {self.chunk_size}. Must be > 0")
            return False

        if self.chunk_overlap < 0:
            logger.error(f"Invalid chunk_overlap: {self.chunk_overlap}. Must be >= 0")
            return False

        if self.chunk_overlap >= self.chunk_size:
            logger.error(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
            return False

        return True

    async def chunk_search_results(self, search_result: "SearchResult", query: str) -> "SearchResult":
        """
        Chunk the content in search results using the configured chunking strategy.
        
        Args:
            search_result: SearchResult object with content to chunk
            query: Original query for relevance ranking
            
        Returns:
            SearchResult with chunked content populated
        """
        logger.info(f"Chunking content for {len(search_result.results)} search results")
        
        for result in search_result.results:
            if result.content:
                chunks = await self.chunk_text(result.content)
                # Rank chunks by relevance and limit chunks per source
                ranked_chunks = self.rank_chunks(chunks, query)
                ranked_chunks = ranked_chunks[:5]  # Default limit, could be configurable
                result.relevant_chunks = " [...] ".join(ranked_chunks)
                logger.debug(f"Selected {len(ranked_chunks)} chunks for {result.url}")
                
        return search_result
