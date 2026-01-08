"""
Base chunker module for text content granulation.

This module provides the abstract base class for text chunking algorithms,
following the same modular pattern as the web search and scraping components.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List

from rank_bm25 import BM25Okapi

from ..core.logging import get_logger
from ..core.settings import settings
from .types import SearchResult, WebPageContent


logger = get_logger(__name__)


class BaseWebChunker(ABC):
    """Base class for content chunkers."""

    chunker_name: str = "base"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0) -> None:
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
    async def chunk_text(self, text: str) -> List[str]:
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
        print("top 5 chunks pre-ranking:")
        for chunk in chunks[:5]:
            print(chunk)
            
        if not chunks or not query:
            return chunks
        
        corpus_words = [chunk.split() for chunk in chunks]
        query_words = query.split()

        bm25 = BM25Okapi(corpus_words)

        ranked_chunks = bm25.get_top_n(query_words, chunks, n=len(chunks))


        print(f"top 5 chunks post-ranking with query {query}:")
        for chunk in ranked_chunks[:5]:
            print(chunk)
        return ranked_chunks

    async def chunk_search_results(
        self, search_result: SearchResult, query: str
    ) -> SearchResult:
        """
        Chunk the content in search results concurrently.
        """
        logger.info(
            f"Chunking content for {len(search_result.results)} search results concurrently"
        )

        async def process_and_update(result: WebPageContent) -> None:
            if not result.content:
                return
            try:
                chunks = await self.chunk_text(result.content)
                ranked_chunks = self.rank_chunks(chunks, query)
                ranked_chunks = ranked_chunks[: settings.chunk_max_chunks_per_source]
                result.relevant_chunks = " [...] ".join(ranked_chunks)
                logger.debug(
                    f"Selected {len(ranked_chunks)}/{len(chunks)} chunks for {result.url}. "
                    f"{sum(len(s) for s in ranked_chunks)}/{sum(len(s) for s in chunks)} characters included."
                )
            except Exception as e:
                logger.error(f"Failed to chunk content for {result.url}: {e}")
                result.relevant_chunks = None

        tasks = [
            process_and_update(result)
            for result in search_result.results
            if result.content
        ]

        # 3. Run all tasks concurrently and wait for them to complete.
        if tasks:
            await asyncio.gather(*tasks)

        return search_result

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
