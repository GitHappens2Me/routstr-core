"""
Fixed-size text chunker implementation.

This module provides a simple fixed-size character chunker that splits text
into chunks of a specified size with optional overlap. This is the simplest
chunking approach and serves as a reliable fallback.
"""

import asyncio
from typing import List

from ..core.logging import get_logger
from .BaseChunker import BaseChunker

logger = get_logger(__name__)


class FixedSizeChunker(BaseChunker):
    """Simple fixed-size character chunker."""

    chunker_name = "fixed"

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 0):
        """
        Initialize the fixed-size chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        super().__init__(chunk_size, chunk_overlap)

        if not self.validate_parameters():
            raise ValueError(f"Invalid parameters for {self.chunker_name} chunker")

    async def chunk_text(self, text: str, **kwargs) -> List[str]:
        """
        Chunk text into fixed-size pieces using a sliding window approach.

        Args:
            text: The text to chunk
            **kwargs: Additional parameters (ignored for fixed-size chunking)

        Returns:
            List of text chunks as strings
        """
        if not text or not text.strip():
            logger.debug("Empty text provided, returning empty chunk list")
            return []

        text_length = len(text)

        # If text is shorter than chunk_size, return it as a single chunk
        if text_length <= self.chunk_size:
            logger.debug(
                f"Text length ({text_length}) <= chunk_size ({self.chunk_size}), returning single chunk"
            )
            return [text]

        chunks: list[str] = []
        start = 0

        while start < text_length:
            end = start + self.chunk_size

            # Don't create tiny chunks at the end
            if end > text_length and len(chunks) > 0:
                # If we have chunks already and the remaining text is very small,
                # append it to the last chunk instead of creating a new one
                remaining_text = text[start:]
                if (
                    len(remaining_text) < self.chunk_size * 0.3
                ):  # Less than 30% of chunk size
                    chunks[-1] += remaining_text
                    logger.debug(
                        f"Appended remaining {len(remaining_text)} chars to last chunk"
                    )
                    break

            # Create chunk
            chunk = text[start:end]
            chunks.append(chunk)

            # Calculate next start position with overlap
            start = end - self.chunk_overlap

            # Prevent infinite loop if overlap >= chunk_size
            if start <= 0:
                start = self.chunk_size

        logger.debug(f"Created {len(chunks)} chunks from {text_length} characters")
        return chunks

    def chunk_texts_batch(self, texts: List[str], **kwargs) -> List[List[str]]:
        """
        Chunk multiple texts concurrently.

        Args:
            texts: List of texts to chunk
            **kwargs: Additional parameters (ignored for fixed-size chunking)

        Returns:
            List of lists containing text chunks for each input text
        """
        logger.debug(f"Batch chunking {len(texts)} texts with {self.chunker_name}")

        # For fixed-size chunking, we can process texts in parallel
        async def process_text(text: str) -> List[str]:
            return await self.chunk_text(text, **kwargs)

        # Run all chunking operations concurrently
        tasks = [process_text(text) for text in texts]
        return asyncio.run(asyncio.gather(*tasks, return_exceptions=True))
