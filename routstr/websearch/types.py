from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WebPageContent:
    """Content retrieved from a single URL.
    
    Represents processed web content including metadata, full text, and
    relevance-ranked chunks ready for AI context injection.
    """

    title: str
    url: str
    summary: Optional[str] = None
    publication_date: Optional[str] = None
    relevance_score: Optional[float] = None
    content: Optional[str] = None  # Complete webpage content
    relevant_chunks: Optional[str] = (
        None  # Relevant chunks combined into one LLM-readable string.
    )

@dataclass
class SearchResult:
    """Complete RAG search result containing processed web content and metadata.
    
    Contains the list of processed WebPageContent objects along with search
    metadata like timing, result count, and optional AI-generated summaries.
    """

    query: str
    results: List[WebPageContent]  # TODO:Rename to pages?
    summary: Optional[str] = None
    total_results: Optional[int] = None
    timestamp: Optional[str] = None
    search_time_ms: Optional[int] = None

