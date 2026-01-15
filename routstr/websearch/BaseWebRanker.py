from abc import ABC, abstractmethod

from .types import SearchResult


class BaseWebRanker(ABC):
    """Base class for search result ranking and pruning."""

    def __init__(self, provider_name: str = "base"):
        self.provider_name = provider_name

    @abstractmethod
    async def rank(self, search_result: SearchResult, query: str) -> SearchResult:
        """
        Rank and prune chunks within a SearchResult.
        This is the primary entry point for the ranking component.
        """
        pass

    @abstractmethod
    async def check_availability(self) -> bool:
        """Verify the ranker is functional."""
        pass
