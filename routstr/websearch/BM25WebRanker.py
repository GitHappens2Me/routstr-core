import string
from dataclasses import replace
from typing import List, Set, Dict
from ..core.logging import get_logger
from ..core.settings import settings
from .BaseWebRanker import BaseWebRanker
from .types import SearchResult, WebPageContent

logger = get_logger(__name__)

# --- Library Check ---
RANK_BM25_AVAILABLE = False
try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    logger.warning("rank_bm25 not installed. BM25 ranking will be unavailable.")
    BM25Okapi = None

class BM25WebRanker(BaseWebRanker):
    """
    BM25-based ranker that performs local-to-global pruning.
    1. Local: Keep the best N chunks from each specific source.
    2. Global: Keep the best M chunks from the entire pool.
    """
    def __init__(self):
        super().__init__(provider_name="bm25")
        self.local_k = getattr(settings, "ranker_local_top_k", 10)
        self.global_k = getattr(settings, "ranker_global_top_k", 20)

    async def check_availability(self) -> bool:
        """Returns True if rank_bm25 is installed and available."""
        return RANK_BM25_AVAILABLE

    async def rank(self, search_result: SearchResult, query: str) -> SearchResult:
        """Orchestrates the 2-step ranking process and prints a funnel report."""
        if not RANK_BM25_AVAILABLE or not search_result.results:
            return search_result

        logger.info(f"Ranking results for query: '{query}' (Local K: {self.local_k}, Global K: {self.global_k})")

        # 1. Statistics Tracking (Pre-ranking)
        stats: Dict[str, Dict[str, int]] = {}
        for p in search_result.results:
            stats[p.url] = {"initial": len(p.relevant_chunks or [])}

        # 2. Step 1: Local Pruning
        local_result = self._rank_local(search_result, query, top_k=self.local_k)
        for p in local_result.results:
            stats[p.url]["after_local"] = len(p.relevant_chunks or [])

        # 3. Step 2: Global Pruning
        final_result = self._rank_global(local_result, query, total_k=self.global_k)
        for p in final_result.results:
            stats[p.url]["final"] = len(p.relevant_chunks or [])

        # 4. Final Reporting (Switched to 'Retained' metric)
        self._print_report(query, stats)

        return final_result

    def _rank_local(self, search_result: SearchResult, query: str, top_k: int) -> SearchResult:
        """Sorts and prunes chunks within each individual page."""
        updated_pages = []
        for page in search_result.results:
            if not page.relevant_chunks:
                updated_pages.append(page)
                continue

            sorted_chunks = self._sort_chunks(query, page.relevant_chunks)
            updated_pages.append(replace(page, relevant_chunks=sorted_chunks[:top_k]))
        
        return replace(search_result, results=updated_pages)

    def _rank_global(self, search_result: SearchResult, query: str, total_k: int) -> SearchResult:
        """Pools all chunks from all pages and picks the top global winners."""
        all_candidates = []
        for page in search_result.results:
            if page.relevant_chunks:
                all_candidates.extend(page.relevant_chunks)

        if not all_candidates:
            return search_result

        global_winners = self._sort_chunks(query, all_candidates)[:total_k]
        winners_set: Set[str] = set(global_winners)

        final_pages = []
        for page in search_result.results:
            if not page.relevant_chunks:
                final_pages.append(page)
                continue
            
            surviving_chunks = [c for c in page.relevant_chunks if c in winners_set]
            final_pages.append(replace(page, relevant_chunks=surviving_chunks))

        return replace(search_result, results=final_pages)

    def _sort_chunks(self, query: str, chunks: List[str]) -> List[str]:
        """
        The Core Ranking Engine.
        Uses the BM25 (Best Matching 25) algorithm to calculate relevance.
        """
        if not chunks: return []

        def tokenize(text):
            # Lowercase and remove punctuation
            return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

        tokenized_corpus = [tokenize(c) for c in chunks]
        tokenized_query = tokenize(query)
        
        bm25 = BM25Okapi(tokenized_corpus)
        # n=len(chunks) returns every chunk in the list, but sorted by score
        return bm25.get_top_n(tokenized_query, chunks, n=len(chunks))

    def _print_report(self, query: str, stats: Dict[str, Dict[str, int]]) -> None:
        """Prints a funnel report showing chunk retention."""
        print("\n" + "="*95)
        print(f"RANKING REPORT | Query: '{query}'")
        print(f"{'Source URL':<55} | {'Start':<6} | {'L-Keep':<7} | {'FINAL'}")
        print("-" * 95)
        
        total_start = total_l_keep = total_final = 0
        
        for url, data in stats.items():
            initial, l_keep, final = data["initial"], data["after_local"], data["final"]
            total_start += initial
            total_l_keep += l_keep
            total_final += final
            
            display_url = (url[:52] + '...') if len(url) > 55 else url
            print(f"{display_url:<55} | {initial:<6} | {l_keep:<7} | {final}")
            
        print("-" * 95)
        print(f"{'TOTALS':<55} | {total_start:<6} | {total_l_keep:<7} | {total_final}")
        print("="*95 + "\n")