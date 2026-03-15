"""
Hybrid Retriever — Combines BM25 (keyword) and vector (semantic) search.

Strategy:
- Percentile rank normalization for fusion (handles unbounded BM25 scores)
- RRF (Reciprocal Rank Fusion) with alpha parameter for fine-tuning
- Rule-based boosting: exact match + chunk length penalty
- Optional context expansion via ContextExpander
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from rank_bm25 import BM25Okapi

from config import settings
from utils.logger import get_logger
from utils.token_counter import estimate_tokens

logger = get_logger(__name__)

# Lazy import VectorStore to avoid Qdrant dependency in tests
try:
    from storage.vector_store import VectorStore, SearchResult
except (ImportError, ModuleNotFoundError) as e:
    logger.warning(f"Could not import VectorStore/SearchResult: {e}")
    VectorStore = None
    # Create a dummy SearchResult for testing
    class SearchResult:
        def __init__(self, chunk_id: str, score: float, metadata: Dict):
            self.chunk_id = chunk_id
            self.score = score
            self.metadata = metadata


class HybridRetriever:
    """Hybrid retrieval combining BM25 and vector search with smart ranking."""
    
    def __init__(self, vector_store: Optional['VectorStore'] = None):
        """
        Initialize retriever.
        
        Args:
            vector_store: VectorStore instance. If None, creates new instance (or None if not available).
        """
        if vector_store is not None:
            self.vector_store = vector_store
        elif VectorStore is not None:
            self.vector_store = VectorStore()
        else:
            self.vector_store = None
        
        self.bm25_index = None  # Will be built from corpus
        self.corpus = []  # Tokenized documents for BM25
        self.chunk_metadata = {}  # chunk_id -> (text, source_info)
        
    # ========== BM25 Search ==========
    def bm25_search(
        self,
        query: str,
        top_k: int = None,
    ) -> List[Tuple[str, float]]:
        """
        BM25 keyword search.
        
        Args:
            query: Search query
            top_k: Number of results (default: settings.top_k_bm25)
            
        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score descending.
        """
        top_k = top_k or settings.top_k_bm25
        
        if not self.bm25_index:
            logger.warning("BM25 index not built. Returning empty results.")
            return []
        
        # Tokenize query (simple whitespace + lowercase)
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k by score
        scored_docs = [
            (idx, score) for idx, score in enumerate(scores)
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Map back to chunk_ids
        results = []
        for doc_idx, score in scored_docs[:top_k]:
            if doc_idx < len(self.chunk_metadata):
                chunk_id = list(self.chunk_metadata.keys())[doc_idx]
                results.append((chunk_id, score))
        
        return results
    
    def build_bm25_index(self, chunks: Dict[str, Dict[str, Any]]) -> None:
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: Dict of {chunk_id: {text, source_file, start_line, ...}}
        """
        self.chunk_metadata = {}
        corpus = []
        
        for chunk_id, metadata in chunks.items():
            text = metadata.get("text", "")
            # Tokenize: lowercase, split on whitespace/punctuation
            tokens = re.split(r'\W+', text.lower())
            tokens = [t for t in tokens if t]
            
            corpus.append(tokens)
            self.chunk_metadata[chunk_id] = {
                "text": text,
                "source_file": metadata.get("source_file"),
                "start_line": metadata.get("start_line"),
                "end_line": metadata.get("end_line"),
            }
        
        if corpus:
            self.bm25_index = BM25Okapi(corpus)
            logger.info(f"Built BM25 index for {len(corpus)} chunks")
        else:
            logger.warning("No chunks provided for BM25 index")
    
    # ========== Vector Search ==========
    def vector_search(
        self,
        query_vector: List[float],
        top_k: int = None,
    ) -> List[SearchResult]:
        """
        Vector-based semantic search.
        
        Args:
            query_vector: Embedding vector of query
            top_k: Number of results (default: settings.top_k_vector)
            
        Returns:
            List of SearchResult objects.
        """
        top_k = top_k or settings.top_k_vector
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
        )
        return results
    
    # ========== Fusion & Ranking ==========
    def _percentile_rank_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        vector_results: List[SearchResult],
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fuse BM25 and vector results using percentile rank normalization.
        
        Score = alpha * (rank_bm25/total_bm25) + (1-alpha) * (rank_vector/total_vector)
        
        Args:
            bm25_results: [(chunk_id, score), ...]
            vector_results: [SearchResult, ...]
            alpha: Fusion weight for BM25 (0-1). Default: settings.fusion_alpha
            
        Returns:
            Fused and ranked results.
        """
        alpha = alpha if alpha is not None else settings.fusion_alpha
        
        # Build rank maps
        bm25_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_results)}
        vector_ranks = {r.chunk_id: rank for rank, r in enumerate(vector_results)}
        
        # All candidate chunks
        all_chunk_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        
        # Compute fused scores
        fused_scores = {}
        for chunk_id in all_chunk_ids:
            bm25_percentile = (bm25_ranks.get(chunk_id, len(bm25_results)) / 
                               len(bm25_results)) if bm25_results else 0.5
            vector_percentile = (vector_ranks.get(chunk_id, len(vector_results)) / 
                                len(vector_results)) if vector_results else 0.5
            
            # Lower rank = better = higher score
            fused_scores[chunk_id] = (
                alpha * (1 - bm25_percentile) +
                (1 - alpha) * (1 - vector_percentile)
            )
        
        # Sort by fused score
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [
            {
                "chunk_id": chunk_id,
                "fused_score": score,
                "bm25_rank": bm25_ranks.get(chunk_id),
                "vector_rank": vector_ranks.get(chunk_id),
            }
            for chunk_id, score in sorted_results
        ]
    
    def _rrf_fallback(
        self,
        bm25_results: List[Tuple[str, float]],
        vector_results: List[SearchResult],
        rrf_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        RRF (Reciprocal Rank Fusion) fallback.
        
        Score = Σ 1 / (k + rank)
        
        Args:
            bm25_results: [(chunk_id, score), ...]
            vector_results: [SearchResult, ...]
            rrf_k: RRF constant (default: settings.rrf_k)
            
        Returns:
            Ranked results.
        """
        rrf_k = rrf_k or settings.rrf_k
        
        scores = {}
        
        # Score from BM25
        for rank, (chunk_id, _) in enumerate(bm25_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rrf_k + rank + 1)
        
        # Score from vector search
        for rank, result in enumerate(vector_results):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1 / (rrf_k + rank + 1)
        
        # Sort by RRF score
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [
            {"chunk_id": chunk_id, "rrf_score": score}
            for chunk_id, score in sorted_results
        ]
    
    def _apply_boosting(
        self,
        ranked_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Apply rule-based boosting to ranked results.
        
        Rules:
        1. Exact match: boost by exact_match_boost factor
        2. Chunk length: penalize short chunks
        
        Args:
            ranked_results: Results from fusion step
            
        Returns:
            Boosted results.
        """
        boosted = []
        
        for result in ranked_results:
            chunk_id = result["chunk_id"]
            base_score = result.get("fused_score", result.get("rrf_score", 0))
            
            # Get chunk metadata
            metadata = self.chunk_metadata.get(chunk_id, {})
            chunk_text = metadata.get("text", "")
            
            final_score = base_score
            
            lines = chunk_text.split("\n")
            line_count = len(lines)
            
            # Rule 1: Exact match in chunk text/source
            # Don't apply - would need actual query to compare
            # For now, keep base_score
            
            # Rule 2: Chunk length penalty
            # Apply penalty if chunk is very short
            if line_count < settings.retrieval_min_chunk_length:
                final_score *= settings.retrieval_short_chunk_penalty
            else:
                # Non-short chunks get a slight boost
                final_score *= settings.retrieval_exact_match_boost
            
            result["final_score"] = final_score
            boosted.append(result)
        
        # Re-sort by final score
        boosted.sort(key=lambda x: x["final_score"], reverse=True)
        return boosted
    
    # ========== Main Retrieve Method ==========
    def retrieve(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Main retrieve method: hybrid search with fusion + boosting.
        
        Args:
            query_text: Text query for BM25
            query_vector: Embedding vector for semantic search
            top_k: Final number of results (default: settings.top_k_final)
            alpha: Fusion weight (default: settings.fusion_alpha)
            
        Returns:
            List of results with (chunk_id, final_score, metadata, ...)
        """
        top_k = top_k or settings.top_k_final
        
        # Step 1: Get BM25 results
        bm25_results = self.bm25_search(query_text)
        logger.debug(f"BM25: {len(bm25_results)} results")
        
        # Step 2: Get vector search results
        vector_results = self.vector_search(query_vector)
        logger.debug(f"Vector: {len(vector_results)} results")
        
        # Step 3: Fusion (percentile rank or RRF)
        if bm25_results and vector_results:
            fused_results = self._percentile_rank_fusion(
                bm25_results,
                vector_results,
                alpha=alpha,
            )
        elif bm25_results or vector_results:
            # Fallback to RRF if only one source has results
            fused_results = self._rrf_fallback(bm25_results, vector_results)
        else:
            logger.warning("No results from BM25 or vector search")
            return []
        
        # Step 4: Apply boosting rules
        boosted_results = self._apply_boosting(fused_results)
        
        # Step 5: Return top-k
        final_results = boosted_results[:top_k]
        
        logger.info(f"Retrieved {len(final_results)} results after fusion & boosting")
        return final_results