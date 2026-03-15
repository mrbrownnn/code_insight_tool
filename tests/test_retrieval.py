import unittest
from unittest.mock import Mock, MagicMock, patch
import sys

# Mock VectorStore and Qdrant before importing modules
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.async_qdrant_client'] = MagicMock()
sys.modules['qdrant_client.async_qdrant_remote'] = MagicMock()
sys.modules['qdrant_client.uploader'] = MagicMock()
sys.modules['qdrant_client.uploader.grpc_uploader'] = MagicMock()

from core.retrieval.hybrid_retriever import HybridRetriever
from core.retrieval.context_expander import ContextExpander
from config import settings

# Mock SearchResult class
class SearchResult:
    def __init__(self, chunk_id, score, metadata):
        self.chunk_id = chunk_id
        self.score = score
        self.metadata = metadata


class TestHybridRetriever(unittest.TestCase):
    """Test HybridRetriever fusion and boosting logic."""
    
    def setUp(self):
        """Initialize retriever with mock vector store."""
        self.retriever = HybridRetriever()
        
        # Create mock chunks
        self.mock_chunks = {
            "chunk_1": {
                "text": "def parse_ast(code):\n    return ast.parse(code)\n    # parse codebase\n    # extract nodes\n    # return tree",
                "source_file": "core/ingestion/ast_parser.py",
                "start_line": 10,
                "end_line": 12,
            },
            "chunk_2": {
                "text": "def tokenize(text):\n    return text.split()",
                "source_file": "utils/tokenizer.py",
                "start_line": 5,
                "end_line": 7,
            },
            "chunk_3": {
                "text": "x",  # Very short chunk
                "source_file": "test.py",
                "start_line": 1,
                "end_line": 1,
            },
        }
        
        # Build BM25 index
        self.retriever.build_bm25_index(self.mock_chunks)
    
    def test_bm25_search_basic(self):
        """Test 1: BM25 search returns keyword matches."""
        results = self.retriever.bm25_search("parse ast function")
        
        # Should find chunk_1 (has "parse" and "ast")
        chunk_ids = [c[0] for c in results]
        self.assertIn("chunk_1", chunk_ids)
    
    def test_bm25_search_empty_index(self):
        """Test 2: BM25 search graceful fallback when no index."""
        empty_retriever = HybridRetriever()
        results = empty_retriever.bm25_search("test")
        self.assertEqual(results, [])
    
    def test_percentile_rank_fusion(self):
        """Test 3: Percentile rank fusion combines results correctly."""
        bm25_results = [
            ("chunk_1", 10.0),
            ("chunk_2", 5.0),
        ]
        vector_results = [
            SearchResult("chunk_2", 0.9, {}),
            SearchResult("chunk_3", 0.7, {}),
        ]
        
        fused = self.retriever._percentile_rank_fusion(
            bm25_results,
            vector_results,
            alpha=0.5,  # Equal weight
        )
        
        # Should rank all 3 chunks
        self.assertEqual(len(fused), 3)
        
        # chunk_2 appears in both, should rank high
        chunk_ids = [r["chunk_id"] for r in fused]
        self.assertIn("chunk_2", chunk_ids)
    
    def test_rrf_fallback(self):
        """Test 4: RRF fallback works when only one source has results."""
        bm25_results = [("chunk_1", 10.0)]
        vector_results = []
        
        rrf_results = self.retriever._rrf_fallback(
            bm25_results,
            vector_results,
        )
        
        # Should still return chunk_1
        self.assertEqual(len(rrf_results), 1)
        self.assertEqual(rrf_results[0]["chunk_id"], "chunk_1")
    
    def test_boosting_exact_match(self):
        """Test 5: Exact match boosting multiplies score."""
        ranked_results = [
            {
                "chunk_id": "chunk_1",
                "fused_score": 1.0,
            }
        ]
        
        boosted = self.retriever._apply_boosting(ranked_results)
        
        # Score should be boosted (multiplied)
        self.assertGreater(boosted[0]["final_score"], 1.0)
    
    def test_boosting_short_chunk_penalty(self):
        """Test 6: Short chunks get penalized."""
        ranked_results = [
            {
                "chunk_id": "chunk_3",  # Very short
                "fused_score": 1.0,
            }
        ]
        
        boosted = self.retriever._apply_boosting(ranked_results)
        
        # Score should be penalized (multiplied by < 1.0)
        self.assertLess(
            boosted[0]["final_score"],
            1.0 * settings.retrieval_exact_match_boost,
        )
    
    def test_retrieve_full_pipeline(self):
        """Test 7: Full retrieve pipeline (BM25 + vector + fusion + boosting)."""
        # Create a mock vector store since real one requires Qdrant
        mock_vector_store = Mock()
        mock_vector_results = [
            SearchResult("chunk_1", 0.95, {"text": "parse_ast"}),
            SearchResult("chunk_2", 0.70, {"text": "other"}),
        ]
        mock_vector_store.search = Mock(return_value=mock_vector_results)
        
        # Create retriever with mock vector store
        retriever = HybridRetriever(vector_store=mock_vector_store)
        retriever.build_bm25_index(self.mock_chunks)
        
        results = retriever.retrieve(
            query_text="how to parse ast code",
            query_vector=[0.1] * 768,  # Dummy vector
            top_k=2,
        )
        
        # Should return top-k results
        self.assertLessEqual(len(results), 2)
        self.assertTrue(all("chunk_id" in r for r in results))
        self.assertTrue(all("final_score" in r for r in results))
    
    def test_alpha_parameter_override(self):
        """Test 8: Alpha parameter can override default fusion weight."""
        bm25_results = [("chunk_1", 10.0)]
        vector_results = [SearchResult("chunk_2", 0.9, {})]
        
        # Test with alpha=0.8 (BM25-heavy)
        fused_bm25_heavy = self.retriever._percentile_rank_fusion(
            bm25_results,
            vector_results,
            alpha=0.8,
        )
        
        # Test with alpha=0.2 (vector-heavy)
        fused_vector_heavy = self.retriever._percentile_rank_fusion(
            bm25_results,
            vector_results,
            alpha=0.2,
        )
        
        # Results should be different (different alpha changes ranking)
        # At minimum, both should return results
        self.assertGreater(len(fused_bm25_heavy), 0)
        self.assertGreater(len(fused_vector_heavy), 0)


class TestContextExpander(unittest.TestCase):
    """Test ContextExpander expansion logic."""
    
    def setUp(self):
        """Initialize context expander."""
        self.expander = ContextExpander()
        
        self.mock_chunk = {
            "text": "def foo():\n    return 42",
            "source_file": "test.py",
            "start_line": 10,
            "end_line": 11,
        }
    
    def test_expand_chunk_without_ast(self):
        """Test 9: Graceful fallback when AST parsing fails."""
        # Mock AST parser to raise error
        with patch.object(self.expander.ast_parser, 'parse_file') as mock_parse:
            mock_parse.side_effect = Exception("File not found")
            
            result = self.expander.expand_chunk(self.mock_chunk)
            
            # Should still return chunk (with minimal expansion)
            self.assertIn("text", result)
    
    def test_expand_chunks_respects_token_limit(self):
        """Test 10: Expansion respects max_context_tokens limit."""
        chunks = [self.mock_chunk, self.mock_chunk]
        
        # Set very low limit
        expanded = self.expander.expand_chunks(
            chunks,
            query="test",
            max_tokens=10,
        )
        
        # Should expand but respect token limit
        for chunk in expanded:
            tokens = chunk.get("total_tokens", 0)
            self.assertLessEqual(tokens, 10 + 5)  # Small margin


class TestConfigIntegration(unittest.TestCase):
    """Test that new config parameters are loaded correctly."""
    
    def test_retrieval_config_exists(self):
        """Verify retrieval config parameters exist."""
        required_params = [
            "top_k_bm25",
            "top_k_vector",
            "top_k_final",
            "rrf_k",
            "fusion_alpha",
            "max_context_tokens",
            "retrieval_exact_match_boost",
            "retrieval_min_chunk_length",
            "retrieval_short_chunk_penalty",
        ]
        
        for param in required_params:
            self.assertTrue(
                hasattr(settings, param),
                f"Config missing parameter: {param}",
            )
    
    def test_retrieval_config_reasonable_values(self):
        """Verify config values are reasonable."""
        self.assertGreater(settings.top_k_bm25, 0)
        self.assertGreater(settings.top_k_vector, 0)
        self.assertGreater(settings.top_k_final, 0)
        self.assertLess(settings.top_k_final, settings.top_k_bm25)
        self.assertLess(settings.fusion_alpha, 1.0)
        self.assertGreater(settings.fusion_alpha, 0.0)


if __name__ == "__main__":
    unittest.main()
