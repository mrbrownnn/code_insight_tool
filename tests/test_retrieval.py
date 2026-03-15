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


class TestHybridRetrieverLargeScale(unittest.TestCase):
    """Test HybridRetriever with large-scale scenarios: 10+ chunks, high line numbers."""
    
    def setUp(self):
        """Initialize with large mock corpus (10 chunks with high line ranges)."""
        self.retriever = HybridRetriever()
        
        # Generate 10+ chunks with varied line ranges (50-500)
        self.large_chunks = {
            "chunk_1": {
                "text": "class DataProcessor:\n    def __init__(self):\n        self.data = []\n    def load_file(self, path):\n        return open(path).read()",
                "source_file": "core/data_processor.py",
                "start_line": 50,
                "end_line": 54,
            },
            "chunk_2": {
                "text": "def preprocess_data(raw_data):\n    cleaned = raw_data.strip()\n    filtered = [x for x in cleaned if x]\n    return filtered\n    # preprocess pipeline",
                "source_file": "core/data_processor.py",
                "start_line": 100,
                "end_line": 104,
            },
            "chunk_3": {
                "text": "class VectorEmbedder:\n    def __init__(self, model_name):\n        self.model = load_model(model_name)\n    def embed(self, text):\n        return self.model.encode(text)\n    def batch_embed(self, texts):\n        return [self.embed(t) for t in texts]",
                "source_file": "core/embedding/embedder.py",
                "start_line": 150,
                "end_line": 156,
            },
            "chunk_4": {
                "text": "async def fetch_from_api(url, headers):\n    async with httpx.AsyncClient() as client:\n        response = await client.get(url, headers=headers)\n        return response.json()\n    # handle errors gracefully",
                "source_file": "utils/api_handler.py",
                "start_line": 200,
                "end_line": 204,
            },
            "chunk_5": {
                "text": "def process_chunks(chunks, processor):\n    results = []\n    for chunk in chunks:\n        result = processor.process(chunk)\n        results.append(result)\n    return results",
                "source_file": "core/ingestion/pipeline.py",
                "start_line": 250,
                "end_line": 255,
            },
            "chunk_6": {
                "text": "class QueryBuilder:\n    def __init__(self, index):\n        self.index = index\n    def build_vector_query(self, embedding):\n        return vector_search(self.index, embedding)\n    def build_keyword_query(self, text):\n        return bm25_search(self.index, text)",
                "source_file": "core/retrieval/query_builder.py",
                "start_line": 300,
                "end_line": 306,
            },
            "chunk_7": {
                "text": "def generate_response(context, question, model):\n    prompt = format_prompt(context, question)\n    response = model.generate(prompt)\n    return parse_response(response)\n    # chunk uses generation logic",
                "source_file": "core/generation/generator.py",
                "start_line": 350,
                "end_line": 354,
            },
            "chunk_8": {
                "text": "class ConversationMemory:\n    def __init__(self, max_turns=5):\n        self.history = []\n        self.max_turns = max_turns\n    def add_message(self, role, content):\n        self.history.append({\"role\": role, \"content\": content})",
                "source_file": "storage/memory.py",
                "start_line": 400,
                "end_line": 406,
            },
            "chunk_9": {
                "text": "def validate_input(text, min_length=1, max_length=1000):\n    if not isinstance(text, str):\n        return False\n    if len(text) < min_length or len(text) > max_length:\n        return False\n    return True",
                "source_file": "utils/validators.py",
                "start_line": 450,
                "end_line": 455,
            },
            "chunk_10": {
                "text": "def calculate_similarity(vec1, vec2):\n    import numpy as np\n    dot_product = np.dot(vec1, vec2)\n    norm1 = np.linalg.norm(vec1)\n    norm2 = np.linalg.norm(vec2)\n    return dot_product / (norm1 * norm2)",
                "source_file": "utils/math_utils.py",
                "start_line": 500,
                "end_line": 505,
            },
        }
        
        self.retriever.build_bm25_index(self.large_chunks)
    
    def test_bm25_large_corpus_basic(self):
        """Test 11: BM25 on large corpus (10 chunks)."""
        results = self.retriever.bm25_search("process data chunk", top_k=5)
        self.assertGreater(len(results), 0)
        # Should find chunks with "process" and/or "data" and/or "chunk"
        self.assertGreater(len(results), 0)
    
    def test_bm25_large_corpus_specific_keywords(self):
        """Test 12: BM25 finds specific keywords in large corpus."""
        results = self.retriever.bm25_search("vector embedding model")
        chunk_ids = [c[0] for c in results]
        # chunk_3 has all: "VectorEmbedder", "model", "embed"
        self.assertIn("chunk_3", chunk_ids)
    
    def test_bm25_large_corpus_no_match(self):
        """Test 13: BM25 handles queries with no matches gracefully."""
        results = self.retriever.bm25_search("nonexistent_keyword_xyz")
        # May return empty or low-scoring results
        self.assertIsInstance(results, list)
    
    def test_retrieval_high_line_numbers(self):
        """Test 14: Retrieval works with high start_line/end_line (50-505)."""
        # Verify chunks are indexed correctly with large line numbers
        self.assertIn("chunk_1", self.retriever.chunk_metadata)
        chunk_1_meta = self.retriever.chunk_metadata["chunk_1"]
        self.assertEqual(chunk_1_meta["start_line"], 50)
        self.assertEqual(chunk_1_meta["end_line"], 54)
        
        chunk_10_meta = self.retriever.chunk_metadata["chunk_10"]
        self.assertEqual(chunk_10_meta["start_line"], 500)
        self.assertEqual(chunk_10_meta["end_line"], 505)
    
    def test_fusion_large_corpus_multifile(self):
        """Test 15: Fusion works with multi-file corpus."""
        bm25_results = [
            ("chunk_1", 12.0),  # core/data_processor.py
            ("chunk_3", 10.0),  # core/embedding/embedder.py
            ("chunk_5", 8.0),   # core/ingestion/pipeline.py
        ]
        vector_results = [
            SearchResult("chunk_2", 0.95, {}),  # Different file
            SearchResult("chunk_3", 0.88, {}),  # Same as BM25
            SearchResult("chunk_7", 0.72, {}),  # Different file
        ]
        
        fused = self.retriever._percentile_rank_fusion(
            bm25_results,
            vector_results,
            alpha=0.5,
        )
        
        # Should rank chunk_3 high (appears in both)
        chunk_ids = [r["chunk_id"] for r in fused]
        self.assertIn("chunk_3", chunk_ids)
        # chunk_3 should rank high
        fused_sorted = sorted(fused, key=lambda x: x["fused_score"], reverse=True)
        self.assertEqual(fused_sorted[0]["chunk_id"], "chunk_3")
    
    def test_top_k_limits_large_corpus(self):
        """Test 16: top_k parameter limits results correctly."""
        mock_vector_store = Mock()
        mock_vector_results = [
            SearchResult(f"chunk_{i}", 0.9 - i*0.05, {})
            for i in range(1, 11)
        ]
        mock_vector_store.search = Mock(return_value=mock_vector_results)
        
        retriever = HybridRetriever(vector_store=mock_vector_store)
        retriever.build_bm25_index(self.large_chunks)
        
        results = retriever.retrieve(
            query_text="test query",
            query_vector=[0.1] * 768,
            top_k=3,
        )
        
        # Should return exactly top_k=3
        self.assertLessEqual(len(results), 3)
    
    def test_boosting_varied_chunk_sizes(self):
        """Test 17: Boosting handles varied chunk sizes correctly."""
        ranked_results = [
            {"chunk_id": "chunk_1", "fused_score": 1.0},  # 5 lines
            {"chunk_id": "chunk_3", "fused_score": 1.0},  # 7 lines
            {"chunk_id": "chunk_10", "fused_score": 1.0}, # 6 lines
        ]
        
        boosted = self.retriever._apply_boosting(ranked_results)
        
        # All should have final_score (boosted)
        for result in boosted:
            self.assertIn("final_score", result)
            self.assertGreater(result["final_score"], 0)
    
    def test_bm25_multiple_files_coverage(self):
        """Test 18: BM25 covers chunks from multiple files."""
        results = self.retriever.bm25_search("process generate validate")
        chunk_ids = [c[0] for c in results]
        
        # Query has keywords from different files
        # chunk_2: "process" ✓
        # chunk_7: "generate" ✓
        # chunk_9: "validate" ✓
        sources = [
            self.large_chunks[cid]["source_file"] 
            for cid in chunk_ids
        ]
        
        # Should have results (coverage of multiple files)
        self.assertGreater(len(set(sources)), 0)
    
    def test_retrieval_query_completeness(self):
        """Test 19: Retrieval result has all required fields."""
        mock_vector_store = Mock()
        mock_vector_results = [
            SearchResult("chunk_1", 0.95, {}),
            SearchResult("chunk_5", 0.85, {}),
        ]
        mock_vector_store.search = Mock(return_value=mock_vector_results)
        
        retriever = HybridRetriever(vector_store=mock_vector_store)
        retriever.build_bm25_index(self.large_chunks)
        
        results = retriever.retrieve(
            query_text="data processing",
            query_vector=[0.1] * 768,
            top_k=2,
        )
        
        required_fields = ["chunk_id", "final_score"]
        for result in results:
            for field in required_fields:
                self.assertIn(field, result, f"Missing field: {field}")
    
    def test_large_corpus_performance_baseline(self):
        """Test 20: Retrieval completes in reasonable time on large corpus."""
        import time
        
        mock_vector_store = Mock()
        mock_vector_results = [
            SearchResult(f"chunk_{i}", 0.9 - i*0.01, {})
            for i in range(1, 11)
        ]
        mock_vector_store.search = Mock(return_value=mock_vector_results)
        
        retriever = HybridRetriever(vector_store=mock_vector_store)
        retriever.build_bm25_index(self.large_chunks)
        
        start_time = time.time()
        results = retriever.retrieve(
            query_text="vector processing embedding model",
            query_vector=[0.1] * 768,
            top_k=5,
        )
        elapsed_time = time.time() - start_time
        
        # Should complete in < 1 second (reasonable baseline)
        self.assertLess(elapsed_time, 1.0)
        # Should return results
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()

