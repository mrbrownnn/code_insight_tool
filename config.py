"""
Code Insight Tool — Configuration Management
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    #LangChain settings
    llm_model: str = "qwen2.5-coder"
    llm_temperature: float = 0.25
    rate_limit: int = 10  # Max requests per second to Ollama
    memory_settings: dict = {
        "k": 5,  # Number of past interactions to remember
    }
    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "code_chunks"

    # --- Ollama (Self-hosted LLM) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder"

    # --- Embedding ---
    embedding_model: str = "microsoft/unixcoder-base"
    embedding_dim: int = 768

    # --- Chunking ---
    chunk_max_tokens: int = 2048
    chunk_overlap_lines: int = 30
    chunk_fallback_window: int = 100

    # --- Supported Languages ---
    supported_languages: List[str] = ["python", "javascript", "java"]
    supported_extensions: dict = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "javascript",
        ".tsx": "javascript",
        ".java": "java",
    }

    # --- Ingestion ---
    max_file_size_bytes: int = 1_048_576  # 1MB
    batch_size: int = 32
    clone_dir: Path = Path("./tmp/repos")

    # --- Retrieval (Hybrid Search) ---
    # RRF & Fusion parameters
    top_k_bm25: int = 20  # Fetch many before fusion
    top_k_vector: int = 20  # Fetch many before fusion
    top_k_final: int = 5  # Final results after fusion
    rrf_k: int = 60  # RRF constant (default 60)
    fusion_alpha: float = 0.5  # Percentile fusion: 0.5=equal BM25/vector, 0-1 range
    
    # Context expansion
    max_context_tokens: int = 4000  # Hard limit after expansion
    
    # Boosting parameters
    retrieval_exact_match_boost: float = 2.0  # Scalar multiplier for exact matches
    retrieval_min_chunk_length: int = 5  # Min lines, below triggers penalty
    retrieval_short_chunk_penalty: float = 0.5  # Multiplier for short chunks

    # --- Memory & History ---
    memory_k: int = 5  # Number of past interactions to remember

    # --- API Keys & External Services ---
    api_key: str = "your_api_key_here"

    # --- Logging ---
    log_level: str = "INFO"

    # --- App ---
    app_name: str = "Code Insight Tool"
    app_version: str = "0.1.0"


# Singleton instance
settings = Settings()
