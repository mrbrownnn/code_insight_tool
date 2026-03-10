"""
Code Insight Tool — Configuration Management
Sử dụng pydantic-settings để load config từ .env file.
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
        ".ts": "typescript",
    }

    # --- Ingestion ---
    max_file_size_bytes: int = 1_048_576  # 1MB
    batch_size: int = 32
    clone_dir: Path = Path("./tmp/repos")

    # --- Logging ---
    log_level: str = "INFO"

    # --- App ---
    app_name: str = "Code Insight Tool"
    app_version: str = "0.1.0"


# Singleton instance
settings = Settings()
