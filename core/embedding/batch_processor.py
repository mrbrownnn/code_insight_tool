"""
Batch Processor — processes chunks in batches for embedding and storage.
"""

from typing import Callable, List, Optional

from core.embedding.embedder import CodeEmbedder
from core.ingestion.chunker import CodeChunk
from storage.vector_store import VectorStore
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class BatchProcessor:
    def __init__(
        self,
        embedder: CodeEmbedder = None,
        vector_store: VectorStore = None,
        batch_size: int = None,
    ):
        self.embedder = embedder or CodeEmbedder()
        self.vector_store = vector_store or VectorStore()
        self.batch_size = batch_size or settings.batch_size

    def process_chunks(
        self,
        chunks: List[CodeChunk],
        collection_name: str = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        total = len(chunks)
        total_stored = 0
        logger.info(f"Processing {total} chunks in batches of {self.batch_size}")
        for i in range(0, total, self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [chunk.to_embedding_text() for chunk in batch]
            vectors = self.embedder.embed_batch(texts)

            payloads = [chunk.to_dict() for chunk in batch]

            stored = self.vector_store.upsert_chunks(
                chunks=payloads,
                vectors=vectors,
                collection_name=collection_name,
            )
            total_stored += stored

            # Progress callback
            if progress_callback:
                progress_callback(min(i + len(batch), total), total)

            logger.debug(
                f"Batch {i // self.batch_size + 1}: "
                f"embedded & stored {len(batch)} chunks"
            )

        logger.info(f"Finished processing: {total_stored}/{total} chunks stored")
        return total_stored
