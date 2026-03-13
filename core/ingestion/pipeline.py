"""
Ingestion Pipeline — end-to-end orchestrator.
Git/Local → Filter → Parse → Chunk → Embed → Store.
""" 

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from core.ingestion.git_handler import GitHandler
from core.ingestion.file_filter import FileFilter, FileInfo
from core.ingestion.ast_parser import ASTParser
from core.ingestion.chunker import Chunker, CodeChunk
from core.embedding.batch_processor import BatchProcessor
from core.embedding.embedder import CodeEmbedder
from storage.vector_store import VectorStore
from storage.metadata_store import MetadataStore
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Result of a complete ingestion run."""

    project_id: int
    project_name: str
    total_files: int
    total_chunks: int
    total_errors: int
    duration_seconds: float
    errors: List[Dict[str, str]]


class IngestionPipeline:
    """Orchestrates the full code ingestion pipeline."""

    def __init__(self):
        self.git_handler = GitHandler()
        self.ast_parser = ASTParser()
        self.chunker = Chunker()
        self.embedder = CodeEmbedder()
        self.vector_store = VectorStore()
        self.batch_processor = BatchProcessor(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        self.metadata_store = MetadataStore()

    def run(
        self,
        source: str,
        project_name: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> IngestionResult:
        logger.info(f"Running ingestion pipeline for {source}")
        start_time = time.time()
        errors = []

        # --- Stage 1: Resolve source ---
        self._report(progress_callback, "Resolving source...", 0.0)

        if GitHandler.is_local_folder(source):
            repo_path = Path(source)
            if project_name is None:
                project_name = repo_path.name
            logger.info(f"Using local folder: {repo_path}")
        else:
            logger.info(f"Cloning from Git: {source}")
            self._report(progress_callback, "Cloning repository...", 0.05)
            repo_path = self.git_handler.clone_repo(source, project_name)
            if project_name is None:
                project_name = repo_path.name

        commit_hash = self.git_handler.get_commit_hash(repo_path)

        # --- Stage 2: Filter files ---
        self._report(progress_callback, "Scanning & filtering files...", 0.15)

        all_files = self.git_handler.list_all_files(repo_path)
        file_filter = FileFilter(repo_path)
        valid_files = file_filter.filter_files(all_files)

        logger.info(f"Found {len(valid_files)} valid source files")

        # --- Stage 3: Parse & Chunk ---
        self._report(progress_callback, "Parsing & chunking code...", 0.25)

        all_chunks: List[CodeChunk] = []
        for i, file_info in enumerate(valid_files):
            try:
                chunks = self._process_file(file_info, repo_path)
                all_chunks.extend(chunks)
            except Exception as e:
                error_msg = f"{file_info.relative_path}: {str(e)}"
                errors.append({"file": file_info.relative_path, "error": str(e)})
                logger.error(f"Error processing file: {error_msg}")

            # Update progress (25% → 65% for parsing)
            pct = 0.25 + (0.40 * (i + 1) / len(valid_files))
            self._report(
                progress_callback,
                f"Parsing: {file_info.relative_path}",
                pct,
            )

        logger.info(f"Total chunks: {len(all_chunks)}")

        # --- Stage 4: Embed & Store ---
        self._report(progress_callback, "Creating vector collection...", 0.65)

        # Create/reset Qdrant collection
        collection_name = f"project_{project_name}"
        try:
            self.vector_store.delete_collection(collection_name)
        except Exception:
            pass  # Collection might not exist
        self.vector_store.create_collection(
            collection_name=collection_name,
            vector_size=self.embedder.embedding_dim,
        )

        self._report(progress_callback, "Embedding & storing chunks...", 0.70)

        def embed_progress(processed: int, total: int):
            pct = 0.70 + (0.25 * processed / total)
            self._report(progress_callback, f"Embedding: {processed}/{total}", pct)

        total_stored = self.batch_processor.process_chunks(
            chunks=all_chunks,
            collection_name=collection_name,
            progress_callback=embed_progress,
        )

        # --- Stage 5: Save metadata ---
        self._report(progress_callback, "Saving metadata...", 0.95)

        duration = time.time() - start_time
        project_id = self.metadata_store.create_project(
            name=project_name,
            path=source,
            commit_hash=commit_hash,
        )
        self.metadata_store.save_index_stats(
            project_id=project_id,
            total_files=len(valid_files),
            total_chunks=total_stored,
            total_errors=len(errors),
            duration_seconds=duration,
        )

        self._report(progress_callback, "Done!", 1.0)

        result = IngestionResult(
            project_id=project_id,
            project_name=project_name,
            total_files=len(valid_files),
            total_chunks=total_stored,
            total_errors=len(errors),
            duration_seconds=duration,
            errors=errors,
        )

        logger.info(
            f"Ingestion complete: {result.total_files} files, "
            f"{result.total_chunks} chunks, {result.total_errors} errors, "
            f"{result.duration_seconds:.1f}s"
        )

        return result

    def _process_file(
        self,
        file_info: FileInfo,
        repo_path: Path,
    ) -> List[CodeChunk]:
        source_code = file_info.path.read_text(encoding="utf-8", errors="ignore")

        # Try AST parsing
        if self.ast_parser.is_supported(file_info.language):
            ast_nodes = self.ast_parser.parse_file(
                file_path=file_info.path,
                language=file_info.language,
                source_code=source_code,
            )

            if ast_nodes:
                return self.chunker.chunk_from_ast(
                    ast_nodes=ast_nodes,
                    file_path=file_info.relative_path,
                    language=file_info.language,
                    source_code=source_code,
                )

        # Fallback to sliding window
        logger.debug(f"Using fallback chunking for {file_info.relative_path}")
        return self.chunker.chunk_fallback(
            source_code=source_code,
            file_path=file_info.relative_path,
            language=file_info.language,
        )

    @staticmethod
    def _report(
        callback: Optional[Callable[[str, float], None]],
        message: str,
        percent: float,
    ) -> None:
        """Report progress to optional callback."""
        if callback:
            callback(message, percent)
