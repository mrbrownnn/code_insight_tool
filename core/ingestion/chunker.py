"""
Smart Chunker — AST-based code chunking with fallback.
Splits code into semantically meaningful chunks for embedding.
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.ingestion.ast_parser import ASTNode
from config import settings
from utils.hash_utils import hash_content
from utils.token_counter import estimate_tokens, is_within_token_limit
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CodeChunk:
    conversation_id: str
    chunk_id: str
    file_path: str
    language: str
    chunk_type: str     
    parent_scope: Optional[str]
    line_start: int
    line_end: int
    source_code: str
    content_hash: str
    metadata: Dict = field(default_factory=dict)
    # embedding include conversation id to make it unique, continue to chat with the past conversation
    def to_embedding_text(self) -> str:
        return (
            f"{self.language} {self.chunk_type} "
            f"in {self.file_path}: {self.source_code}"
        )

    def to_dict(self) -> Dict:
        return {
            "conversation_id": self.conversation_id,
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "conversation_name": self.conversation_name,
            "parent_scope": self.parent_scope,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_hash": self.content_hash,
            "source_code": self.source_code,
            **self.metadata,
        }


class Chunker:
    def __init__(
        self,
        max_tokens: int = None,
        fallback_window: int = None,
        overlap_lines: int = None,
    ):
        self.max_tokens = max_tokens or settings.chunk_max_tokens
        self.fallback_window = fallback_window or settings.chunk_fallback_window
        self.overlap_lines = overlap_lines or settings.chunk_overlap_lines

    def chunk_from_ast(
        self,
        ast_nodes: List[ASTNode],
        file_path: str,
        language: str,
        source_code: str,
    ) -> List[CodeChunk]:
        chunks = []
        covered_lines = set()

        for node in ast_nodes:
            node_chunks = self._chunk_node(node, file_path, language)
            chunks.extend(node_chunks)

            # Track which lines are covered by AST nodes
            for line_num in range(node.start_line, node.end_line + 1):
                covered_lines.add(line_num)

            # Process children (methods inside classes)
            for child in node.children:
                child_chunks = self._chunk_node(child, file_path, language)
                chunks.extend(child_chunks)
                for line_num in range(child.start_line, child.end_line + 1):
                    covered_lines.add(line_num)

        # Chunk uncovered lines (imports, top-level code, etc.)
        uncovered_chunks = self._chunk_uncovered_lines(
            source_code, file_path, language, covered_lines
        )
        chunks.extend(uncovered_chunks)

        logger.debug(f"Chunked {file_path}: {len(chunks)} chunks")
        return chunks

    def chunk_fallback(
        self,
        source_code: str,
        file_path: str,
        language: str,
    ) -> List[CodeChunk]:
    
        lines = source_code.split("\n")
        chunks = []

        start = 0
        while start < len(lines):
            end = min(start + self.fallback_window, len(lines))
            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines)

            if chunk_text.strip():
                chunks.append(
                    CodeChunk(
                        chunk_id=self._generate_id(),
                        file_path=file_path,
                        language=language,
                        chunk_type="block",
                        name=f"block_{start + 1}_{end}",
                        parent_scope=None,
                        line_start=start + 1,
                        line_end=end,
                        source_code=chunk_text,
                        content_hash=hash_content(chunk_text),
                    )
                )

            start = end - self.overlap_lines
            if start >= len(lines) - self.overlap_lines:
                break

        logger.debug(f"Fallback chunked {file_path}: {len(chunks)} chunks")
        return chunks

    def _chunk_node(
        self,
        node: ASTNode,
        file_path: str,
        language: str,
    ) -> List[CodeChunk]:
        chunk_type = self._classify_node_type(node.node_type)

        if is_within_token_limit(node.source_code, self.max_tokens):
            return [
                CodeChunk(
                    chunk_id=self._generate_id(),
                    file_path=file_path,
                    language=language,
                    chunk_type=chunk_type,
                    name=node.name,
                    parent_scope=node.parent_name,
                    line_start=node.start_line,
                    line_end=node.end_line,
                    source_code=node.source_code,
                    content_hash=hash_content(node.source_code),
                )
            ]
        else:
            # Node too large — split with sliding window
            return self._split_large_node(node, file_path, language, chunk_type)

    def _split_large_node(
        self,
        node: ASTNode,
        file_path: str,
        language: str,
        chunk_type: str,
    ) -> List[CodeChunk]:
        """Split an oversized AST node using sliding window."""
        lines = node.source_code.split("\n")
        chunks = []

        start = 0
        part = 1
        while start < len(lines):
            end = min(start + self.fallback_window, len(lines))
            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines)

            if chunk_text.strip():
                chunks.append(
                    CodeChunk(
                        chunk_id=self._generate_id(),
                        file_path=file_path,
                        language=language,
                        chunk_type=chunk_type,
                        name=f"{node.name}_part{part}",
                        parent_scope=node.parent_name,
                        line_start=node.start_line + start,
                        line_end=node.start_line + end - 1,
                        source_code=chunk_text,
                        content_hash=hash_content(chunk_text),
                    )
                )
                part += 1

            start = end - self.overlap_lines
            if start >= len(lines) - self.overlap_lines:
                break

        return chunks

    def _chunk_uncovered_lines(
        self,
        source_code: str,
        file_path: str,
        language: str,
        covered_lines: set,
    ) -> List[CodeChunk]:
        """Chunk lines not covered by any AST node (imports, comments, etc.)."""
        lines = source_code.split("\n")
        chunks = []
        current_block = []
        block_start = None

        for i, line in enumerate(lines, start=1):
            if i not in covered_lines and line.strip():
                if block_start is None:
                    block_start = i
                current_block.append(line)
            else:
                if current_block and len(current_block) >= 3:
                    chunk_text = "\n".join(current_block)
                    chunks.append(
                        CodeChunk(
                            chunk_id=self._generate_id(),
                            file_path=file_path,
                            language=language,
                            chunk_type="block",
                            name=f"imports_block_{block_start}",
                            parent_scope=None,
                            line_start=block_start,
                            line_end=block_start + len(current_block) - 1,
                            source_code=chunk_text,
                            content_hash=hash_content(chunk_text),
                        )
                    )
                current_block = []
                block_start = None

        # Handle remaining block
        if current_block and len(current_block) >= 3:
            chunk_text = "\n".join(current_block)
            chunks.append(
                CodeChunk(
                    chunk_id=self._generate_id(),
                    file_path=file_path,
                    language=language,
                    chunk_type="block",
                    name=f"imports_block_{block_start}",
                    parent_scope=None,
                    line_start=block_start,
                    line_end=block_start + len(current_block) - 1,
                    source_code=chunk_text,
                    content_hash=hash_content(chunk_text),
                )
            )

        return chunks

    @staticmethod
    def _classify_node_type(ast_type: str) -> str:
        """Map AST node type to a simpler chunk_type label."""
        if "class" in ast_type or "interface" in ast_type:
            return "class"
        if "method" in ast_type or "constructor" in ast_type:
            return "method"
        if "function" in ast_type or "arrow" in ast_type:
            return "function"
        return "block"

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique chunk ID."""
        return str(uuid.uuid4())
