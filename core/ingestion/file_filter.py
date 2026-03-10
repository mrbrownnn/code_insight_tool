"""
File Filter — filter files by .gitignore, size, language, and binary detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pathspec

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FileInfo:
    """Information about a filtered source file."""

    path: Path
    relative_path: str
    language: str
    size: int


class FileFilter:
    """Filters source files based on .gitignore, language support, size, etc."""

    # Common binary file extensions to skip
    BINARY_EXTENSIONS = {
        ".zip", ".tar", ".gz", ".rar", ".7z",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".exe", ".dll", ".so", ".dylib", ".o",
        ".pyc", ".pyo", ".class", ".jar",
        ".woff", ".woff2", ".ttf", ".eot",
        ".db", ".sqlite", ".sqlite3",
    }

    # Files to always skip
    SKIP_FILES = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "poetry.lock", "Pipfile.lock",
        ".DS_Store", "Thumbs.db",
    }

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.gitignore_spec = self._load_gitignore()

    def _load_gitignore(self) -> Optional[pathspec.PathSpec]:
        """Load .gitignore patterns from the project root."""
        gitignore_path = self.root_dir / ".gitignore"
        if not gitignore_path.exists():
            return None

        with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
            patterns = f.read().splitlines()

        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        logger.info(f"Loaded {len(patterns)} .gitignore patterns")
        return spec

    def filter_files(self, files: List[Path]) -> List[FileInfo]:
        """Filter a list of files, returning only supported source files.

        Args:
            files: List of absolute file paths.

        Returns:
            List of FileInfo for valid source files.
        """
        result = []

        for file_path in files:
            # Skip binary extensions
            if file_path.suffix.lower() in self.BINARY_EXTENSIONS:
                continue

            # Skip known non-source files
            if file_path.name in self.SKIP_FILES:
                continue

            # Check .gitignore
            relative = file_path.relative_to(self.root_dir)
            if self.gitignore_spec and self.gitignore_spec.match_file(
                str(relative)
            ):
                continue

            # Check file size
            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            if size > settings.max_file_size_bytes:
                logger.debug(f"Skipping large file: {relative} ({size} bytes)")
                continue

            if size == 0:
                continue

            # Detect language
            language = self._detect_language(file_path)
            if language is None:
                continue

            result.append(
                FileInfo(
                    path=file_path,
                    relative_path=str(relative),
                    language=language,
                    size=size,
                )
            )

        logger.info(
            f"Filtered: {len(result)}/{len(files)} files are valid source files"
        )
        return result

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension.

        Args:
            file_path: Path to the file.

        Returns:
            Language name or None if unsupported.
        """
        ext = file_path.suffix.lower()
        return settings.supported_extensions.get(ext)
