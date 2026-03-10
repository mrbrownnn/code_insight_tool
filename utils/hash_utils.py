"""
Content and file hashing utilities.
Used for chunk fingerprinting and incremental indexing.
"""

import hashlib
from pathlib import Path


def hash_content(text: str) -> str:
    """Generate SHA256 hash of text content.

    Used for chunk fingerprinting — detect if a code chunk has changed
    since last indexing.

    Args:
        text: The text content to hash.

    Returns:
        Hex string of SHA256 hash (e.g., "sha256:ab3f1c...").
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def hash_file(file_path: Path) -> str:
    """Generate SHA256 hash of a file's contents.

    Used for incremental indexing — only re-embed files that changed.

    Args:
        file_path: Path to the file.

    Returns:
        Hex string of SHA256 hash.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"
