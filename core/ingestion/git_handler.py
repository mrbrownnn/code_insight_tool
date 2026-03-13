"""
Git Handler — clone repos and manage local copies.
"""

import shutil
from pathlib import Path
from typing import Optional

from git import Repo, InvalidGitRepositoryError

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class GitHandler:
    """Handles Git repository operations."""

    def __init__(self, clone_dir: Path = None):
        self.clone_dir = clone_dir or settings.clone_dir
        self.clone_dir.mkdir(parents=True, exist_ok=True)

    def clone_repo(self, url: str, project_name: str = None) -> Path:
        """Clone a Git repository to local storage.

        Args:
            url: Git repository URL.
            project_name: Optional name for the local folder.

        Returns:
            Path to the cloned repository.
        """
        if project_name is None:
            project_name = url.rstrip("/").split("/")[-1]
            if project_name.endswith(".git"):
                project_name = project_name[:-4]

        target_dir = self.clone_dir / project_name

        if target_dir.exists():
            logger.info(f"Removing existing clone at {target_dir}")
            shutil.rmtree(target_dir)

        logger.info(f"Cloning {url} → {target_dir}")
        Repo.clone_from(url, str(target_dir))
        logger.info(f"Cloned successfully: {target_dir}")
        return target_dir

    def get_commit_hash(self, repo_path: Path) -> Optional[str]:
        try:
            repo = Repo(str(repo_path))
            return repo.head.commit.hexsha
        except (InvalidGitRepositoryError, ValueError):
            logger.warning(f"{repo_path} is not a valid Git repository")
            return None

    def list_all_files(self, repo_path: Path) -> list[Path]:
        files = []
        for item in repo_path.rglob("*"):
            if item.is_file() and ".git" not in item.parts:
                files.append(item)
        return sorted(files)

    @staticmethod
    def is_local_folder(path_str: str) -> bool:
        logger.info(f"Path {path_str} is a valid local folder")
        return Path(path_str).exists() and Path(path_str).is_dir()
