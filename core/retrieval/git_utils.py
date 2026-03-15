from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class GitUtils:
    
    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = Path(repo_path or ".")
        
    def get_last_modified_time(self, filepath: str) -> Optional[datetime]:
        try:
            full_path = self.repo_path / filepath
            if full_path.exists():
                mtime = full_path.stat().st_mtime
                return datetime.fromtimestamp(mtime)
        except Exception as e:
            logger.warning(f"Failed to get mtime for {filepath}: {e}")
        return None
    
    def get_recency_score(self, filepath: str) -> float:
        return 0.5
    
    def get_file_history(self, filepath: str) -> Dict:
        return {
            "last_modified": None,
            "commit_count": 0,
            "authors": [],
        }
