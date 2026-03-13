"""
SQLite Metadata Store — quản lý project metadata và index stats.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)
DB_PATH = Path("storage/metadata.db")


class MetadataStore:
    """SQLite-based metadata storage for projects and indexing stats."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                commit_hash TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                indexed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS index_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                total_files INTEGER,
                total_chunks INTEGER,
                total_errors INTEGER,
                duration_seconds REAL,
                indexed_at TEXT,
                FOREIGN KEY(project_id) REFERENCES projects(id)
            );
            """
        )
            conn.commit()
            logger.info(f"Metadata DB initialized at {self.db_path}")
        finally:
            conn.close()

    def create_project(
        self,
        name: str,
        path: str,
        commit_hash: Optional[str] = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                INSERT INTO projects (name, path, commit_hash, indexed_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, path, commit_hash, now),
            )
            conn.commit()
            project_id = cursor.lastrowid
            logger.info(f"Created project '{name}' (id={project_id})")
            return project_id
        finally:
            conn.close()

    def save_index_stats(
        self,
        project_id: int,
        total_files: int,
        total_chunks: int,
        total_errors: int,
        duration_seconds: float,
    ) -> None:
        """Save indexing statistics for a project.

        Args:
            project_id: The project ID.
            total_files: Number of files processed.
            total_chunks: Number of chunks generated.
            total_errors: Number of errors encountered.
            duration_seconds: Total indexing time.
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO index_stats
                    (project_id, total_files, total_chunks, total_errors,
                     duration_seconds, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (project_id, total_files, total_chunks, total_errors,
                 duration_seconds, now),
            )
            conn.commit()
            logger.info(
                f"Saved stats for project {project_id}: "
                f"{total_files} files, {total_chunks} chunks, "
                f"{total_errors} errors, {duration_seconds:.1f}s"
            )
        finally:
            conn.close()

    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get a project by ID.

        Returns:
            Project dict or None.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get all projects.

        Returns:
            List of project dicts.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_latest_stats(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get the latest index stats for a project.

        Returns:
            Stats dict or None.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT * FROM index_stats
                WHERE project_id = ?
                ORDER BY indexed_at DESC
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def delete_project(self, project_id: int) -> None:
        """Delete a project and its stats.

        Args:
            project_id: Project to delete.
        """
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()
            logger.info(f"Deleted project {project_id}")
        finally:
            conn.close()
