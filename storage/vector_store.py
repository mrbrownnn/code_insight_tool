from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    payload: Dict[str, Any]


class VectorStore:
    def __init__(
        self,
        host: str = None,
        port: int = None,
    ):
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.client = QdrantClient(host=self.host, port=self.port)
        logger.info(f"Connected to Qdrant at {self.host}:{self.port}")

    def create_collection(
        self,
        collection_name: str = None,
        vector_size: int = None,
    ) -> None:
        name = collection_name or settings.qdrant_collection_name
        size = vector_size or settings.embedding_dim
        #create connection to qdrant
        self.client = QdrantClient(host=self.host, port=self.port)
        logger.info(f"Connected to Qdrant at {self.host}:{self.port}")

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection '{name}' (vector_size={size}, cosine)")

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
        collection_name: str = None,
    ) -> int:
        """Batch upsert chunks with their embedding vectors.

        Args:
            chunks: List of chunk metadata dicts. Each must have 'chunk_id'.
            vectors: Corresponding embedding vectors.
            collection_name: Target collection.

        Returns:
            Number of points upserted.
        """
        name = collection_name or settings.qdrant_collection_name

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
            )

        points = []
        for chunk, vector in zip(chunks, vectors):
            chunk_id = chunk.pop("chunk_id")
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=chunk,
                )
            )

        self.client.upsert(
            collection_name=name,
            points=points,
        )

        logger.info(f"Upserted {len(points)} points to '{name}'")
        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filters: Optional[Dict[str, str]] = None,
        collection_name: str = None,
    ) -> List[SearchResult]:
        name = collection_name or settings.qdrant_collection_name

        # Build Qdrant filter from dict
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            SearchResult(
                chunk_id=str(hit.id),
                score=hit.score,
                payload=hit.payload or {},
            )
            for hit in results
        ]

    def delete_collection(self, collection_name: str = None) -> None:
        name = collection_name or settings.qdrant_collection_name
        self.client.delete_collection(collection_name=name)
        logger.info(f"Deleted collection '{name}'")
        #Breakpoint() debugger function
        try:
            self.client.get_collection(collection_name = name)
        except Exception as e:
            logger.info(f"Collection '{name}' not found")

    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:

        name = collection_name or settings.qdrant_collection_name
        info = self.client.get_collection(collection_name=name)
        return {
            "name": name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }
