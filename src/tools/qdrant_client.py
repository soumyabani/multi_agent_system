from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from src.config import settings


@dataclass
class RetrievalResult:
    """Normalized result from a Qdrant similarity search."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class QdrantKnowledgeClient:
    """Thin wrapper around Qdrant for retrieving local incident/deal-desk knowledge."""

    def __init__(self, collection_name: str | None = None, url: str | None = None, api_key: str | None = None) -> None:
        qdrant_cfg = settings.qdrant_config
        self.collection_name = collection_name or qdrant_cfg.collection
        client_kwargs: dict[str, Any] = {"url": url or qdrant_cfg.url}
        resolved_api_key = api_key or qdrant_cfg.api_key
        if resolved_api_key:
            client_kwargs["api_key"] = resolved_api_key
        self.client = QdrantClient(**client_kwargs)

    def ensure_collection(self, vector_size: int, distance: str = "Cosine", **_: Any) -> None:
        """Create the knowledge collection if it does not exist."""
        if self.client.collection_exists(self.collection_name):
            return

        distance_value = getattr(Distance, distance.upper(), Distance.COSINE)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance_value),
        )

    def upsert_document(
        self,
        document_id: int | str | UUID,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or replace a single document in the collection."""
        if not isinstance(document_id, (int, str, UUID)):
            raise TypeError("document_id must be an int, str, or UUID.")
        if isinstance(document_id, int) and document_id < 0:
            raise ValueError("document_id must be >= 0 when an integer is used.")
        if isinstance(document_id, str) and not document_id.strip():
            raise ValueError("document_id string must not be empty.")

        payload = {"content": content}
        if metadata:
            safe_metadata = {key: value for key, value in metadata.items() if key != "content"}
            payload.update(safe_metadata)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=document_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

    def search(self, query_embedding: list[float], limit: int = 5, score_threshold: float = 0.0) -> list[RetrievalResult]:
        """Return semantically relevant documents for a query embedding."""
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )

        results: list[RetrievalResult] = []
        for item in response.points:
            payload = getattr(item, "payload", {}) or {}
            content = str(payload.get("content", ""))
            score = float(getattr(item, "score", 0.0) or 0.0)
            metadata = {k: v for k, v in payload.items() if k != "content"}
            results.append(RetrievalResult(content=content, score=score, metadata=metadata))
        return results

    def fetch_by_filter(self, filters: dict[str, Any], limit: int = 5) -> list[RetrievalResult]:
        """Return documents that match a payload filter expression."""
        if not filters:
            return []

        must_conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
        ]

        response = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must_conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results: list[RetrievalResult] = []
        for point in response[0]:
            payload = point.payload or {}
            content = str(payload.get("content", ""))
            metadata = {k: v for k, v in payload.items() if k != "content"}
            results.append(RetrievalResult(content=content, score=0.0, metadata=metadata))
        return results


def build_qdrant_client() -> QdrantKnowledgeClient:
    """Factory for local environment-backed Qdrant client."""
    return QdrantKnowledgeClient()
