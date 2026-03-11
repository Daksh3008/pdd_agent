"""Handles text-based frame querying against the Pinecone index."""

import logging
import os

import numpy as np

from config import Settings
from infrastructure.embedding_service import EmbeddingService
from infrastructure.vector_store import VectorStore

logger = logging.getLogger(__name__)


class QueryService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._settings = settings
        self._embedding = embedding_service
        self._vector_store = vector_store

    def query_frames(self, query_text: str, top_k: int = 5, video_id: str | None = None) -> dict:
        # Text embedding (512-d for CLIP ViT-B/32)
        text_emb = self._embedding.get_text_embedding(query_text)

        # Place text embedding in both halves so it matches against
        # both the image and text portions of stored vectors.
        combined = np.concatenate(
            [text_emb.flatten(), text_emb.flatten()]
        ).tolist()

        metadata_filter = {"video_id": {"$eq": video_id}} if video_id else None
        results = self._vector_store.query(combined, top_k=top_k, filter=metadata_filter)

        matches: list[dict] = []
        for m in results.get("matches", []):
            meta = m.get("metadata", {})
            video_id = meta.get("video_id", "")
            filename = meta.get("filename", "")
            image_url = f"/static/{video_id}/frames/{filename}" if video_id and filename else ""

            matches.append(
                {
                    "vector_id": m["id"],
                    "score": m["score"],
                    "metadata": meta,
                    "image_url": image_url,
                }
            )

        return {"query": query_text, "results": matches}
