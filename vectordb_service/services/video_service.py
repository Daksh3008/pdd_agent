"""Orchestrates the full video-processing pipeline: scenes → frames → Gemini → embeddings → Pinecone."""

import logging
import os
import uuid

import numpy as np

from config import Settings
from infrastructure.embedding_service import EmbeddingService
from infrastructure.frame_extractor import extract_scene_frames
from infrastructure.gemini_describer import describe_frames
from infrastructure.scene_detector import detect_scenes
from infrastructure.vector_store import VectorStore

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._settings = settings
        self._embedding = embedding_service
        self._vector_store = vector_store

    # -- public ---------------------------------------------------------

    def process_video(self, video_bytes: bytes, original_filename: str) -> dict:
        video_id = uuid.uuid4().hex[:12]
        video_dir = os.path.join(self._settings.data_dir, video_id)
        frames_dir = os.path.join(video_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Persist uploaded file
        video_path = os.path.join(video_dir, original_filename)
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # 1. Scene detection
        logger.info("Detecting scenes for video %s", video_id)
        scenes = detect_scenes(video_path, self._settings.scene_threshold)
        logger.info("Scenes detected: %d", len(scenes))

        # 2. Frame extraction
        frames = extract_scene_frames(video_path, scenes, frames_dir)
        logger.info("Frames extracted: %d", len(frames))

        # 3a. Gemini batch description — send all frames in video order
        sorted_frames = sorted(frames, key=lambda f: (f["scene_id"], f["frame_no"]))
        try:
            gemini_results = describe_frames(
                api_key=self._settings.gemini_api_key,
                frame_infos=sorted_frames,
                model=self._settings.gemini_model,
            )
        except Exception:
            logger.exception("Gemini batch description failed")
            gemini_results = {}

        # 3b. Embeddings + vector assembly (CLIP is fast on CPU, keep sequential)
        vectors_to_upsert: list[tuple] = []

        for frame_info in frames:
            filepath = frame_info["filepath"]
            processed = gemini_results.get(filepath)
            if processed is None:
                continue

            try:
                image_emb = self._embedding.get_image_embedding(filepath)
                search_text = processed.get("search_keywords", "")
                text_emb = self._embedding.get_text_embedding(search_text)

                combined = np.concatenate(
                    [image_emb.flatten(), text_emb.flatten()]
                ).tolist()

                metadata = {
                    "video_id": video_id,
                    "filename": frame_info["filename"],
                    "scene_id": frame_info["scene_id"],
                    "step_number": processed.get("step_number", 0),
                    "step_title": processed.get("step_title", ""),
                    "description": processed.get("description", ""),
                }

                vector_id = f"{video_id}_scene_{frame_info['scene_id']}_{frame_info['label']}"
                vectors_to_upsert.append((vector_id, combined, metadata))
            except Exception:
                logger.exception(
                    "Embedding failed for scene_%d_%s, skipping",
                    frame_info["scene_id"], frame_info["label"],
                )

        # 4. Upsert to Pinecone
        if vectors_to_upsert:
            self._vector_store.upsert(vectors_to_upsert)
            logger.info("Upserted %d vectors", len(vectors_to_upsert))

        # Clean up the raw video file to save disk space (keep frames)
        os.remove(video_path)

        return {
            "video_id": video_id,
            "scenes_detected": len(scenes),
            "frames_extracted": len(frames),
            "frames_indexed": len(vectors_to_upsert),
        }
