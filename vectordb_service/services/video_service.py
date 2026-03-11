"""Orchestrates the full video-processing pipeline: scenes → frames → OmniParser → embeddings → Pinecone."""

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import replicate

from config import Settings
from infrastructure.embedding_service import EmbeddingService
from infrastructure.frame_extractor import extract_scene_frames
from infrastructure.omniparser_client import parse_frame
from infrastructure.omniparser_processor import process_omniparser_output
from infrastructure.scene_detector import detect_scenes
from infrastructure.vector_store import VectorStore

logger = logging.getLogger(__name__)

_OMNIPARSER_WORKERS = 5  # parallel Replicate calls


class VideoService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        replicate_client: replicate.Client,
    ) -> None:
        self._settings = settings
        self._embedding = embedding_service
        self._vector_store = vector_store
        self._replicate = replicate_client

    # -- internal helpers -----------------------------------------------

    def _run_omniparser(self, frame_info: dict) -> tuple[dict, dict]:
        """Call OmniParser for a single frame (runs in a thread)."""
        raw = parse_frame(
            frame_info["filepath"],
            self._settings.omniparser_model,
            self._replicate,
        )
        processed = process_omniparser_output(frame_info["frame_no"], raw)
        return frame_info, processed

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

        # 3a. OmniParser — run in parallel threads (I/O-bound API calls)
        omni_results: dict[str, dict] = {}  # filepath → processed output
        with ThreadPoolExecutor(max_workers=_OMNIPARSER_WORKERS) as pool:
            futures = {
                pool.submit(self._run_omniparser, fi): fi for fi in frames
            }
            for future in as_completed(futures):
                fi = futures[future]
                try:
                    _, processed = future.result()
                    omni_results[fi["filepath"]] = processed
                except Exception:
                    logger.exception(
                        "OmniParser failed for scene_%d_%s, skipping",
                        fi["scene_id"], fi["label"],
                    )

        # 3b. Embeddings + vector assembly (CLIP is fast on CPU, keep sequential)
        vectors_to_upsert: list[tuple] = []

        for frame_info in frames:
            filepath = frame_info["filepath"]
            processed = omni_results.get(filepath)
            if processed is None:
                continue

            try:
                image_emb = self._embedding.get_image_embedding(filepath)
                summary = processed["summary"]
                text_for_emb = (
                    " ".join(summary["visible_text"])
                    + " "
                    + " ".join(summary["possible_actions"])
                )
                text_emb = self._embedding.get_text_embedding(text_for_emb)

                combined = np.concatenate(
                    [image_emb.flatten(), text_emb.flatten()]
                ).tolist()

                metadata = {
                    "video_id": video_id,
                    "frame_no": frame_info["frame_no"],
                    "scene_id": frame_info["scene_id"],
                    "label": frame_info["label"],
                    "filename": frame_info["filename"],
                    "omniparser_summary_text": " ".join(summary["visible_text"]),
                    "omniparser_summary_actions": " ".join(summary["possible_actions"]),
                    "compact_elements": processed["compact_elements"],
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
