# pipeline/audio_pipeline.py

"""
Audio Pipeline Orchestrator.
Meeting recording (with audio) → Transcript → LLM → PDD Document.

Flow:
1. Extract audio from video (FFmpeg)
2. Transcribe audio (Whisper)
3. Extract entities & project name
4. Generate document sections (purpose, overview, as-is, to-be)
5. Extract process steps + detailed steps
6. Generate flowchart
7. Extract & match frames to steps
8. Assemble PDD document
"""

import os
import time
import re
from typing import Optional, List, Tuple, Dict

from core.config import config
from core.gemini_client import gemini_client
from core.token_tracker import reset_tracker
from core.utils import build_entity_hint

from audio.video_to_audio import convert_video_to_audio
from audio.transcriber import transcribe_audio, read_transcript

from llm_tasks.entity_extraction import extract_entities_and_project
from llm_tasks.document_sections import (
    get_document_purpose_text,
    get_overview_and_justification,
    get_as_is_process,
    get_to_be_process
)
from llm_tasks.process_steps import extract_process_steps, get_detailed_process_steps
from llm_tasks.requirements import (
    get_input_requirements,
    get_interface_requirements,
    get_exception_handling
)
from llm_tasks.flowchart_dot import generate_dot_and_apps

from pipeline.common import (
    save_persistent_document, save_dot_code,
    generate_flowchart, build_document,
    print_pipeline_header, print_pipeline_footer
)
from document.pdd_generator import PDDGenerator

class AudioPipeline:
    """Pipeline for meeting recordings with audio."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.paths.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _extract_candidate_frames(
        self, video_path: str, transcript_path: str,
        transcript: str, frames_dir: str, num_target: int
    ) -> List[Tuple[str, str, float]]:
        """
        Extract a pool of candidate frames for step matching.
        Returns list of (frame_path, transcript_text, timestamp) tuples.
        """
        from video.ocr_engine import OCR_AVAILABLE

        print(f"    Extracting candidate frame pool (target: {num_target * 3})...")
        os.makedirs(frames_dir, exist_ok=True)
        candidates = []
        used_timestamps = set()
        pool_size = min(num_target * 3, 60)

        # Source 1: Keyword-based from transcript
        try:
            from audio.frame_extractor import (
                extract_frames_with_transcripts,
                extract_frame, get_video_duration
            )

            keyword_pairs = extract_frames_with_transcripts(
                video_path, transcript_path, frames_dir
            )
            for fp, desc in keyword_pairs:
                if len(candidates) < pool_size:
                    candidates.append((fp, desc, -1))
        except Exception as e:
            print(f"    Keyword extraction error: {e}")

        # Source 2: Evenly spaced to fill pool
        remaining = pool_size - len(candidates)
        if remaining > 0:
            try:
                from audio.frame_extractor import (
                    extract_frame, get_video_duration
                )
                print(f"    Source 2: {remaining} evenly-spaced frames...")
                duration = get_video_duration(video_path)
                if duration > 0:
                    start_t = duration * 0.02
                    end_t = duration * 0.98
                    interval = (end_t - start_t) / (remaining + 1)

                    # Build transcript lookup
                    transcript_lookup = {}
                    try:
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                m = re.match(
                                    r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(.*)',
                                    line.strip()
                                )
                                if m:
                                    transcript_lookup[float(m.group(1))] = (
                                        m.group(3).strip()
                                    )
                    except Exception:
                        pass

                    for i in range(remaining):
                        ts = start_t + interval * (i + 1)
                        if any(abs(ts - used) < 1.5 for used in used_timestamps):
                            ts += interval * 0.3
                        fp = extract_frame(video_path, ts, frames_dir)
                        if fp:
                            closest_text = ""
                            min_diff = float('inf')
                            for t_ts, t_text in transcript_lookup.items():
                                diff = abs(t_ts - ts)
                                if diff < min_diff:
                                    min_diff = diff
                                    closest_text = t_text
                            candidates.append((fp, closest_text, ts))
                            used_timestamps.add(ts)
            except Exception as e:
                print(f"    Evenly-spaced extraction error: {e}")

        print(f"    Total candidate frames: {len(candidates)}")
        return candidates

    def process(
        self,
        video_path: str,
        project_name: str = None,
        whisper_model: str = None,
        transcript_path: str = None
    ) -> Optional[str]:
        """
        Process a meeting recording into a PDD document.

        Args:
            video_path: Path to video file.
            project_name: Optional project name (auto-detected if None).
            whisper_model: Whisper model size override.
            transcript_path: Pre-existing transcript path (skips transcription).

        Returns:
            Path to generated document, or None on failure.
        """
        t0 = time.time()
        tracker = reset_tracker()
        gemini_client.set_tracker(tracker)

        print_pipeline_header(
            "Meeting Recording (Audio)",
            video_path=video_path,
            project_name=project_name or "(auto-detect)",
            extra_info={"OCR": "Available" if self._ocr_available() else "Not available"}
        )

        if not os.path.exists(video_path):
            print(f"Error: {video_path} not found")
            return None

        # ── Step 1: Transcript ──
        if transcript_path and os.path.exists(transcript_path):
            print(f"\n[1/6] Using existing transcript: {transcript_path}")
        else:
            print("\n[1/6] Extracting audio...")
            audio = convert_video_to_audio(video_path, self.output_dir)
            if not audio:
                return None
            print("[1/6] Transcribing...")
            transcript_path = transcribe_audio(
                audio, self.output_dir,
                model_name=whisper_model or config.whisper.model_name
            )
            if not transcript_path:
                return None

        transcript = read_transcript(transcript_path)
        if not transcript:
            return None
        print(f"  Transcript: {len(transcript):,} chars")

        # ── Step 2: LLM Extraction ──
        print("\n[2/6] LLM extraction...")

        t = time.time()
        entities, detected = extract_entities_and_project(transcript)
        if not project_name:
            project_name = detected
        eh = build_entity_hint(entities)
        print(f"  ✓ Project: {project_name} ({time.time()-t:.0f}s)")

        t = time.time()
        purpose = get_document_purpose_text(transcript, project_name, eh)
        print(f"  ✓ Purpose ({time.time()-t:.0f}s)")

        t = time.time()
        ov_just = get_overview_and_justification(transcript, project_name, eh)
        print(f"  ✓ Overview+Justification ({time.time()-t:.0f}s)")

        t = time.time()
        as_is = get_as_is_process(transcript, project_name, eh)
        print(f"  ✓ As-Is ({time.time()-t:.0f}s)")

        t = time.time()
        to_be = get_to_be_process(transcript, project_name, eh)
        print(f"  ✓ To-Be ({time.time()-t:.0f}s)")

        # ── Step 3: Process Steps ──
        print("\n[3/6] Extracting steps...")

        t = time.time()
        steps = extract_process_steps(transcript, entities)
        print(f"  ✓ {len(steps)} process steps ({time.time()-t:.0f}s)")
        for i, s in enumerate(steps[:5]):
            print(f"    {i+1}. {s}")
        if len(steps) > 5:
            print(f"    ... +{len(steps)-5} more")

        t = time.time()
        detailed = get_detailed_process_steps(transcript, project_name, eh)
        print(f"  ✓ {len(detailed)} detailed steps ({time.time()-t:.0f}s)")

        t = time.time()
        input_reqs = get_input_requirements(transcript, project_name, eh)
        print(f"  ✓ {len(input_reqs)} inputs ({time.time()-t:.0f}s)")

        t = time.time()
        exceptions = get_exception_handling(transcript, project_name, eh)
        print(f"  ✓ {len(exceptions)} exceptions ({time.time()-t:.0f}s)")

        t = time.time()
        dot_code, apps = generate_dot_and_apps(steps, transcript, entities)
        print(f"  ✓ {len(apps)} interfaces ({time.time()-t:.0f}s)")

        # ── Step 4: Flowchart ──
        print("\n[4/6] Flowchart...")
        if dot_code:
            save_dot_code(dot_code, project_name, self.output_dir)
        fc_path = generate_flowchart(dot_code, self.output_dir, project_name)

        # ── Step 5: Screenshots + Matching ──
        print("\n[5/6] Screenshots & matching...")
        matched_frames = []

        if os.path.exists(video_path) and detailed:
            try:
                from video.frame_matcher import match_pipeline, build_candidates
                from video.ocr_engine import OCR_AVAILABLE

                num_steps = len(detailed)
                raw_candidates = self._extract_candidate_frames(
                    video_path, transcript_path, transcript,
                    os.path.join(self.output_dir, "frames"),
                    num_target=num_steps
                )

                if raw_candidates:
                    frame_pairs = [(fp, desc) for fp, desc, _ in raw_candidates]
                    timestamps = [ts for _, _, ts in raw_candidates]
                    candidates = build_candidates(frame_pairs, timestamps)

                    print(f"  Running frame matcher "
                          f"({'with OCR' if OCR_AVAILABLE else 'transcript-only'})...")
                    matched_frames = match_pipeline(
                        candidates, detailed, run_ocr=OCR_AVAILABLE
                    )
            except Exception as e:
                print(f"  Frame matching error: {e}")

        # ── Step 6: Generate Document ──
        print("\n[6/6] Generating document...")

        # Convert steps list to dicts for document
        process_steps_dicts = None
        if steps:
            process_steps_dicts = [
                {"number": i + 1, "description": s} for i, s in enumerate(steps)
            ]

        doc_path = build_document(
            project_name=project_name,
            output_dir=self.output_dir,
            purpose=purpose,
            overview=ov_just.get("overview", ""),
            justification=ov_just.get("justification", ""),
            as_is=as_is,
            to_be=to_be,
            process_steps=process_steps_dicts,
            input_requirements=input_reqs,
            detailed_steps=detailed,
            interface_requirements=apps,
            exception_handling=exceptions,
            flowchart_path=fc_path,
        )

        # Append screenshots if matched
        if matched_frames:
            valid_frames = [
                (p, d) for p, d in matched_frames if p and os.path.exists(p)
            ]
            if valid_frames:
                print(f"  Adding {len(valid_frames)} screenshots to Section 2.4...")
                gen = PDDGenerator()
                gen.append_frames_with_text(doc_path, valid_frames, detailed_steps=detailed)

        persistent = save_persistent_document(doc_path, project_name)

        # Token report
        tracker.print_report()
        tracker.save_csv(project_name)

        total = time.time() - t0
        print_pipeline_footer(
            persistent, project_name,
            {
                "Steps": len(steps),
                "Detailed": len(detailed),
                "Frames": len(matched_frames)
            },
            total
        )
        return doc_path

    def process_transcript_only(
        self,
        transcript_path: str,
        project_name: str = None,
        video_path: str = None
    ) -> Optional[str]:
        """Process transcript only — reuses full pipeline if video is provided."""
        if video_path and os.path.exists(video_path):
            return self.process(
                video_path, project_name, transcript_path=transcript_path
            )

        t0 = time.time()
        tracker = reset_tracker()
        gemini_client.set_tracker(tracker)

        print_pipeline_header("Transcript Only", project_name=project_name or "(auto-detect)")

        if not os.path.exists(transcript_path):
            print(f"Error: {transcript_path} not found")
            return None

        transcript = read_transcript(transcript_path)
        if not transcript:
            return None

        entities, detected = extract_entities_and_project(transcript)
        if not project_name:
            project_name = detected
        eh = build_entity_hint(entities)

        purpose = get_document_purpose_text(transcript, project_name, eh)
        ov_just = get_overview_and_justification(transcript, project_name, eh)
        as_is = get_as_is_process(transcript, project_name, eh)
        to_be = get_to_be_process(transcript, project_name, eh)
        steps = extract_process_steps(transcript, entities)
        detailed = get_detailed_process_steps(transcript, project_name, eh)
        input_reqs = get_input_requirements(transcript, project_name, eh)
        exceptions = get_exception_handling(transcript, project_name, eh)
        dot_code, apps = generate_dot_and_apps(steps, transcript, entities)

        if dot_code:
            save_dot_code(dot_code, project_name, self.output_dir)
        fc_path = generate_flowchart(dot_code, self.output_dir, project_name)

        process_steps_dicts = None
        if steps:
            process_steps_dicts = [
                {"number": i + 1, "description": s} for i, s in enumerate(steps)
            ]

        doc_path = build_document(
            project_name=project_name,
            output_dir=self.output_dir,
            purpose=purpose,
            overview=ov_just.get("overview", ""),
            justification=ov_just.get("justification", ""),
            as_is=as_is,
            to_be=to_be,
            process_steps=process_steps_dicts,
            input_requirements=input_reqs,
            detailed_steps=detailed,
            interface_requirements=apps,
            exception_handling=exceptions,
            flowchart_path=fc_path,
        )

        if detailed:
            from document.pdd_generator import PDDGenerator
            gen = PDDGenerator()
            gen.append_frames_with_text(doc_path, [], detailed_steps=detailed)

        persistent = save_persistent_document(doc_path, project_name)

        tracker.print_report()
        tracker.save_csv(project_name)

        total = time.time() - t0
        print_pipeline_footer(
            persistent, project_name,
            {"Steps": len(steps), "Detailed": len(detailed)},
            total
        )
        return doc_path

    @staticmethod
    def _ocr_available() -> bool:
        try:
            from video.ocr_engine import OCR_AVAILABLE
            return OCR_AVAILABLE
        except ImportError:
            return False