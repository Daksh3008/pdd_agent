# pipeline/video_pipeline.py

"""
Video Pipeline Orchestrator.
Silent screen recording (no audio) → Vision AI → PDD Document.

Flow:
1. Extract key frames via SSIM scene detection
2. OCR all frames (parallel)
3. Detect auth screens
4. Analyze transitions with vision AI (smart selection)
5. Synthesize PDD steps (parallel)
6. Generate document sections (parallel)
7. Generate flowchart
8. Annotate frames
9. Assemble PDD document
"""

import os
import time
import cv2
import concurrent.futures
from typing import Optional, Dict, List

from core.config import config
from core.gemini_client import gemini_client
from core.token_tracker import reset_tracker
from core.utils import detect_auth_screen, detect_operations_delta

from video.scene_detector import detect_scene_changes, get_video_info
from video.smart_sampler import select_key_frames, compute_target_frames
from video.ocr_engine import ocr_frame, OCR_AVAILABLE
from video.change_detector import detect_changes_between_frames
from video.frame_annotator import annotate_frame

from llm_tasks.vision_describer import analyze_transitions_smart, identify_application
from llm_tasks.step_synthesizer import synthesize_pdd_steps
from llm_tasks.document_sections import (
    generate_all_sections_parallel,
    generate_section_clarification_questions,
)
from llm_tasks.flowchart_dot import generate_flowchart_dot_from_steps

from pipeline.common import (
    save_persistent_document, save_flowchart_persistent,
    generate_flowchart, build_document,
    print_pipeline_header, print_pipeline_footer
)


def _compute_max_vision_calls(num_frames: int) -> int:
    """Scale vision calls with frame count."""
    base = config.llm.min_vision_calls
    scaled = (num_frames // 10) * config.llm.vision_calls_per_10_frames
    target = max(base, scaled)
    return min(target, config.llm.absolute_max_vision_calls)


def _parallel_ocr(frame_paths: List[str], with_boxes: bool = False) -> Dict[str, Dict]:
    """Perform OCR on multiple frames in parallel."""
    results = {}
    if not OCR_AVAILABLE:
        print("    [OCR] Tesseract not available, returning empty results")
        return {fp: {"text": "", "boxes": [], "word_count": 0} for fp in frame_paths}

    print(f"    [OCR] Processing {len(frame_paths)} frames in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_path = {
            executor.submit(ocr_frame, fp, with_boxes): fp
            for fp in frame_paths
        }
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results[path] = future.result()
            except Exception as e:
                print(f"    [OCR] Error on {os.path.basename(path)}: {e}")
                results[path] = {"text": "", "boxes": [], "word_count": 0}

    non_empty = sum(1 for v in results.values() if v["text"].strip())
    print(f"    [OCR] {non_empty}/{len(frame_paths)} frames had readable text")
    return results


def _extract_micro_frames(
    video_path: str, scene_changes: List[Dict],
    output_dir: str, fps: float
) -> List[Dict]:
    """Extract additional frames around each scene change."""
    if not scene_changes:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    enhanced_frames = list(scene_changes)
    existing_timestamps = set(round(sc.get("timestamp", 0), 1) for sc in scene_changes)

    offsets = [-0.3, 0.2]
    micro_count = 0

    for i, scene in enumerate(scene_changes):
        timestamp = scene.get("timestamp", 0)
        for offset in offsets:
            t = timestamp + offset
            if t < 0.1 or t > duration - 0.1:
                continue
            if any(abs(t - et) < 0.15 for et in existing_timestamps):
                continue

            frame_idx = int(t * fps)
            if frame_idx < 0 or frame_idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret and frame is not None:
                minutes = int(t // 60)
                seconds = int(t % 60)
                millis = int((t % 1) * 100)
                filename = f"frame_micro_{i:03d}_off{int(offset*10):+d}_{minutes}m{seconds:02d}s{millis:02d}.jpg"
                path = os.path.join(output_dir, filename)
                cv2.imwrite(path, frame)

                enhanced_frames.append({
                    "path": path,
                    "timestamp": t,
                    "frame_index": frame_idx,
                    "ssim_score": -1.0,
                    "is_micro_frame": True
                })
                existing_timestamps.add(round(t, 1))
                micro_count += 1

    cap.release()
    enhanced_frames.sort(key=lambda x: x.get("timestamp", 0))

    if micro_count > 0:
        print(f"    [MicroFrames] Added {micro_count} micro-frames")
    return enhanced_frames


class VideoPipeline:
    """Pipeline for silent screen recordings."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.paths.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.pending_questions: List[str] = []

    def process(
        self,
        video_path: str,
        project_name: str,
        ssim_threshold: float = None,
        max_frames: int = None,
        annotate: bool = None,
        enable_micro_frames: bool = True,
        clarification_qa: Optional[Dict[str, str]] = None,
        require_clarification: bool = True,
    ) -> Optional[str]:
        """
        Process a silent screen recording into a PDD document.

        Args:
            video_path: Path to video file.
            project_name: Project name (required).
            ssim_threshold: Scene detection sensitivity override.
            max_frames: Max key frames override.
            annotate: Enable screenshot annotation override.
            enable_micro_frames: Extract micro-frames around changes.
            clarification_qa: Human answers for LLM clarifying questions.
            require_clarification: If True, asks clarifying questions before section generation.

        Returns:
            Path to generated document, or None on failure.
        """
        t0 = time.time()
        tracker = reset_tracker()
        gemini_client.set_tracker(tracker)
        self.pending_questions = []

        ssim_threshold = ssim_threshold or config.frame.ssim_threshold
        annotate = annotate if annotate is not None else config.annotation.enabled

        print_pipeline_header(
            "Silent Screen Recording",
            video_path=video_path,
            project_name=project_name,
            extra_info={
                "OCR": "Available" if OCR_AVAILABLE else "Not available",
                "Micro-frames": "Enabled" if enable_micro_frames else "Disabled",
                "Workers": str(config.llm.max_workers)
            }
        )

        if not os.path.exists(video_path):
            print(f"Error: Video not found: {video_path}")
            return None

        video_info = get_video_info(video_path)
        duration = video_info['duration']
        fps = video_info['fps']
        target_frames = compute_target_frames(duration, max_frames)

        print(f"\n  Video: {duration:.0f}s ({duration/60:.1f}min), "
              f"{video_info['width']}x{video_info['height']}, {fps:.1f}fps")
        print(f"  Target frames: {target_frames}")

        frames_dir = os.path.join(self.output_dir, "frames")
        annotated_dir = os.path.join(self.output_dir, "annotated")
        os.makedirs(frames_dir, exist_ok=True)

        # ── PHASE 1: Frame Extraction ──
        print(f"\n{'='*40}")
        print("PHASE 1/5: Extracting key frames...")
        print(f"{'='*40}")
        t = time.time()

        scene_changes = detect_scene_changes(
            video_path, ssim_threshold=ssim_threshold
        )
        if enable_micro_frames and scene_changes:
            scene_changes = _extract_micro_frames(
                video_path, scene_changes, frames_dir, fps
            )
        key_frames = select_key_frames(
            scene_changes, output_dir=frames_dir,
            max_frames=max_frames, video_path=video_path,
            video_duration=duration
        )
        if not key_frames:
            print("Error: No key frames extracted")
            return None
        print(f"  ✓ {len(key_frames)} key frames ({time.time()-t:.0f}s)")

        # ── PHASE 2: OCR + Auth Detection ──
        print(f"\n{'='*40}")
        print("PHASE 2/5: OCR + Auth Detection...")
        print(f"{'='*40}")
        t = time.time()

        frame_paths = [kf["path"] for kf in key_frames]
        ocr_results = _parallel_ocr(frame_paths, with_boxes=annotate)

        auth_flags = []
        auth_count = 0
        for kf in key_frames:
            ocr_data = ocr_results.get(kf["path"], {})
            kf["ocr_text"] = ocr_data.get("text", "")
            kf["ocr_boxes"] = ocr_data.get("boxes", [])

            auth_info = detect_auth_screen(kf["ocr_text"])
            auth_flags.append(auth_info)
            if auth_info["is_auth"]:
                auth_count += 1

        print(f"  ✓ OCR complete ({time.time()-t:.0f}s)")
        if auth_count > 0:
            print(f"  🔐 Detected {auth_count} auth/login screens")

        # ── PHASE 3: Vision + Change Analysis ──
        print(f"\n{'='*40}")
        print("PHASE 3/5: Vision Analysis...")
        print(f"{'='*40}")
        t = time.time()

        max_vision = _compute_max_vision_calls(len(key_frames)) + auth_count
        print(f"  Vision budget: {max_vision} calls")

        # Temporarily override max vision calls
        original_max = config.llm.max_vision_calls
        config.llm.max_vision_calls = max_vision

        app_name = ""
        if key_frames:
            app_name = identify_application(key_frames[0]["path"])
            if app_name:
                print(f"  ✓ Application: {app_name}")

        change_data = detect_changes_between_frames(key_frames, ocr_results)

        ocr_diffs = []
        detected_operations = []
        for i, cd in enumerate(change_data):
            ocr_diffs.append(cd.get("text_diff", {}))
            before_kf = key_frames[i] if i < len(key_frames) else {}
            after_kf = key_frames[i + 1] if i + 1 < len(key_frames) else {}
            ops = detect_operations_delta(
                before_kf.get('ocr_text', ''),
                after_kf.get('ocr_text', ''), ""
            )
            detected_operations.append(ops)

        transitions = analyze_transitions_smart(
            key_frames, ocr_diffs, detected_operations,
            change_data, auth_flags=auth_flags
        )

        config.llm.max_vision_calls = original_max

        vision_used = sum(1 for tr in transitions if tr.get("used_vision"))
        print(f"  ✓ {len(transitions)} transitions, {vision_used} vision ({time.time()-t:.0f}s)")

        # ── PHASE 4: PDD Content ──
        print(f"\n{'='*40}")
        print("PHASE 4/5: Generating PDD content...")
        print(f"{'='*40}")
        t = time.time()

        pdd_steps = synthesize_pdd_steps(
            transitions, change_data, app_name=app_name
        )
        print(f"  ✓ {len(pdd_steps)} PDD steps")

        for i, s in enumerate(pdd_steps[:5]):
            desc_preview = s['description'][:70]
            if len(s['description']) > 70:
                desc_preview += "..."
            print(f"    {s['number']}. {desc_preview}")
        if len(pdd_steps) > 5:
            print(f"    ... +{len(pdd_steps)-5} more")

        step_descriptions = [s["description"] for s in pdd_steps]
        vision_descriptions = [
            kf.get("vision_description", kf.get("ocr_text", ""))
            for kf in key_frames
        ]

        clean_qa = {}
        if clarification_qa:
            for q, a in clarification_qa.items():
                q_clean = (q or "").strip()
                a_clean = (a or "").strip()
                if q_clean and a_clean:
                    clean_qa[q_clean] = a_clean

        if require_clarification and not clean_qa:
            self.pending_questions = generate_section_clarification_questions(
                project_name=project_name,
                app_name=app_name,
                step_descriptions=step_descriptions,
                vision_descriptions=vision_descriptions,
            )
            if self.pending_questions:
                print("  ! Clarification required before generating sections")
                for i, q in enumerate(self.pending_questions, start=1):
                    print(f"    Q{i}. {q}")
                return None

        sections = generate_all_sections_parallel(
            project_name,
            app_name,
            step_descriptions,
            vision_descriptions,
            clarification_qa=clean_qa,
        )

        purpose = sections.get("purpose") or ""
        ov_just = sections.get("overview_justification") or {}
        as_is = sections.get("as_is") or ""
        to_be = sections.get("to_be") or ""
        input_reqs = sections.get("prerequisites") or []
        exceptions = sections.get("exceptions") or []
        interfaces = sections.get("interfaces") or []

        # Flowchart
        dot_code = generate_flowchart_dot_from_steps(pdd_steps, project_name)
        fc_path = generate_flowchart(dot_code, self.output_dir, project_name)

        print(f"  ✓ Phase 4 complete ({time.time()-t:.0f}s)")

        # ── PHASE 5: Document Assembly ──
        print(f"\n{'='*40}")
        print("PHASE 5/5: Generating document...")
        print(f"{'='*40}")
        t = time.time()

        annotated_frames = {}
        if annotate and pdd_steps:
            for step in pdd_steps:
                step_num = step["number"]
                frame_path = step.get("frame_after_path", "")
                change_region = step.get("change_region")
                label = step["description"][:50]
                if frame_path and os.path.exists(frame_path):
                    ann_path = annotate_frame(
                        frame_path=frame_path,
                        output_dir=annotated_dir,
                        step_number=step_num,
                        change_region=change_region,
                        action_label=label,
                        enabled=annotate
                    )
                    if ann_path and os.path.exists(ann_path):
                        annotated_frames[step_num] = ann_path
            print(f"  ✓ {len(annotated_frames)} frames annotated")

        doc_path = build_document(
            project_name=project_name,
            output_dir=self.output_dir,
            purpose=purpose,
            overview=ov_just.get("overview", "") if isinstance(ov_just, dict) else "",
            justification=ov_just.get("justification", "") if isinstance(ov_just, dict) else "",
            as_is=as_is,
            to_be=to_be,
            process_steps=pdd_steps,
            input_requirements=input_reqs,
            detailed_steps=pdd_steps,
            interface_requirements=interfaces,
            exception_handling=exceptions,
            flowchart_path=fc_path,
            app_name=app_name,
            annotated_frames=annotated_frames
        )

        persistent = save_persistent_document(doc_path, project_name)

        tracker.print_report()
        tracker.save_csv(project_name)

        total = time.time() - t0

        all_ops = set()
        for ops_list in detected_operations:
            for op in ops_list:
                if op.get("confidence", 0) >= 0.7:
                    all_ops.add(op["display_name"])

        stats = {
            "Steps": len(pdd_steps),
            "Frames": len(key_frames),
            "Vision calls": vision_used,
            "Transitions": len(transitions),
        }
        if auth_count > 0:
            stats["Auth screens"] = auth_count
        if all_ops:
            stats["Operations"] = ', '.join(sorted(all_ops))

        print_pipeline_footer(persistent, project_name, stats, total)
        return doc_path