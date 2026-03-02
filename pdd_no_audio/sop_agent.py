# pdd_no_audio/sop_agent.py

"""
PDD Agent for Silent Screen Recordings — Main Orchestrator.
Scales frame extraction and vision calls with video duration.
Auth-aware: detects login/logout screens and ensures proper documentation.
Optimized: parallel OCR, parallel step synthesis, parallel section generation.
"""

import os
import time
import shutil
import concurrent.futures
from typing import Optional, Dict, List

from pdd_no_audio.config import (
    path_config, doc_config, frame_config,
    annotation_config, vision_config, text_config,
    flowchart_config, llm_params
)
from pdd_no_audio.clients.text_llm import text_client
from pdd_no_audio.clients.vision_llm import vision_client
from pdd_no_audio.token_tracker import TokenTracker, reset_tracker
from pdd_no_audio.frame_extraction.scene_detector import detect_scene_changes, get_video_info
from pdd_no_audio.frame_extraction.smart_sampler import select_key_frames, compute_target_frames
from pdd_no_audio.frame_analysis.ocr_engine import ocr_frame, OCR_AVAILABLE
from pdd_no_audio.frame_analysis.vision_describer import (
    analyze_transitions_smart, identify_application
)
from pdd_no_audio.frame_analysis.change_detector import detect_changes_between_frames
from pdd_no_audio.frame_analysis.frame_annotator import annotate_frame
from pdd_no_audio.llm_tasks.step_synthesizer import synthesize_pdd_steps
from pdd_no_audio.llm_tasks.sop_sections import (
    generate_all_sections_parallel,
    generate_flowchart_dot
)
from pdd_no_audio.pdd_document import PDDGenerator
from pdd_no_audio.utils import detect_operations, detect_auth_screen

try:
    import sys
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    if src_path not in sys.path:
        sys.path.insert(0, os.path.abspath(src_path))
    from flowchart_generator import generate_flowchart_from_dot
    FLOWCHART_AVAILABLE = True
except ImportError:
    FLOWCHART_AVAILABLE = False
    print("    [PDDAgent] Flowchart generator not available")


def _compute_max_vision_calls(num_frames: int) -> int:
    """Scale vision calls with frame count."""
    base = llm_params.min_vision_calls
    scaled = (num_frames // 10) * llm_params.vision_calls_per_10_frames
    target = max(base, scaled)
    target = min(target, llm_params.absolute_max_vision_calls)
    return target


def _parallel_ocr(frame_paths: List[str], with_boxes: bool = False) -> Dict[str, Dict]:
    """Perform OCR on multiple frames in parallel using threads."""
    results = {}
    if not OCR_AVAILABLE:
        print("    [OCR] Tesseract not available, returning empty results")
        return {fp: {"text": "", "boxes": [], "word_count": 0} for fp in frame_paths}

    print(f"    [OCR] Processing {len(frame_paths)} frames in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_path = {executor.submit(ocr_frame, fp, with_boxes): fp for fp in frame_paths}
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


class SOPAgent:
    """PDD Agent for silent screen recordings."""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or path_config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_persistent(self, doc_path: str, project_name: str) -> str:
        d = path_config.output_dir
        os.makedirs(d, exist_ok=True)
        safe = "".join(
            c for c in project_name if c.isalnum() or c in (' ', '-', '_')
        ).strip()[:50] or "PDD"
        path = os.path.join(d, f"{safe}_PDD.docx")
        c = 1
        base = path
        while os.path.exists(path):
            path = f"{base.rsplit('.', 1)[0]}_{c}.docx"
            c += 1
        shutil.copy2(doc_path, path)
        print(f"  📁 Saved: {path}")
        return path

    def _save_flowchart_persistent(
        self, flowchart_path: str, dot_code: str, project_name: str
    ):
        d = path_config.output_dir
        os.makedirs(d, exist_ok=True)
        safe = "".join(
            c for c in project_name if c.isalnum() or c in (' ', '-', '_')
        ).strip()[:50] or "flowchart"

        if flowchart_path and os.path.exists(flowchart_path):
            png_dest = os.path.join(d, f"{safe}_flowchart.png")
            c = 1
            while os.path.exists(png_dest):
                png_dest = os.path.join(d, f"{safe}_flowchart_{c}.png")
                c += 1
            shutil.copy2(flowchart_path, png_dest)
            print(f"  📁 Flowchart PNG: {png_dest}")

        if dot_code:
            dot_dest = os.path.join(d, f"{safe}_flowchart.dot")
            c = 1
            while os.path.exists(dot_dest):
                dot_dest = os.path.join(d, f"{safe}_flowchart_{c}.dot")
                c += 1
            with open(dot_dest, 'w', encoding='utf-8') as f:
                f.write(dot_code)
            print(f"  📁 Flowchart DOT: {dot_dest}")

    def process_video(
        self,
        video_path: str,
        project_name: str,
        ssim_threshold: float = None,
        max_frames: int = None,
        annotate: bool = None
    ) -> Optional[str]:
        t0 = time.time()

        tracker = reset_tracker()
        text_client.set_tracker(tracker)
        vision_client.set_tracker(tracker)

        ssim_threshold = ssim_threshold or frame_config.ssim_threshold
        annotate = annotate if annotate is not None else annotation_config.enabled

        print("=" * 65)
        print("PDD Agent — Silent Screen Recording")
        print(f"  Video: {video_path}")
        print(f"  Project: {project_name}")
        print(f"  Document: {doc_config.document_type_full}")
        print(f"  Vision Model: {vision_config.model}")
        print(f"  Text Model: {text_config.model}")
        print(f"  OCR: {'Available' if OCR_AVAILABLE else 'Not available'}")
        print(f"  Parallel workers: {llm_params.text_llm_workers}")
        print("=" * 65)

        if not os.path.exists(video_path):
            print(f"Error: Video not found: {video_path}")
            return None

        video_info = get_video_info(video_path)
        duration = video_info['duration']
        duration_min = duration / 60

        target_frames = compute_target_frames(duration, max_frames)

        print(f"\n  Video: {duration:.0f}s ({duration_min:.1f}min), "
              f"{video_info['width']}x{video_info['height']}, "
              f"{video_info['fps']:.1f}fps")
        print(f"  Target frames: {target_frames} "
              f"({frame_config.frames_per_minute}/min × {duration_min:.1f}min)")

        frames_dir = os.path.join(self.output_dir, "frames")
        annotated_dir = os.path.join(self.output_dir, "annotated")

        # ── PHASE 1: Frame Extraction ──
        print(f"\n{'='*40}")
        print("PHASE 1/5: Extracting key frames...")
        print(f"{'='*40}")
        t = time.time()

        scene_changes = detect_scene_changes(video_path, ssim_threshold=ssim_threshold)
        key_frames = select_key_frames(
            scene_changes, output_dir=frames_dir,
            max_frames=max_frames, video_path=video_path,
            video_duration=duration
        )
        if not key_frames:
            print("Error: No key frames extracted")
            return None
        print(f"  ✓ {len(key_frames)} key frames ({time.time()-t:.0f}s)")

        # ── PHASE 2: OCR (parallel) + Auth Detection ──
        print(f"\n{'='*40}")
        print("PHASE 2/5: Reading screen text (OCR) + Auth Detection...")
        print(f"{'='*40}")
        t = time.time()

        frame_paths = [kf["path"] for kf in key_frames]
        # Only request boxes if annotation is enabled
        ocr_results = _parallel_ocr(frame_paths, with_boxes=annotate)

        auth_flags = []
        auth_count = 0
        for kf in key_frames:
            ocr_data = ocr_results.get(kf["path"], {})
            kf["ocr_text"] = ocr_data.get("text", "")
            kf["ocr_boxes"] = ocr_data.get("boxes", [])

            # Detect auth screens from OCR
            auth_info = detect_auth_screen(kf["ocr_text"])
            auth_flags.append(auth_info)
            if auth_info["is_auth"]:
                auth_count += 1

        print(f"  ✓ OCR complete ({time.time()-t:.0f}s)")
        if auth_count > 0:
            print(f"  🔐 Detected {auth_count} auth/login screens")
            for i, af in enumerate(auth_flags):
                if af["is_auth"]:
                    print(f"    Frame {i}: {af['auth_type']} "
                          f"(confidence={af['confidence']:.2f}, "
                          f"indicators={af['indicators'][:5]})")

        # ── PHASE 3: Vision + Change Analysis ──
        print(f"\n{'='*40}")
        print("PHASE 3/5: Analyzing screens (Smart Vision)...")
        print(f"{'='*40}")
        t = time.time()

        max_vision = _compute_max_vision_calls(len(key_frames))
        # Add budget for auth screens (they always need vision)
        max_vision += auth_count
        print(f"  Vision budget: {max_vision} calls "
              f"({len(key_frames)} frames, +{auth_count} auth)")

        original_max = llm_params.max_vision_calls
        llm_params.max_vision_calls = max_vision

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
            ops = detect_operations(
                f"{before_kf.get('ocr_text', '')} {after_kf.get('ocr_text', '')}",
                "", ""
            )
            detected_operations.append(ops)

        transitions = analyze_transitions_smart(
            key_frames, ocr_diffs, detected_operations, change_data,
            auth_flags=auth_flags
        )

        llm_params.max_vision_calls = original_max

        vision_used = sum(1 for t in transitions if t.get("used_vision"))
        print(f"  ✓ Analysis complete ({time.time()-t:.0f}s)")
        print(f"    {len(transitions)} transitions, {vision_used} used vision")

        all_ops = set()
        for ops_list in detected_operations:
            for op in ops_list:
                all_ops.add(op["display_name"])
        if all_ops:
            print(f"    Operations: {', '.join(sorted(all_ops))}")

        # ── PHASE 4: PDD Content (parallel steps + parallel sections) ──
        print(f"\n{'='*40}")
        print("PHASE 4/5: Generating PDD content (parallel)...")
        print(f"{'='*40}")
        t = time.time()

        # 4a. Synthesize steps (parallel)
        pdd_steps = synthesize_pdd_steps(
            transitions, change_data, app_name=app_name
        )
        print(f"  ✓ {len(pdd_steps)} PDD steps")

        for i, s in enumerate(pdd_steps[:5]):
            ops = s.get("operations_detected", [])
            auth = s.get("auth_info", {})
            op_str = f" [{', '.join(op['display_name'] for op in ops)}]" if ops else ""
            auth_str = f" 🔐{auth['auth_type']}" if auth.get("is_auth") else ""
            print(f"    {s['number']}. {s['description'][:70]}{op_str}{auth_str}")
        if len(pdd_steps) > 5:
            print(f"    ... +{len(pdd_steps)-5} more")

        step_descriptions = [s["description"] for s in pdd_steps]
        vision_descriptions = [
            kf.get("vision_description", kf.get("ocr_text", ""))
            for kf in key_frames
        ]

        # 4b. Generate all sections in parallel
        sections = generate_all_sections_parallel(
            project_name, app_name, step_descriptions, vision_descriptions
        )

        purpose = sections.get("purpose") or ""
        ov_just = sections.get("overview_justification") or {"overview": "", "justification": ""}
        as_is = sections.get("as_is") or ""
        to_be = sections.get("to_be") or ""
        input_reqs = sections.get("prerequisites") or []
        exceptions = sections.get("exceptions") or []
        interfaces = sections.get("interfaces") or []

        print(f"  ✓ All sections generated")

        # 4c. Flowchart
        dot_code = generate_flowchart_dot(pdd_steps, project_name)
        flowchart_path = ""
        if dot_code and FLOWCHART_AVAILABLE:
            fc_output = os.path.join(self.output_dir, "flowchart")
            result = generate_flowchart_from_dot(dot_code, fc_output)
            if result:
                flowchart_path = result
                print(f"  ✓ Flowchart generated")
                self._save_flowchart_persistent(flowchart_path, dot_code, project_name)
        elif dot_code:
            self._save_flowchart_persistent("", dot_code, project_name)

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
                        frame_path=frame_path, output_dir=annotated_dir,
                        step_number=step_num, change_region=change_region,
                        action_label=label, enabled=annotate
                    )
                    if ann_path and os.path.exists(ann_path):
                        annotated_frames[step_num] = ann_path
            print(f"  ✓ {len(annotated_frames)} frames annotated")

        safe = "".join(
            c for c in project_name if c.isalnum() or c in (' ', '-', '_')
        ).strip()[:50] or "PDD"
        doc_path = os.path.join(self.output_dir, f"{safe}_PDD.docx")

        gen = PDDGenerator()
        gen.generate(
            project_name=project_name, app_name=app_name,
            document_purpose=purpose,
            overview=ov_just.get("overview", "") if isinstance(ov_just, dict) else "",
            justification=ov_just.get("justification", "") if isinstance(ov_just, dict) else "",
            as_is=as_is, to_be=to_be,
            process_steps=pdd_steps,
            input_requirements=input_reqs,
            detailed_steps=pdd_steps,
            interface_requirements=interfaces,
            exception_handling=exceptions,
            flowchart_path=flowchart_path,
            output_path=doc_path,
            annotated_frames=annotated_frames
        )

        persistent = self._save_persistent(doc_path, project_name)
        tracker.print_report()
        tracker.save_csv(project_name)

        total = time.time() - t0
        print(f"\n{'='*65}")
        print(f"✓ PDD Complete!")
        print(f"  Document: {persistent}")
        print(f"  Steps: {len(pdd_steps)} | Frames: {len(key_frames)}")
        print(f"  Vision calls: {vision_used} | Transitions: {len(transitions)}")
        if auth_count > 0:
            print(f"  Auth screens: {auth_count}")
        if all_ops:
            print(f"  Operations: {', '.join(sorted(all_ops))}")
        print(f"  Time: {total/60:.1f} minutes")
        print(f"{'='*65}")

        return doc_path


def generate_sop_from_video(
    video_path: str, project_name: str,
    output_dir: str = None, ssim_threshold: float = None,
    max_frames: int = None, annotate: bool = None
) -> Optional[str]:
    return SOPAgent(output_dir).process_video(
        video_path=video_path, project_name=project_name,
        ssim_threshold=ssim_threshold, max_frames=max_frames, annotate=annotate
    )