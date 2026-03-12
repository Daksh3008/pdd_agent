# src/pdd_agent.py

"""
PDD Agent - Main Orchestrator.
Integrates OCR-based frame matching and token tracking.
"""

import os
import time
import shutil
from typing import Optional, List, Tuple, Dict

from config import path_config, whisper_config, doc_config
from llm_client import llm_client
from token_tracker import TokenTracker, reset_tracker
from video_to_audio import convert_video_to_audio
from transcribe_audio import transcribe_audio, read_transcript
from llm_tasks import (
    extract_entities_and_project,
    get_document_purpose_text,
    get_overview_and_justification,
    get_as_is_process,
    get_to_be_process,
    extract_process_steps,
    get_input_requirements,
    get_interface_requirements,
    get_exception_handling,
    get_detailed_process_steps,
    generate_dot_and_apps,
    identify_key_timestamps,
    paraphrase_batch,
    _build_entity_hint
)
from flowchart_generator import generate_flowchart_from_dot
from frame_extractor import (
    extract_frames_with_transcripts,
    extract_frame,
    extract_evenly_spaced_frames,
    get_video_duration,
    extract_timestamps_from_transcript
)
from frame_matcher import match_pipeline, build_candidates, OCR_AVAILABLE
from pdd_document import PDDGenerator


class PDDAgent:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or path_config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if not llm_client.is_available():
            print("⚠ Ollama LLM not available.")

    def _save_persistent(self, doc_path, project_name):
        d = path_config.output_dir
        os.makedirs(d, exist_ok=True)
        safe = "".join(
            c for c in project_name if c.isalnum() or c in (' ', '-', '_')
        ).strip()[:50] or "PDD"
        doc_type = doc_config.document_type
        path = os.path.join(d, f"{safe}_{doc_type}.docx")
        c = 1
        base = path
        while os.path.exists(path):
            path = f"{base.rsplit('.', 1)[0]}_{c}.docx"
            c += 1
        shutil.copy2(doc_path, path)
        print(f"  📁 Saved: {path}")
        return path

    def _save_dot(self, dot_code, project_name):
        d = path_config.output_dir
        os.makedirs(d, exist_ok=True)
        safe = "".join(
            c for c in project_name if c.isalnum() or c in (' ', '-', '_')
        ).strip()[:50] or "fc"
        path = os.path.join(d, f"{safe}_flowchart_dotcode.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(dot_code)
        print(f"  📁 DOT: {path}")
        local = os.path.join(self.output_dir, f"{safe}_flowchart_dotcode.txt")
        if local != path:
            with open(local, 'w', encoding='utf-8') as f:
                f.write(dot_code)

    def _extract_candidate_frames(
        self, video_path, transcript_path, transcript, frames_dir, num_target
    ):
        """
        Extract a POOL of candidate frames (more than needed).
        These will be matched to steps by the frame_matcher.
        
        Returns list of (frame_path, transcript_text, timestamp) tuples.
        """
        print(f"    Extracting candidate frame pool "
              f"(target: {num_target * 3} candidates for {num_target} steps)...")

        os.makedirs(frames_dir, exist_ok=True)
        candidates = []
        used_timestamps = set()

        # Pool size: 3x the steps needed (gives matcher more choices)
        pool_size = min(num_target * 3, 60)

        # Source 1: LLM-identified key moments
        key_moments = identify_key_timestamps(transcript, transcript_path)
        if key_moments:
            print(f"    Source 1: {len(key_moments)} LLM key moments")
            for m in key_moments:
                ts = m["timestamp"]
                if any(abs(ts - used) < 2.0 for used in used_timestamps):
                    continue
                fp = extract_frame(video_path, ts, frames_dir)
                if fp:
                    candidates.append((
                        fp, m.get("description", ""), ts
                    ))
                    used_timestamps.add(ts)

        # Source 2: Keyword-based from transcript
        if len(candidates) < pool_size:
            print(f"    Source 2: Keyword-based extraction...")
            keyword_pairs = extract_frames_with_transcripts(
                video_path, transcript_path, frames_dir
            )
            existing_paths = set(fp for fp, _, _ in candidates)
            for fp, desc in keyword_pairs:
                if fp not in existing_paths and len(candidates) < pool_size:
                    candidates.append((fp, desc, -1))
                    existing_paths.add(fp)

        # Source 3: Evenly spaced to fill pool
        remaining = pool_size - len(candidates)
        if remaining > 0:
            print(f"    Source 3: {remaining} evenly-spaced frames...")
            duration = get_video_duration(video_path)
            if duration > 0:
                start_t = duration * 0.02
                end_t = duration * 0.98
                interval = (end_t - start_t) / (remaining + 1)

                # Build transcript lookup for timestamp → text
                transcript_lookup = {}
                try:
                    import re as _re
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            m = _re.match(
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
                        # Find closest transcript text
                        closest_text = ""
                        min_diff = float('inf')
                        for t_ts, t_text in transcript_lookup.items():
                            diff = abs(t_ts - ts)
                            if diff < min_diff:
                                min_diff = diff
                                closest_text = t_text
                        candidates.append((fp, closest_text, ts))
                        used_timestamps.add(ts)

        print(f"    Total candidate frames: {len(candidates)}")
        return candidates

    def process_video(
        self, video_path, project_name=None,
        whisper_model=None, transcript_path=None
    ):
        t0 = time.time()

        # Initialize token tracker
        tracker = reset_tracker()
        llm_client.set_tracker(tracker)

        print("=" * 60)
        print("PDD Agent")
        print(f"  Model: {llm_client.config.model}")
        print(f"  Document: {doc_config.document_type_full}")
        print(f"  OCR: {'Available' if OCR_AVAILABLE else 'Not available'}")
        print("=" * 60)

        if not os.path.exists(video_path):
            print(f"Error: {video_path} not found")
            return None

        # Transcript
        if transcript_path and os.path.exists(transcript_path):
            print(f"\n[1/6] Using transcript: {transcript_path}")
        else:
            print("\n[1/6] Extracting audio...")
            audio = convert_video_to_audio(video_path, self.output_dir)
            if not audio:
                return None
            print("[2/6] Transcribing...")
            transcript_path = transcribe_audio(
                audio, self.output_dir,
                model_name=whisper_model or whisper_config.model_name
            )
            if not transcript_path:
                return None

        transcript = read_transcript(transcript_path)
        if not transcript:
            return None
        print(f"  Transcript: {len(transcript):,} chars")

        # ── LLM Extraction ──
        print("\n[3/6] LLM extraction...")

        t = time.time()
        entities, detected = extract_entities_and_project(transcript)
        if not project_name:
            project_name = detected
        eh = _build_entity_hint(entities)
        print(f"  ✓ Project: {project_name} ({time.time()-t:.0f}s)")

        t = time.time()
        purpose = get_document_purpose_text(transcript, project_name, eh)
        print(f"  ✓ Purpose ({time.time()-t:.0f}s)")

        t = time.time()
        ov_just = get_overview_and_justification(
            transcript, project_name, eh
        )
        print(f"  ✓ Overview+Justification ({time.time()-t:.0f}s)")

        t = time.time()
        as_is = get_as_is_process(transcript, project_name, eh)
        print(f"  ✓ As-Is ({time.time()-t:.0f}s)")

        t = time.time()
        to_be = get_to_be_process(transcript, project_name, eh)
        print(f"  ✓ To-Be ({time.time()-t:.0f}s)")

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

        # ── Flowchart ──
        print("\n[4/6] Flowchart...")
        if dot_code:
            self._save_dot(dot_code, project_name)
        fc = os.path.join(self.output_dir, "flowchart")
        if dot_code:
            r = generate_flowchart_from_dot(dot_code, fc)
            fc = r if r else ""
        else:
            fc = ""

        # ── Screenshots + Matching ──
        print("\n[5/6] Screenshots & matching...")
        matched_frames = []

        if os.path.exists(video_path) and detailed:
            # Step 1: Extract candidate pool
            num_steps = len(detailed)
            raw_candidates = self._extract_candidate_frames(
                video_path, transcript_path, transcript,
                os.path.join(self.output_dir, "frames"),
                num_target=num_steps
            )

            if raw_candidates:
                # Step 2: Build candidate dicts for matcher
                frame_pairs = [
                    (fp, desc) for fp, desc, _ in raw_candidates
                ]
                timestamps = [
                    ts for _, _, ts in raw_candidates
                ]
                candidates = build_candidates(frame_pairs, timestamps)

                # Step 3: Run matching pipeline (OCR + similarity)
                print(f"  Running frame matcher "
                      f"({'with OCR' if OCR_AVAILABLE else 'transcript-only'})...")
                matched_frames = match_pipeline(
                    candidates, detailed,
                    run_ocr=OCR_AVAILABLE
                )

                # Step 4: Paraphrase descriptions for matched frames
                if matched_frames:
                    texts = [desc for _, desc in matched_frames if desc]
                    if texts:
                        print(f"  Improving {len(texts)} descriptions...")
                        improved = paraphrase_batch(texts)
                        improved_iter = iter(improved)
                        matched_frames = [
                            (path, next(improved_iter) if desc else "")
                            for path, desc in matched_frames
                        ]

        # ── Generate Document ──
        print("\n[6/6] Generating document...")
        safe = "".join(
            c for c in project_name if c.isalnum() or c in (' ', '-', '_')
        ).strip()[:50] or "PDD"
        doc_type = doc_config.document_type
        doc_path = os.path.join(self.output_dir, f"{safe}_{doc_type}.docx")

        gen = PDDGenerator()
        gen.generate(
            project_name=project_name,
            process_summary=to_be,
            inputs_outputs=as_is,
            flowchart_path=fc,
            document_purpose=purpose,
            applications=apps,
            process_steps=steps,
            output_path=doc_path,
            overview=ov_just.get("overview", ""),
            justification=ov_just.get("justification", ""),
            as_is=as_is,
            to_be=to_be,
            input_requirements=input_reqs,
            exception_handling=exceptions,
            detailed_steps=detailed
        )

        # Append screenshots
        if matched_frames:
            valid_frames = [
                (p, d) for p, d in matched_frames if p and os.path.exists(p)
            ]
            print(
                f"  Adding {len(valid_frames)} matched screenshots "
                f"to Section 2.4..."
            )
            gen.append_frames_with_text(
                doc_path, valid_frames, detailed_steps=detailed
            )
        elif detailed:
            print(f"  Adding {len(detailed)} detailed steps (no screenshots)...")
            gen.append_frames_with_text(
                doc_path, [], detailed_steps=detailed
            )

        persistent = self._save_persistent(doc_path, project_name)

        # ── Token Report ──
        tracker.print_report()
        tracker.save_csv(project_name)

        total = time.time() - t0
        print(f"\n{'='*60}")
        print(f"✓ Complete! {persistent}")
        print(
            f"  Steps: {len(steps)} | Detailed: {len(detailed)} | "
            f"Frames: {len(matched_frames)}"
        )
        print(f"  Time: {total/60:.1f}min")
        print(f"{'='*60}")
        return doc_path

    def process_transcript(
        self, transcript_path, project_name=None, video_path=None
    ):
        """Process transcript — reuses process_video if video available."""
        if video_path and os.path.exists(video_path):
            return self.process_video(
                video_path, project_name,
                transcript_path=transcript_path
            )

        t0 = time.time()

        # Initialize tracker
        tracker = reset_tracker()
        llm_client.set_tracker(tracker)

        print("=" * 60)
        print("PDD Agent - Transcript Only")
        print("=" * 60)

        if not os.path.exists(transcript_path):
            return None
        transcript = read_transcript(transcript_path)
        if not transcript:
            return None

        entities, detected = extract_entities_and_project(transcript)
        if not project_name:
            project_name = detected
        eh = _build_entity_hint(entities)

        purpose = get_document_purpose_text(transcript, project_name, eh)
        ov_just = get_overview_and_justification(
            transcript, project_name, eh
        )
        as_is = get_as_is_process(transcript, project_name, eh)
        to_be = get_to_be_process(transcript, project_name, eh)
        steps = extract_process_steps(transcript, entities)
        detailed = get_detailed_process_steps(transcript, project_name, eh)
        input_reqs = get_input_requirements(transcript, project_name, eh)
        exceptions = get_exception_handling(transcript, project_name, eh)
        dot_code, apps = generate_dot_and_apps(steps, transcript, entities)

        if dot_code:
            self._save_dot(dot_code, project_name)
        fc = os.path.join(self.output_dir, "flowchart")
        fc = (
            generate_flowchart_from_dot(dot_code, fc) if dot_code else ""
        ) or ""

        safe = "".join(
            c for c in project_name if c.isalnum() or c in (' ', '-', '_')
        ).strip()[:50] or "PDD"
        doc_type = doc_config.document_type
        doc_path = os.path.join(self.output_dir, f"{safe}_{doc_type}.docx")

        gen = PDDGenerator()
        gen.generate(
            project_name=project_name,
            process_summary=to_be,
            inputs_outputs=as_is,
            flowchart_path=fc,
            document_purpose=purpose,
            applications=apps,
            process_steps=steps,
            output_path=doc_path,
            overview=ov_just.get("overview", ""),
            justification=ov_just.get("justification", ""),
            as_is=as_is,
            to_be=to_be,
            input_requirements=input_reqs,
            exception_handling=exceptions,
            detailed_steps=detailed
        )

        if detailed:
            gen.append_frames_with_text(
                doc_path, [], detailed_steps=detailed
            )

        persistent = self._save_persistent(doc_path, project_name)

        # Token report
        tracker.print_report()
        tracker.save_csv(project_name)

        print(f"\n✓ Done: {persistent} ({(time.time()-t0)/60:.1f}min)")
        return doc_path


def generate_pdd_from_video(
    video_path, project_name=None, output_dir=None, transcript_path=None
):
    return PDDAgent(output_dir).process_video(
        video_path, project_name, transcript_path=transcript_path
    )

def generate_pdd_from_transcript(
    transcript_path, project_name=None, output_dir=None, video_path=None
):
    return PDDAgent(output_dir).process_transcript(
        transcript_path, project_name, video_path
    )