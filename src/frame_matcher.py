# src/frame_matcher.py

"""
Frame-to-Step matching using OCR + Text Similarity.
Modular design — each component is independent and replaceable.

Pipeline:
1. Extract candidate frames (more than needed)
2. OCR each frame to get on-screen text
3. Each frame also has transcript text from its timestamp
4. For each detailed step, score all frames using combined OCR + transcript similarity
5. Assign best-matching frame to each step (no duplicates)
"""

import os
import re
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter

# OCR import with graceful fallback
try:
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = (
        r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    )
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print(
        "    [FrameMatcher] pytesseract not installed. "
        "Install with: pip install pytesseract Pillow"
    )


# ============================================================
# MODULE 1: OCR Text Extraction
# ============================================================

def ocr_extract(frame_path: str) -> str:
    """
    Extract text from a single frame using Tesseract OCR.
    """
    if not OCR_AVAILABLE:
        return ""

    if not os.path.exists(frame_path):
        return ""

    try:
        image = Image.open(frame_path)
        text = pytesseract.image_to_string(
            image,
            config='--psm 6 --oem 3'
        )
        text = _clean_ocr_text(text)
        return text
    except Exception as e:
        print(f"    [OCR] Error on {os.path.basename(frame_path)}: {e}")
        return ""


def ocr_batch(frame_paths: List[str]) -> Dict[str, str]:
    """
    OCR multiple frames.
    """
    results = {}
    total = len(frame_paths)

    if not OCR_AVAILABLE:
        print("    [OCR] Tesseract not available, skipping OCR")
        return {fp: "" for fp in frame_paths}

    print(f"    [OCR] Processing {total} frames...")

    for i, fp in enumerate(frame_paths):
        text = ocr_extract(fp)
        results[fp] = text
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"    [OCR] {i+1}/{total} done")

    non_empty = sum(1 for v in results.values() if v.strip())
    print(f"    [OCR] {non_empty}/{total} frames had readable text")

    return results


def _clean_ocr_text(text: str) -> str:
    """Clean OCR output — remove noise, normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r'[|]{2,}', ' ', text)
    text = re.sub(r'[_]{3,}', ' ', text)
    text = re.sub(r'[~`]{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    lines = text.split('\n')
    lines = [l.strip() for l in lines if len(l.strip()) > 2]
    return ' '.join(lines).strip()


# ============================================================
# MODULE 2: Text Similarity Scoring
# ============================================================

# Synonyms/related words commonly found in automation processes
# Maps step description words → OCR/transcript equivalents
WORD_SYNONYMS = {
    'connect': {'login', 'log', 'sign', 'authenticate', 'access', 'open', 'launch'},
    'login': {'connect', 'sign', 'log', 'authenticate', 'credentials', 'password', 'username'},
    'navigate': {'go', 'open', 'click', 'select', 'menu', 'tab', 'page', 'home', 'dashboard'},
    'extract': {'download', 'export', 'get', 'pull', 'fetch', 'retrieve', 'save'},
    'download': {'extract', 'export', 'save', 'get', 'fetch'},
    'export': {'download', 'extract', 'save', 'csv', 'excel', 'file'},
    'validate': {'check', 'verify', 'confirm', 'ensure', 'review', 'inspect', 'compare'},
    'filter': {'sort', 'search', 'find', 'select', 'criteria', 'column', 'remove'},
    'process': {'handle', 'execute', 'perform', 'run', 'action', 'apply'},
    'generate': {'create', 'produce', 'build', 'make', 'report', 'output'},
    'report': {'generate', 'summary', 'output', 'results', 'details', 'log'},
    'update': {'modify', 'change', 'edit', 'set', 'status', 'save'},
    'credentials': {'login', 'password', 'username', 'user', 'authentication'},
    'application': {'portal', 'system', 'app', 'tool', 'platform', 'software', 'website'},
    'portal': {'application', 'website', 'system', 'dashboard', 'console', 'platform'},
    'user': {'account', 'member', 'person', 'employee', 'staff', 'admin'},
    'remove': {'delete', 'revoke', 'deactivate', 'disable', 'clear', 'unassign'},
    'license': {'licence', 'subscription', 'seat', 'entitlement', 'assignment'},
    'server': {'machine', 'host', 'instance', 'node', 'system'},
    'patch': {'update', 'fix', 'install', 'deploy', 'security'},
    'schedule': {'template', 'cron', 'trigger', 'timer', 'recurring'},
    'scan': {'check', 'detect', 'analyze', 'inspect', 'audit', 'assess'},
    'status': {'state', 'result', 'outcome', 'condition', 'progress'},
    'record': {'entry', 'row', 'item', 'data', 'line', 'record'},
    'click': {'press', 'select', 'tap', 'choose', 'button', 'hit'},
    'search': {'find', 'look', 'query', 'filter', 'locate'},
    'email': {'mail', 'notification', 'send', 'message', 'alert'},
    'active': {'enabled', 'running', 'online', 'live', 'operational'},
    'inactive': {'disabled', 'offline', 'dormant', 'idle', 'unused'},
}


def text_similarity(text1: str, text2: str) -> float:
    """
    Calculate word-overlap similarity between two texts.
    Returns 0.0 to 1.0.
    """
    if not text1 or not text2:
        return 0.0

    words1 = _extract_meaningful_words(text1)
    words2 = _extract_meaningful_words(text2)

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    if not union:
        return 0.0

    return len(intersection) / len(union)


def enhanced_similarity(text1: str, text2: str) -> float:
    """
    Enhanced similarity with synonym matching and importance weighting.
    
    Three signals combined:
    1. Direct word overlap (exact matches)
    2. Synonym matches (related words)
    3. Key noun matches (application names, specific terms get extra weight)
    """
    if not text1 or not text2:
        return 0.0

    words1 = _extract_meaningful_words(text1)
    words2 = _extract_meaningful_words(text2)

    if not words1 or not words2:
        return 0.0

    # Common automation words (low signal — appear everywhere)
    common_words = {
        'click', 'open', 'navigate', 'select', 'enter', 'system',
        'page', 'button', 'field', 'data', 'process', 'step',
        'application', 'portal', 'user', 'file', 'report',
        'status', 'update', 'check', 'verify', 'login', 'log',
        'the', 'for', 'and', 'with', 'from', 'into', 'using'
    }

    # ── Signal 1: Direct matches ──
    direct_matches = words1 & words2

    # ── Signal 2: Synonym matches ──
    synonym_matches = set()
    for w1 in words1:
        if w1 in direct_matches:
            continue
        synonyms = WORD_SYNONYMS.get(w1, set())
        for w2 in words2:
            if w2 in synonyms:
                synonym_matches.add(w1)
                break

    # ── Signal 3: Substring matches (catches partial words) ──
    # e.g., "management" in step matches "manage" in OCR
    substring_matches = set()
    for w1 in words1:
        if w1 in direct_matches or w1 in synonym_matches:
            continue
        if len(w1) >= 5:
            for w2 in words2:
                if len(w2) >= 5:
                    if w1[:5] == w2[:5]:  # First 5 chars match
                        substring_matches.add(w1)
                        break

    # ── Calculate weighted score ──
    all_words = words1 | words2
    if not all_words:
        return 0.0

    score = 0.0
    max_score = 0.0

    for word in all_words:
        # Weight: specific words count more than common ones
        if word in common_words:
            weight = 1.0
        elif len(word) >= 6:
            weight = 4.0  # Long specific words (application names, etc.)
        else:
            weight = 2.0

        max_score += weight

        if word in direct_matches:
            score += weight  # Full credit
        elif word in synonym_matches:
            score += weight * 0.7  # 70% credit for synonyms
        elif word in substring_matches:
            score += weight * 0.5  # 50% credit for substring

    if max_score == 0:
        return 0.0

    return score / max_score


def _extract_meaningful_words(text: str) -> Set[str]:
    """Extract meaningful words (lowercase, length > 2, no stopwords)."""
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are',
        'was', 'were', 'been', 'have', 'has', 'had', 'not', 'but',
        'all', 'can', 'will', 'would', 'should', 'could', 'may',
        'its', 'than', 'then', 'them', 'they', 'their', 'there',
        'each', 'which', 'when', 'where', 'how', 'who', 'whom',
        'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'over', 'some', 'any', 'also',
        'shall', 'must', 'using', 'based', 'upon'
    }

    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    return set(words) - stopwords


# ============================================================
# MODULE 3: Frame-Step Scoring
# ============================================================

def score_frame_against_step(
    frame_ocr_text: str,
    frame_transcript_text: str,
    step_description: str,
    ocr_weight: float = 0.6,
    transcript_weight: float = 0.4
) -> float:
    """
    Score how well a frame matches a step description.
    Combines OCR similarity and transcript similarity.
    Uses enhanced similarity with synonym matching.
    """
    ocr_score = 0.0
    transcript_score = 0.0

    if frame_ocr_text:
        ocr_score = enhanced_similarity(frame_ocr_text, step_description)

    if frame_transcript_text:
        transcript_score = enhanced_similarity(
            frame_transcript_text, step_description
        )

    # If OCR is empty, rely entirely on transcript
    if not frame_ocr_text:
        return transcript_score

    # If transcript is empty, rely entirely on OCR
    if not frame_transcript_text:
        return ocr_score

    # Combined score — take the HIGHER of weighted-average and max
    # This helps when one signal is strong but the other is zero
    weighted = (ocr_score * ocr_weight) + (transcript_score * transcript_weight)
    best_single = max(ocr_score, transcript_score)

    # Use 70% weighted + 30% best-single to avoid penalizing
    # when only one signal matches well
    return (weighted * 0.7) + (best_single * 0.3)


# ============================================================
# MODULE 4: Frame-Step Assignment (Main Matching Logic)
# ============================================================

# Minimum score to consider a match valid
# Below this, frame goes to chronological fallback
MIN_MATCH_SCORE = 0.02


def match_frames_to_steps(
    frame_candidates: List[Dict],
    detailed_steps: List[Dict],
    allow_reuse: bool = False
) -> List[Tuple[str, str]]:
    """
    Match frames to steps using OCR + transcript similarity.
    """
    if not frame_candidates or not detailed_steps:
        return []

    num_steps = len(detailed_steps)
    num_frames = len(frame_candidates)

    print(
        f"    [Matcher] Matching {num_frames} frames "
        f"to {num_steps} steps..."
    )

    # Debug: print sample OCR and step text for first few
    if frame_candidates and detailed_steps:
        sample_frame = frame_candidates[0]
        sample_step = detailed_steps[0]
        ocr_preview = sample_frame.get("ocr", "")[:80]
        trans_preview = sample_frame.get("transcript", "")[:80]
        step_preview = sample_step.get("description", "")[:80]
        print(f"    [Matcher] Sample OCR: '{ocr_preview}'")
        print(f"    [Matcher] Sample Transcript: '{trans_preview}'")
        print(f"    [Matcher] Sample Step: '{step_preview}'")

    # Build score matrix
    scores = []
    for step in detailed_steps:
        step_scores = []
        for frame in frame_candidates:
            score = score_frame_against_step(
                frame_ocr_text=frame.get("ocr", ""),
                frame_transcript_text=frame.get("transcript", ""),
                step_description=step.get("description", "")
            )
            step_scores.append(score)
        scores.append(step_scores)

    # Greedy assignment: for each step, pick highest-scoring unused frame
    assigned = []
    used_frames: Set[int] = set()

    for step_idx in range(num_steps):
        best_frame_idx = -1
        best_score = -1.0

        for frame_idx in range(num_frames):
            if not allow_reuse and frame_idx in used_frames:
                continue
            if scores[step_idx][frame_idx] > best_score:
                best_score = scores[step_idx][frame_idx]
                best_frame_idx = frame_idx

        if best_frame_idx >= 0 and best_score >= MIN_MATCH_SCORE:
            frame = frame_candidates[best_frame_idx]
            assigned.append((
                frame["path"],
                detailed_steps[step_idx].get("description", "")
            ))
            if not allow_reuse:
                used_frames.add(best_frame_idx)
            if best_score >= 0.08:
                print(
                    f"    [Matcher]   Step {step_idx+1}: "
                    f"frame {best_frame_idx} "
                    f"(score={best_score:.2f}) ✓"
                )
            else:
                print(
                    f"    [Matcher]   Step {step_idx+1}: "
                    f"frame {best_frame_idx} "
                    f"(score={best_score:.2f}) ~weak"
                )
        else:
            assigned.append((
                "",
                detailed_steps[step_idx].get("description", "")
            ))
            print(
                f"    [Matcher]   Step {step_idx+1}: "
                f"no match (best={best_score:.2f})"
            )

    matched = sum(1 for path, _ in assigned if path)
    print(
        f"    [Matcher] Matched: {matched}/{num_steps} steps "
        f"({num_steps - matched} unmatched)"
    )

    return assigned


def fill_unmatched_chronologically(
    assigned: List[Tuple[str, str]],
    frame_candidates: List[Dict],
    used_paths: Set[str] = None
) -> List[Tuple[str, str]]:
    """
    Fill unmatched steps with remaining frames in chronological order.
    """
    if used_paths is None:
        used_paths = set(path for path, _ in assigned if path)

    # Get unused frames sorted by timestamp
    unused = [
        f for f in frame_candidates
        if f["path"] not in used_paths
    ]
    unused.sort(key=lambda f: f.get("timestamp", 0))

    unused_iter = iter(unused)
    filled = []

    for path, desc in assigned:
        if path:
            filled.append((path, desc))
        else:
            next_frame = next(unused_iter, None)
            if next_frame:
                filled.append((next_frame["path"], desc))
                used_paths.add(next_frame["path"])
            else:
                filled.append(("", desc))

    filled_count = sum(1 for p, _ in filled if p)
    unfilled = sum(1 for p, _ in filled if not p)
    if unfilled > 0:
        print(
            f"    [Matcher] After chronological fill: "
            f"{filled_count} with frames, {unfilled} without"
        )

    return filled


# ============================================================
# MODULE 5: Complete Pipeline
# ============================================================

def match_pipeline(
    frame_candidates: List[Dict],
    detailed_steps: List[Dict],
    run_ocr: bool = True
) -> List[Tuple[str, str]]:
    """
    Complete frame matching pipeline.
    """
    if not frame_candidates:
        print("    [Matcher] No candidate frames provided")
        return [("", step.get("description", "")) for step in detailed_steps]

    if not detailed_steps:
        print("    [Matcher] No detailed steps provided")
        return []

    # Step 1: OCR
    if run_ocr and OCR_AVAILABLE:
        paths = [f["path"] for f in frame_candidates]
        ocr_results = ocr_batch(paths)
        for frame in frame_candidates:
            frame["ocr"] = ocr_results.get(frame["path"], "")
    else:
        if run_ocr and not OCR_AVAILABLE:
            print(
                "    [Matcher] OCR requested but Tesseract not available. "
                "Using transcript-only matching."
            )
        for frame in frame_candidates:
            frame["ocr"] = ""

    # Step 2: Score and match
    assigned = match_frames_to_steps(frame_candidates, detailed_steps)

    # Step 3: Fill unmatched chronologically
    assigned = fill_unmatched_chronologically(
        assigned, frame_candidates
    )

    return assigned


# ============================================================
# Utility: Build frame candidates from extraction results
# ============================================================

def build_candidates(
    frame_pairs: List[Tuple[str, str]],
    timestamps: List[float] = None
) -> List[Dict]:
    """
    Convert (path, transcript_text) pairs into candidate dicts.
    """
    candidates = []
    for i, (path, text) in enumerate(frame_pairs):
        ts = (
            timestamps[i] if timestamps and i < len(timestamps)
            else i * 10.0
        )
        candidates.append({
            "path": path,
            "transcript": text,
            "ocr": "",
            "timestamp": ts
        })
    return candidates