# pdd_no_audio/frame_analysis/vision_describer.py

"""
Vision model (llama3.2-vision:11b) frame description.

OPTIMIZATION STRATEGY:
- Extract ALL frames (40) for full context
- OCR ALL frames (fast — seconds each)
- Pixel diff ALL pairs (fast — milliseconds)
- Vision model ONLY for transitions where OCR cannot explain the change
- Text LLM gets BOTH vision descriptions AND OCR text for synthesis

This keeps full context while cutting vision calls from ~80 to ~12.

WHICH TRANSITIONS GET VISION:
1. ALWAYS: First transition (need to understand starting application)
2. ALWAYS: Last transition (need to understand final state)
3. Page transitions (>50% pixel change — completely new screen)
4. Operations detected by OCR keywords (VLOOKUP, FILTER, etc.)
5. Low OCR change ratio (screen changed visually but text didn't — need eyes)
6. Modal/popup appearances

WHICH TRANSITIONS SKIP VISION:
1. High OCR change ratio (text clearly explains what happened)
2. Minor changes (small pixel diff, some text changed)
3. Form filling (new text appeared in specific fields — OCR captures it)
4. Scrolling (same content, shifted position)
"""

import re
from typing import List, Dict, Optional

from pdd_no_audio.clients.vision_llm import vision_client
from pdd_no_audio.utils import clean_vision_response, build_operation_context
from pdd_no_audio.config import llm_params


VISION_SYSTEM_PROMPT = """You are a senior business analyst analyzing screenshots from a business process demonstration.

RULES:
1. Describe EXACTLY what you see — every detail matters.
2. Identify the APPLICATION (Excel, SAP, browser, etc.).
3. For spreadsheets: identify VLOOKUP, FILTER, SORT, PIVOT, formulas, column names, cell ranges.
4. For web apps: identify page names, buttons, fields, menu paths.
5. Note dialogs, popups, ribbon tabs, toolbar selections, status messages.
6. Be SPECIFIC: mention actual text labels, column headers, button names.
7. Do NOT mention screenshots or recordings."""


def describe_transition(
    frame_before_path: str,
    frame_after_path: str,
    ocr_diff_summary: str = "",
    operation_context: str = "",
    call_index: int = 0
) -> Dict[str, str]:
    """
    SINGLE vision call per transition.
    Describes both the screen state AND the action performed.
    """
    prompt = f"""Analyze these two consecutive screenshots. The user performed an action between them.

PART 1 — AFTER SCREEN: Describe the second screenshot in detail:
- What application is shown?
- What page/tab/sheet is active?
- What data, columns, fields, buttons are visible?
- Any formula bar content, dialogs, menus, ribbon selections?

PART 2 — ACTION PERFORMED: What EXACT action did the user perform?
- For Excel: Was it VLOOKUP, FILTER, SORT, DUPLICATE REMOVAL, a formula, copy/paste?
  Which columns, cells, or ranges were involved?
- For web apps: What button was clicked? What field was filled? What was selected?
- What menu item or toolbar option was used?

Format your response EXACTLY as:
SCREEN: [detailed description of what is shown on the second screenshot]
ACTION: [one detailed sentence starting with "The user..."]
"""

    if ocr_diff_summary:
        prompt += f"\nText changes detected between screens:\n{ocr_diff_summary}"
    if operation_context:
        prompt += f"\n{operation_context}"

    response = vision_client.generate(
        prompt=prompt,
        image_paths=[frame_before_path, frame_after_path],
        system_prompt=VISION_SYSTEM_PROMPT,
        call_name=f"Transition_{call_index}"
    )

    result = {"screen_description": "", "action_description": ""}

    if response:
        cleaned = clean_vision_response(response)

        screen_match = re.search(
            r'SCREEN:\s*(.*?)(?=ACTION:|$)', cleaned, re.DOTALL | re.IGNORECASE
        )
        action_match = re.search(
            r'ACTION:\s*(.*?)$', cleaned, re.DOTALL | re.IGNORECASE
        )

        if screen_match:
            result["screen_description"] = screen_match.group(1).strip()
        if action_match:
            result["action_description"] = action_match.group(1).strip()

        # Fallback if parsing failed
        if not result["screen_description"] and not result["action_description"]:
            result["screen_description"] = cleaned
            result["action_description"] = cleaned

    return result


def analyze_transitions_smart(
    key_frames: List[Dict],
    ocr_diffs: List[Dict] = None,
    detected_operations: List[List[Dict]] = None,
    change_data: List[Dict] = None
) -> List[Dict]:
    """
    Smart transition analysis.

    ALL frames are kept for context and screenshots.
    Vision model is called ONLY for transitions that need visual understanding.
    OCR-only transitions still get full step synthesis from text LLM.

    The text LLM (qwen2.5:14b) receives:
    - For vision transitions: vision description + OCR text + pixel diff
    - For OCR-only transitions: OCR text before/after + pixel diff + detected operations

    Both produce equally detailed PDD steps because the text LLM is doing
    the heavy lifting of writing — vision just provides richer input signal.
    """
    if len(key_frames) < 2:
        return []

    total_pairs = len(key_frames) - 1
    max_vision = llm_params.max_vision_calls
    min_vision = min(llm_params.min_vision_calls, total_pairs)
    ocr_threshold = llm_params.ocr_sufficient_threshold

    # ── Score each transition for vision priority ──
    scored = []
    for i in range(total_pairs):
        score = 0.0
        reasons = []

        # RULE 1: First and last always get vision
        if i == 0:
            score += 100.0
            reasons.append("first_transition")
        if i == total_pairs - 1:
            score += 90.0
            reasons.append("last_transition")

        # RULE 2: Page transitions need vision (new application screen)
        if change_data and i < len(change_data):
            magnitude = change_data[i].get("pixel_change_magnitude", 0)
            change_type = change_data[i].get("change_type", "")

            if change_type == "page_transition":
                score += 60.0
                reasons.append("page_transition")
            elif change_type == "modal_popup":
                score += 50.0
                reasons.append("modal_popup")
            elif magnitude > 0.3:
                score += 40.0
                reasons.append(f"large_change_{magnitude:.2f}")

        # RULE 3: Operations detected → vision identifies specifics
        if detected_operations and i < len(detected_operations):
            ops = detected_operations[i]
            if ops:
                # Excel operations especially benefit from vision
                excel_ops = [op for op in ops if op["category"] == "Excel"]
                if excel_ops:
                    score += 45.0
                    reasons.append(f"excel_ops:{','.join(op['operation'] for op in excel_ops)}")
                else:
                    score += 20.0
                    reasons.append("operations_detected")

        # RULE 4: Low OCR change but significant pixel change → need vision
        if ocr_diffs and i < len(ocr_diffs):
            ocr_ratio = ocr_diffs[i].get("change_ratio", 0)
            pixel_mag = change_data[i].get("pixel_change_magnitude", 0) if change_data and i < len(change_data) else 0

            if ocr_ratio < 0.1 and pixel_mag > 0.05:
                # Screen changed visually but text didn't — vision needed
                score += 35.0
                reasons.append("visual_change_no_text")
            elif ocr_ratio > ocr_threshold:
                # Lots of text changed — OCR is sufficient
                score -= 15.0
                reasons.append("ocr_sufficient")

        # RULE 5: Every Nth transition gets vision for continuity
        if i > 0 and i < total_pairs - 1 and i % 5 == 0:
            score += 10.0
            reasons.append("periodic_check")

        scored.append((i, score, reasons))

    # ── Select transitions for vision ──
    scored.sort(key=lambda x: x[1], reverse=True)

    vision_indices = set()
    for idx, score, reasons in scored:
        if len(vision_indices) >= max_vision:
            break
        if score > 0 or len(vision_indices) < min_vision:
            vision_indices.add(idx)

    # Log selection
    print(
        f"    [Vision] Smart selection: {len(vision_indices)} vision calls "
        f"for {total_pairs} transitions"
    )
    for idx, score, reasons in scored[:len(vision_indices)]:
        if idx in vision_indices:
            print(
                f"      Transition {idx+1}: score={score:.0f} "
                f"[{', '.join(reasons)}]"
            )
    skipped = total_pairs - len(vision_indices)
    if skipped > 0:
        print(f"      ... {skipped} transitions using OCR-only")

    # ── Process all transitions ──
    transitions = []
    vision_count = 0

    for i in range(total_pairs):
        before = key_frames[i]
        after = key_frames[i + 1]

        # Build OCR context (available for ALL transitions)
        ocr_summary = ""
        ocr_before_text = before.get("ocr_text", "")
        ocr_after_text = after.get("ocr_text", "")

        if ocr_diffs and i < len(ocr_diffs):
            diff = ocr_diffs[i]
            added = diff.get("added_words", [])[:15]
            removed = diff.get("removed_words", [])[:15]
            if added:
                ocr_summary += f"New text appeared: {', '.join(added)}\n"
            if removed:
                ocr_summary += f"Text disappeared: {', '.join(removed)}\n"

        op_context = ""
        if detected_operations and i < len(detected_operations):
            op_context = build_operation_context(detected_operations[i])

        if i in vision_indices:
            # ── VISION PATH ──
            vision_result = describe_transition(
                before["path"], after["path"],
                ocr_diff_summary=ocr_summary,
                operation_context=op_context,
                call_index=vision_count
            )
            vision_count += 1

            after["vision_description"] = vision_result["screen_description"]

            transitions.append({
                "frame_before": before,
                "frame_after": after,
                "change_description": vision_result["action_description"],
                "timestamp_before": before.get("timestamp", 0),
                "timestamp_after": after.get("timestamp", 0),
                "order": i,
                "used_vision": True,
                "ocr_context": ocr_summary,
                "operation_context": op_context
            })

            if vision_count % 3 == 0:
                print(
                    f"    [Vision] {vision_count}/{len(vision_indices)} "
                    f"vision calls complete"
                )

        else:
            # ── OCR-ONLY PATH ──
            # Build rich description from OCR for the text LLM to work with
            ocr_desc = _build_rich_ocr_description(
                ocr_before_text, ocr_after_text,
                ocr_diffs[i] if ocr_diffs and i < len(ocr_diffs) else {},
                change_data[i] if change_data and i < len(change_data) else {},
                detected_operations[i] if detected_operations and i < len(detected_operations) else []
            )

            after["vision_description"] = ocr_after_text

            transitions.append({
                "frame_before": before,
                "frame_after": after,
                "change_description": ocr_desc,
                "timestamp_before": before.get("timestamp", 0),
                "timestamp_after": after.get("timestamp", 0),
                "order": i,
                "used_vision": False,
                "ocr_context": ocr_summary,
                "operation_context": op_context
            })

    print(
        f"    [Vision] Complete: {vision_count} vision, "
        f"{total_pairs - vision_count} OCR-only"
    )
    return transitions


def _build_rich_ocr_description(
    ocr_before: str,
    ocr_after: str,
    ocr_diff: Dict,
    change_info: Dict,
    operations: List[Dict]
) -> str:
    """
    Build a detailed change description from OCR data alone.
    This gives the text LLM enough context to write a good PDD step
    even without vision model input.
    """
    parts = []

    # Change type
    change_type = change_info.get("change_type", "")
    if change_type == "page_transition":
        parts.append("The user navigated to a different page or screen.")
    elif change_type == "modal_popup":
        parts.append("A dialog or popup appeared on screen.")
    elif change_type == "form_input":
        parts.append("The user entered data or interacted with a form element.")
    elif change_type == "menu_interaction":
        parts.append("The user interacted with a menu or navigation element.")

    # Text changes
    added = ocr_diff.get("added_words", [])
    removed = ocr_diff.get("removed_words", [])

    if added:
        # Look for meaningful patterns in added text
        added_str = ', '.join(added[:15])
        parts.append(f"New text visible on screen: {added_str}")

    if removed:
        removed_str = ', '.join(removed[:10])
        parts.append(f"Previous text no longer visible: {removed_str}")

    # Operations
    if operations:
        op_names = [op["display_name"] for op in operations]
        parts.append(f"Detected operations: {', '.join(op_names)}")

    # Pixel change magnitude
    magnitude = change_info.get("pixel_change_magnitude", 0)
    if magnitude > 0.5:
        parts.append("Major visual change — likely a page or application switch.")
    elif magnitude > 0.2:
        parts.append("Significant screen change — new section or large update.")
    elif magnitude > 0.05:
        parts.append("Moderate screen change — specific area updated.")

    # Key text from after screen (gives context about current state)
    if ocr_after and len(ocr_after) > 20:
        # Extract what looks like headings or labels
        after_preview = ocr_after[:300]
        parts.append(f"Current screen text includes: {after_preview}")

    if parts:
        return "The user performed an action. " + " ".join(parts)

    return "The user performed an action on the screen."


def identify_application(frame_path: str) -> str:
    """Identify the application shown in a frame."""
    prompt = """What application, website, or software tool is shown?
Reply with ONLY the application name. Nothing else.
Examples: "Microsoft Excel", "Google Chrome - Salesforce", "SAP GUI"."""

    response = vision_client.generate(
        prompt=prompt,
        image_paths=[frame_path],
        call_name="IdentifyApp"
    )

    if response:
        name = response.strip().strip('"\'').split('\n')[0].strip()
        if len(name) > 2 and len(name) < 50:
            return name
    return ""


# Legacy compatibility — kept so sop_agent.py imports still work
def describe_frame_batch(key_frames: List[Dict]) -> List[Dict]:
    """No-op — frame descriptions now happen inside analyze_transitions_smart."""
    return key_frames


def describe_changes_batch(
    key_frames: List[Dict],
    ocr_diffs: List[Dict] = None,
    detected_operations: List[List[Dict]] = None
) -> List[Dict]:
    """Legacy wrapper — calls analyze_transitions_smart."""
    return analyze_transitions_smart(key_frames, ocr_diffs, detected_operations)