# pdd_no_audio/llm_tasks/step_synthesizer.py

"""
PDD step synthesis using text LLM (qwen2.5:14b).
Converts vision descriptions and OCR data into detailed PDD process steps.
Enhanced to identify and describe specific operations (VLOOKUP, FILTER, etc.).
"""

import re
import time
from typing import List, Dict

from pdd_no_audio.clients.text_llm import text_client
from pdd_no_audio.llm_tasks.system_prompts import PDD_SYSTEM_PROMPT
from pdd_no_audio.utils import (
    timed, parse_numbered_steps, deduplicate_steps,
    safe_sample, detect_operations, build_operation_context
)


def synthesize_single_step(
    transition: Dict,
    change_info: Dict = None,
    step_index: int = 0
) -> str:
    """
    Synthesize a single PDD step from a frame transition.
    Produces detailed, automation-ready descriptions.
    """
    before_desc = transition.get("frame_before", {}).get("vision_description", "")
    after_desc = transition.get("frame_after", {}).get("vision_description", "")
    change_desc = transition.get("change_description", "")

    # Detect operations from all available text
    before_ocr = transition.get("frame_before", {}).get("ocr_text", "")
    after_ocr = transition.get("frame_after", {}).get("ocr_text", "")
    operations = detect_operations(
        f"{before_ocr} {after_ocr}",
        f"{before_desc} {after_desc}",
        change_desc
    )

    change_context = ""
    if change_info:
        change_type = change_info.get("change_type", "")
        text_diff = change_info.get("text_diff", {})
        added = text_diff.get("added_words", [])[:10]
        removed = text_diff.get("removed_words", [])[:10]

        if change_type:
            change_context += f"Change type: {change_type}\n"
        if added:
            change_context += f"New text on screen: {', '.join(added)}\n"
        if removed:
            change_context += f"Text no longer visible: {', '.join(removed)}\n"

    op_context = build_operation_context(operations)

    prompt = f"""Write a DETAILED PDD process step describing EXACTLY what action was performed.

BEFORE screen state:
{safe_sample(before_desc, 600)}

AFTER screen state:
{safe_sample(after_desc, 600)}

Action observed:
{safe_sample(change_desc, 400)}

{change_context}
{op_context}

INSTRUCTIONS:
- Write 2-4 sentences describing the EXACT operation performed.
- If this is an Excel operation (VLOOKUP, FILTER, SORT, DUPLICATE REMOVAL, etc.):
  Describe the function, the columns involved, the criteria used, and the purpose.
  Example: "The system applies a VLOOKUP formula in Column D, using the Employee ID from Column A as the lookup value against the reference table in Sheet2 (range A:C), to retrieve the corresponding department name. This ensures each record is enriched with department classification for downstream processing."
- If this is a web/application operation:
  Describe the exact navigation path, fields interacted with, and values entered.
  Example: "The system navigates to the 'Reports' tab in the main navigation menu and selects 'Monthly Summary' from the dropdown. The date range filter is configured from 01/01/2024 to 31/01/2024, and the 'Generate' button is clicked to produce the report."
- Write in third person: "The system...", "The automation..."
- Use ONLY names and labels from the descriptions above.

DETAILED STEP:"""

    response = text_client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        call_name=f"SynthStep_{step_index}"
    )

    if response:
        step = response.strip()
        step = re.sub(r'^(Step\s*\d+[:.]\s*)', '', step, flags=re.IGNORECASE)
        step = re.sub(r'^(Detailed Step[:.]\s*)', '', step, flags=re.IGNORECASE)
        step = step.strip('"').strip()
        if len(step) > 20:
            return step

    if change_desc:
        return change_desc
    return "The system proceeds to the next operation in the process sequence."


def synthesize_pdd_steps(
    transitions: List[Dict],
    change_data: List[Dict] = None
) -> List[Dict]:
    """
    Synthesize all PDD steps from transitions.
    Produces detailed, operation-aware descriptions.
    """
    if not transitions:
        return []

    start = time.time()
    total = len(transitions)
    print(f"    [Synthesize] Generating {total} detailed PDD steps...")

    steps = []
    for i, transition in enumerate(transitions):
        change_info = change_data[i] if change_data and i < len(change_data) else None

        step_text = synthesize_single_step(
            transition, change_info, step_index=i
        )

        # Detect operations for metadata
        before_ocr = transition.get("frame_before", {}).get("ocr_text", "")
        after_ocr = transition.get("frame_after", {}).get("ocr_text", "")
        before_desc = transition.get("frame_before", {}).get("vision_description", "")
        after_desc = transition.get("frame_after", {}).get("vision_description", "")
        change_desc = transition.get("change_description", "")

        operations = detect_operations(
            f"{before_ocr} {after_ocr}",
            f"{before_desc} {after_desc}",
            change_desc
        )

        steps.append({
            "number": i + 1,
            "description": step_text,
            "frame_before_path": transition.get("frame_before", {}).get("path", ""),
            "frame_after_path": transition.get("frame_after", {}).get("path", ""),
            "timestamp": transition.get("timestamp_after", 0),
            "change_type": change_info.get("change_type", "") if change_info else "",
            "change_region": change_info.get("primary_region") if change_info else None,
            "operations_detected": operations
        })

        if (i + 1) % 5 == 0 or (i + 1) == total:
            op_names = [op["display_name"] for op in operations]
            op_str = f" | Ops: {', '.join(op_names)}" if op_names else ""
            print(f"    [Synthesize] {i + 1}/{total} steps done{op_str}")

    unique_steps = _deduplicate_pdd_steps(steps)

    timed(f"Synthesis ({len(unique_steps)} steps)", start)
    return unique_steps


def _deduplicate_pdd_steps(steps: List[Dict]) -> List[Dict]:
    """Remove near-duplicate consecutive steps."""
    if len(steps) <= 1:
        return steps

    unique = [steps[0]]
    for step in steps[1:]:
        prev_key = re.sub(r'[^a-z]', '', unique[-1]["description"].lower())[:50]
        curr_key = re.sub(r'[^a-z]', '', step["description"].lower())[:50]
        if prev_key != curr_key:
            unique.append(step)
        else:
            print(f"    [Synthesize] Removed duplicate: '{step['description'][:50]}...'")

    for i, step in enumerate(unique):
        step["number"] = i + 1

    return unique