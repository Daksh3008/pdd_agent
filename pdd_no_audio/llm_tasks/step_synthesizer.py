# pdd_no_audio/llm_tasks/step_synthesizer.py

"""
PDD step synthesis using text LLM (qwen2.5:14b).
Converts vision descriptions and OCR data into detailed PDD process steps.
Enhanced: auth-aware step generation + parallel synthesis for speed.
"""

import re
import time
import concurrent.futures
from typing import List, Dict, Tuple

from pdd_no_audio.clients.text_llm import TextLLMClient
from pdd_no_audio.config import text_config, llm_params
from pdd_no_audio.llm_tasks.system_prompts import PDD_SYSTEM_PROMPT
from pdd_no_audio.utils import (
    timed, parse_numbered_steps, deduplicate_steps,
    safe_sample, detect_operations, build_operation_context,
    detect_auth_screen, get_auth_step_description
)


def _create_worker_client() -> TextLLMClient:
    """Create a fresh TextLLMClient for a worker thread."""
    client = TextLLMClient()
    # Workers don't track tokens individually — main tracker records via call_name
    return client


def synthesize_single_step(
    transition: Dict,
    change_info: Dict = None,
    step_index: int = 0,
    app_name: str = "",
    client: TextLLMClient = None
) -> str:
    """
    Synthesize a single PDD step from a frame transition.
    Auth-aware: generates detailed auth steps for login/logout screens.
    """
    if client is None:
        from pdd_no_audio.clients.text_llm import text_client
        client = text_client

    before_desc = transition.get("frame_before", {}).get("vision_description", "")
    after_desc = transition.get("frame_after", {}).get("vision_description", "")
    change_desc = transition.get("change_description", "")

    # Detect operations
    before_ocr = transition.get("frame_before", {}).get("ocr_text", "")
    after_ocr = transition.get("frame_after", {}).get("ocr_text", "")
    operations = detect_operations(
        f"{before_ocr} {after_ocr}",
        f"{before_desc} {after_desc}",
        change_desc
    )

    # Check for auth screens
    auth_info = transition.get("auth_info", {})
    if not auth_info or not auth_info.get("is_auth"):
        # Re-check from OCR/vision text
        auth_info = detect_auth_screen(
            f"{before_ocr} {after_ocr}",
            f"{before_desc} {after_desc} {change_desc}"
        )

    # For high-confidence auth screens, use template + LLM enrichment
    if auth_info.get("is_auth") and auth_info.get("confidence", 0) >= 0.5:
        auth_template = get_auth_step_description(
            auth_info["auth_type"],
            auth_info.get("indicators", []),
            app_name
        )

        # Enrich the template with actual screen details from vision/OCR
        enrichment_prompt = f"""Refine this authentication step description using the actual screen details observed.

Template description:
{auth_template}

Screen details observed:
Before: {safe_sample(before_desc, 300)}
After: {safe_sample(after_desc, 300)}
Action: {safe_sample(change_desc, 200)}
OCR text: {safe_sample(f'{before_ocr} {after_ocr}', 300)}

INSTRUCTIONS:
- Keep the professional PDD format from the template.
- Replace generic placeholders with actual application names, field labels, and button text observed.
- If specific field names or button labels are visible, use them exactly.
- Write in third person: "The system...", "The automation..."
- Keep it 2-4 sentences.

Refined step:"""

        response = client.generate(
            prompt=enrichment_prompt,
            system_prompt=PDD_SYSTEM_PROMPT,
            call_name=f"SynthStep_Auth_{step_index}"
        )

        if response:
            step = response.strip()
            step = re.sub(r'^(Step\s*\d+[:.]\s*)', '', step, flags=re.IGNORECASE)
            step = re.sub(r'^(Refined step[:.]\s*)', '', step, flags=re.IGNORECASE)
            step = step.strip('"').strip()
            if len(step) > 30:
                return step

        # Fallback to template
        return auth_template

    # Standard step synthesis (non-auth)
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
{safe_sample(before_desc, 400)}

AFTER screen state:
{safe_sample(after_desc, 400)}

Action observed:
{safe_sample(change_desc, 300)}

{change_context}
{op_context}

INSTRUCTIONS:
- Write 2-4 sentences describing the EXACT operation performed.
- If this is an Excel operation (VLOOKUP, FILTER, SORT, DUPLICATE REMOVAL, etc.):
  Describe the function, the columns involved, the criteria used, and the purpose.
- If this is a web/application operation:
  Describe the exact navigation path, fields interacted with, and values entered.
- If this is a LOGIN/LOGOUT/AUTHENTICATION action:
  Describe the authentication flow, fields used, and outcome.
- Write in third person: "The system...", "The automation..."
- Use ONLY names and labels from the descriptions above.

DETAILED STEP:"""

    response = client.generate(
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


def _synth_worker(args: Tuple) -> Tuple[int, str, List[Dict]]:
    """Worker function for parallel step synthesis."""
    i, transition, change_info, app_name = args
    client = _create_worker_client()
    step_text = synthesize_single_step(
        transition, change_info, step_index=i,
        app_name=app_name, client=client
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
    return (i, step_text, operations)


def synthesize_pdd_steps(
    transitions: List[Dict],
    change_data: List[Dict] = None,
    app_name: str = ""
) -> List[Dict]:
    """
    Synthesize all PDD steps from transitions.
    Uses parallel workers for speed (configurable via llm_params.text_llm_workers).
    Skips separate enhance_steps call — merging is done inline.
    """
    if not transitions:
        return []

    start = time.time()
    total = len(transitions)
    workers = min(llm_params.text_llm_workers, total)
    print(f"    [Synthesize] Generating {total} PDD steps ({workers} parallel workers)...")

    # Build work items
    work_items = []
    for i, transition in enumerate(transitions):
        change_info = change_data[i] if change_data and i < len(change_data) else None
        work_items.append((i, transition, change_info, app_name))

    # Parallel synthesis
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_synth_worker, item): item[0] for item in work_items}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                i, step_text, operations = future.result()
                results[i] = (step_text, operations)
                done_count += 1
                if done_count % 5 == 0 or done_count == total:
                    print(f"    [Synthesize] {done_count}/{total} steps done")
            except Exception as e:
                print(f"    [Synthesize] Error on step {idx}: {e}")
                results[idx] = ("The system proceeds to the next operation.", [])

    # Assemble in order
    steps = []
    for i in range(total):
        step_text, operations = results.get(i, ("The system proceeds.", []))
        transition = transitions[i]
        change_info = change_data[i] if change_data and i < len(change_data) else {}

        steps.append({
            "number": i + 1,
            "description": step_text,
            "frame_before_path": transition.get("frame_before", {}).get("path", ""),
            "frame_after_path": transition.get("frame_after", {}).get("path", ""),
            "timestamp": transition.get("timestamp_after", 0),
            "change_type": change_info.get("change_type", "") if change_info else "",
            "change_region": change_info.get("primary_region") if change_info else None,
            "operations_detected": operations,
            "auth_info": transition.get("auth_info", {})
        })

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