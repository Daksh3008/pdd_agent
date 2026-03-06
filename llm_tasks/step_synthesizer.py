# llm_tasks/step_synthesizer.py

"""
PDD step synthesis from vision transitions.
Converts vision descriptions and OCR data into detailed PDD process steps.
Used by the video pipeline.

OPTIMIZED: Batches multiple transitions into single LLM calls
to reduce API usage from N calls to N/batch_size calls.
"""

import re
import time
import concurrent.futures
from typing import List, Dict, Tuple

from core.gemini_client import gemini_client
from core.config import config
from core.utils import (
    timed, safe_sample, detect_operations_delta,
    detect_auth_screen, get_auth_step_description,
    redact_pii_text, enforce_tone
)
from llm_tasks.system_prompts import PDD_SYSTEM_PROMPT


# ============================================================
# Constants
# ============================================================

BATCH_SIZE = 8  # Transitions per LLM call


def _sanitize_step_response(text: str) -> str:
    """Remove prompt echoes and instructions from step text."""
    if not text:
        return ""

    patterns = [
        r'INSTRUCTIONS?:.*?(?=The system|The user|The automation|$)',
        r'BEFORE\s+screen\s+state:.*?(?=AFTER|The system|$)',
        r'AFTER\s+screen\s+state:.*?(?=ACTION|The system|$)',
        r'DETAILED\s+STEP:?\s*',
        r'STEP\s+DESCRIPTION:?\s*',
        r'OUTPUT:?\s*',
        r'Write\s+2-4\s+sentences.*',
        r'Write\s+in\s+third\s+person.*',
        r'[Pp]lease\s+provide.*',
        r'I\s+need\s+more\s+information.*',
        r'I\s+cannot\s+determine.*',
        r'^(?:Sure|Certainly|Of course)[,!.]?\s*',
        r'^(?:Based on|Looking at|From|According to)\s+(?:the|this).*?(?=The system|The user|$)',
    ]

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    cleaned = re.sub(r'\n\s*\n+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = re.sub(r'^(?:Step\s*\d+[:.]\s*)+', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip('"\'')

    # Strip markdown
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'__([^_]+)__', r'\1', cleaned)

    return cleaned if len(cleaned) >= 15 else ""


def _validate_step_quality(step_text: str) -> bool:
    """Check if a step description is valid and usable."""
    if not step_text or len(step_text) < 20:
        return False

    instruction_indicators = [
        'write 2-4', 'write in third person', 'use only names',
        'describe the function', 'if this is a', 'please provide',
        'i need more', 'instructions:', 'rules:'
    ]
    lower = step_text.lower()
    if any(ind in lower for ind in instruction_indicators):
        return False

    action_verbs = [
        'the system', 'the user', 'the automation', 'the process',
        'navigates', 'clicks', 'selects', 'enters', 'opens', 'closes',
        'performs', 'executes', 'applies', 'submits', 'saves', 'loads',
        'displays', 'shows', 'validates', 'verifies', 'processes'
    ]
    if not any(verb in lower for verb in action_verbs):
        return False

    return True


def _extract_specific_ui_elements(text: str) -> str:
    """Extract specific UI element names from text for enrichment."""
    if not text:
        return ""

    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return ', '.join(quoted[:3])

    patterns = [
        r'(?:button|btn)\s+(?:labeled\s+)?["\']?([A-Za-z\s]+)["\']?',
        r'(?:field|input)\s+(?:labeled\s+)?["\']?([A-Za-z\s]+)["\']?',
        r'(?:click(?:ed|s)?|select(?:ed|s)?)\s+(?:on\s+)?["\']?([A-Za-z\s]+)["\']?',
    ]

    elements = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        elements.extend(matches)

    if elements:
        return ', '.join(elements[:3])
    return ""


# ============================================================
# Batched Step Synthesis (replaces per-step LLM calls)
# ============================================================

def _build_batch_prompt(batch_transitions: List[Dict], app_name: str = "") -> str:
    """
    Build a single prompt that describes multiple transitions
    and asks for one step description per transition.
    """
    sections = []

    for i, item in enumerate(batch_transitions):
        transition = item["transition"]
        idx = item["index"]

        before_desc = transition.get("frame_before", {}).get("vision_description", "")
        after_desc = transition.get("frame_after", {}).get("vision_description", "")
        change_desc = transition.get("change_description", "")

        before_ocr = transition.get("frame_before", {}).get("ocr_text", "")
        after_ocr = transition.get("frame_after", {}).get("ocr_text", "")

        # Truncate to keep prompt manageable
        before_summary = safe_sample(before_desc or before_ocr, 200)
        after_summary = safe_sample(after_desc or after_ocr, 200)
        change_summary = safe_sample(change_desc, 150)

        sections.append(
            f"TRANSITION {idx}:\n"
            f"  Before: {before_summary}\n"
            f"  After: {after_summary}\n"
            f"  Action: {change_summary}"
        )

    transitions_text = "\n\n".join(sections)

    app_context = f'Application: "{app_name}"\n' if app_name else ""

    prompt = f"""Write PDD process step descriptions for the following screen transitions.

{app_context}
{transitions_text}

RULES:
- Write EXACTLY one step per TRANSITION, numbered to match.
- Each step is 2-3 sentences describing what the system/automation does.
- Start each with "The system..." or "The automation..."
- Be specific about UI elements, buttons, fields mentioned.
- Third person, present tense, active voice.
- NEVER include personal names, emails, or phone numbers.
- Do NOT use markdown formatting.

OUTPUT FORMAT (one per line, numbered to match transitions):
{chr(10).join(f'STEP {item["index"]}: [description]' for item in batch_transitions)}
"""

    return prompt


def _parse_batch_response(response: str, expected_indices: List[int]) -> Dict[int, str]:
    """Parse a batched response into individual step descriptions."""
    results = {}
    if not response:
        return results

    # Try pattern: STEP N: description
    for idx in expected_indices:
        pattern = rf'STEP\s*{idx}\s*:\s*(.*?)(?=STEP\s*\d+\s*:|$)'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            text = _sanitize_step_response(text)
            text = redact_pii_text(text)
            if _validate_step_quality(text):
                results[idx] = text

    # Fallback: try numbered lines
    if len(results) < len(expected_indices) // 2:
        lines = response.split('\n')
        line_idx = 0
        for idx in expected_indices:
            if idx in results:
                continue
            # Find next substantial line
            while line_idx < len(lines):
                line = lines[line_idx].strip()
                line_idx += 1
                cleaned = re.sub(r'^\d+[\.\):\s]+', '', line).strip()
                cleaned = re.sub(r'^STEP\s*\d+\s*:\s*', '', cleaned, flags=re.IGNORECASE).strip()
                cleaned = _sanitize_step_response(cleaned)
                cleaned = redact_pii_text(cleaned)
                if cleaned and len(cleaned) > 20:
                    results[idx] = cleaned
                    break

    return results


def synthesize_single_step_local(
    transition: Dict,
    change_info: Dict = None,
    step_index: int = 0,
    app_name: str = ""
) -> str:
    """
    Generate a step description WITHOUT an LLM call.
    Uses OCR data, change description, and auth detection.
    Used as fallback when LLM batch doesn't cover a step.
    """
    before_desc = transition.get("frame_before", {}).get("vision_description", "")
    after_desc = transition.get("frame_after", {}).get("vision_description", "")
    change_desc = transition.get("change_description", "")

    before_ocr = transition.get("frame_before", {}).get("ocr_text", "")
    after_ocr = transition.get("frame_after", {}).get("ocr_text", "")
    operations = detect_operations_delta(before_ocr, after_ocr, change_desc)

    # Check for auth screens
    auth_info = transition.get("auth_info", {})
    if not auth_info or not auth_info.get("is_auth"):
        auth_info = detect_auth_screen(
            f"{before_ocr} {after_ocr}",
            f"{before_desc} {after_desc} {change_desc}"
        )

    # Auth screen — use template
    if auth_info.get("is_auth") and auth_info.get("confidence", 0) >= 0.5:
        auth_template = get_auth_step_description(
            auth_info["auth_type"],
            auth_info.get("indicators", []),
            app_name
        )
        if change_desc and len(change_desc) > 30:
            specific = _extract_specific_ui_elements(change_desc)
            if specific:
                auth_template = auth_template.replace(
                    "the login page",
                    f"the login page ({specific})"
                )
        return auth_template

    # Use change description if available
    if change_desc and len(change_desc) > 20:
        sanitized = _sanitize_step_response(change_desc)
        if sanitized and len(sanitized) > 20:
            if not sanitized.lower().startswith(('the ', 'a ', 'an ')):
                sanitized = f"The system {sanitized[0].lower()}{sanitized[1:]}"
            return redact_pii_text(sanitized)

    # Use operations
    if operations:
        op_names = [op["display_name"] for op in operations[:2]]
        return f"The system performs the following operation: {', '.join(op_names)}."

    return "The system proceeds to the next step in the process sequence."


def _simple_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple word-based similarity between two texts."""
    if not text1 or not text2:
        return 0.0

    words1 = set(re.findall(r'[a-z]+', text1.lower()))
    words2 = set(re.findall(r'[a-z]+', text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def _deduplicate_pdd_steps(steps: List[Dict]) -> List[Dict]:
    """Remove near-duplicate consecutive steps using semantic similarity."""
    if len(steps) <= 1:
        return steps

    unique = [steps[0]]

    generic_phrases = [
        "the system proceeds",
        "the user performed an action",
        "screen state changed",
        "the system continues"
    ]

    for step in steps[1:]:
        prev_desc = unique[-1]["description"]
        curr_desc = step["description"]

        is_generic = any(phrase in curr_desc.lower() for phrase in generic_phrases)
        similarity = _simple_text_similarity(prev_desc, curr_desc)

        if similarity > 0.85:
            print(f"    [Synthesize] Merged similar step {step['number']}: {similarity:.0%} similar")
            continue

        if is_generic and len(prev_desc) > 50:
            print(f"    [Synthesize] Skipped generic step {step['number']}")
            continue

        ops_prev = set(o["operation"] for o in unique[-1].get("operations_detected", []))
        ops_curr = set(o["operation"] for o in step.get("operations_detected", []))

        if ops_prev and ops_prev == ops_curr and similarity > 0.6:
            print(f"    [Synthesize] Merged repeated operation step {step['number']}")
            continue

        unique.append(step)

    for i, step in enumerate(unique):
        step["number"] = i + 1

    if len(unique) < len(steps):
        print(f"    [Synthesize] Deduplicated: {len(steps)} -> {len(unique)} steps")

    return unique


# ============================================================
# Main Entry Point
# ============================================================

def synthesize_pdd_steps(
    transitions: List[Dict],
    change_data: List[Dict] = None,
    app_name: str = ""
) -> List[Dict]:
    """
    Synthesize all PDD steps from transitions.
    Uses BATCHED LLM calls to reduce API usage.
    N transitions -> ceil(N/BATCH_SIZE) LLM calls instead of N calls.
    """
    if not transitions:
        return []

    start = time.time()
    total = len(transitions)
    print(f"    [Synthesize] Processing {total} transitions (batch size: {BATCH_SIZE})...")

    # Pre-filter minor transitions
    filtered_transitions = []
    filtered_change_data = []

    for i, transition in enumerate(transitions):
        change_info = change_data[i] if change_data and i < len(change_data) else {}

        is_auth = transition.get("auth_info", {}).get("is_auth", False)
        change_type = change_info.get("change_type", "")
        magnitude = change_info.get("pixel_change_magnitude", 0)

        if change_type == "minor_change" and magnitude < 0.02 and not is_auth:
            print(f"    [Synthesize] Skipping minor transition {i+1}")
            continue

        filtered_transitions.append(transition)
        filtered_change_data.append(change_info)

    if len(filtered_transitions) < len(transitions):
        print(f"    [Synthesize] Filtered {len(transitions) - len(filtered_transitions)} minor transitions")

    total_filtered = len(filtered_transitions)

    # Separate auth steps (handled locally, no LLM needed) from LLM steps
    auth_results = {}    # index -> step_text
    llm_needed = []      # list of {"index": i, "transition": t, "change_info": c}

    for i, transition in enumerate(filtered_transitions):
        auth_info = transition.get("auth_info", {})
        before_ocr = transition.get("frame_before", {}).get("ocr_text", "")
        after_ocr = transition.get("frame_after", {}).get("ocr_text", "")

        if not auth_info.get("is_auth"):
            auth_info = detect_auth_screen(f"{before_ocr} {after_ocr}")

        if auth_info.get("is_auth") and auth_info.get("confidence", 0) >= 0.5:
            # Handle auth locally — no LLM call needed
            step_text = get_auth_step_description(
                auth_info["auth_type"],
                auth_info.get("indicators", []),
                app_name
            )
            auth_results[i] = step_text
        else:
            change_info = filtered_change_data[i] if i < len(filtered_change_data) else {}
            llm_needed.append({
                "index": i,
                "transition": transition,
                "change_info": change_info
            })

    if auth_results:
        print(f"    [Synthesize] {len(auth_results)} auth steps handled locally (no LLM)")

    # Batch LLM calls
    llm_results = {}  # index -> step_text

    if llm_needed:
        num_batches = (len(llm_needed) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"    [Synthesize] {len(llm_needed)} steps need LLM -> {num_batches} batch calls")

        for batch_num in range(num_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(llm_needed))
            batch_items = llm_needed[batch_start:batch_end]

            prompt = _build_batch_prompt(batch_items, app_name)
            expected_indices = [item["index"] for item in batch_items]

            response = gemini_client.generate(
                prompt=prompt,
                system_prompt=PDD_SYSTEM_PROMPT,
                call_name=f"SynthBatch_{batch_num+1}of{num_batches}",
                temperature=0.3,
                max_retries=3
            )

            parsed = _parse_batch_response(response, expected_indices)
            llm_results.update(parsed)

            # Report progress
            done_so_far = len(auth_results) + len(llm_results)
            print(f"    [Synthesize] Batch {batch_num+1}/{num_batches} done "
                  f"({len(parsed)}/{len(batch_items)} parsed, "
                  f"{done_so_far}/{total_filtered} total)")

    # Fill any gaps with local fallback (no LLM)
    fallback_count = 0
    for i in range(total_filtered):
        if i not in auth_results and i not in llm_results:
            transition = filtered_transitions[i]
            change_info = filtered_change_data[i] if i < len(filtered_change_data) else {}
            llm_results[i] = synthesize_single_step_local(
                transition, change_info, step_index=i, app_name=app_name
            )
            fallback_count += 1

    if fallback_count:
        print(f"    [Synthesize] {fallback_count} steps used local fallback")

    # Assemble steps in order
    steps = []
    for i in range(total_filtered):
        transition = filtered_transitions[i]
        change_info = filtered_change_data[i] if i < len(filtered_change_data) else {}

        step_text = auth_results.get(i) or llm_results.get(i, "The system proceeds to the next operation.")

        before_ocr = transition.get("frame_before", {}).get("ocr_text", "")
        after_ocr = transition.get("frame_after", {}).get("ocr_text", "")
        change_desc = transition.get("change_description", "")
        operations = detect_operations_delta(before_ocr, after_ocr, change_desc)

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

    # Deduplicate
    unique_steps = _deduplicate_pdd_steps(steps)

    api_calls = (len(llm_needed) + BATCH_SIZE - 1) // BATCH_SIZE if llm_needed else 0
    timed(f"Synthesis ({len(unique_steps)} steps, {api_calls} API calls)", start)
    return unique_steps