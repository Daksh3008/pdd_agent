# pdd_no_audio/llm_tasks/step_synthesizer.py

"""
PDD step synthesis using text LLM (qwen2.5:14b).
Converts vision descriptions and OCR data into detailed PDD process steps.
Enhanced: auth-aware step generation + parallel synthesis for speed.
FIXED: Anti-echo sanitization to prevent prompt leakage into steps.
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
    safe_sample, detect_operations_delta, build_operation_context,
    detect_auth_screen, get_auth_step_description
)


def _create_worker_client() -> TextLLMClient:
    """Create a fresh TextLLMClient for a worker thread."""
    client = TextLLMClient()
    return client


def _sanitize_step_response(text: str) -> str:
    """
    Remove prompt echoes, instruction blocks, and model refusals from step text.
    This is critical to prevent prompt leakage into the final document.
    """
    if not text:
        return ""
    
    # Patterns that indicate prompt echoes or instructions
    instruction_patterns = [
        # Headers and section markers
        r'INSTRUCTIONS?:.*?(?=The system|The user|The automation|$)',
        r'BEFORE\s+screen\s+state:.*?(?=AFTER|The system|$)',
        r'AFTER\s+screen\s+state:.*?(?=Action|The system|$)',
        r'Action\s+observed:.*?(?=The system|$)',
        r'DETAILED\s+STEP:?\s*',
        r'STEP\s+DESCRIPTION:?\s*',
        r'OUTPUT:?\s*',
        
        # Context blocks from prompt
        r'Change\s+type:.*?(?=The system|$)',
        r'Detected\s+operations.*?(?=The system|$)',
        r'New\s+text\s+on\s+screen:.*?(?=The system|$)',
        r'Text\s+no\s+longer\s+visible:.*?(?=The system|$)',
        r'Operations:\s*[-•].*?(?=The system|$)',
        
        # Instruction fragments
        r'Write\s+2-4\s+sentences.*',
        r'Write\s+in\s+third\s+person.*',
        r'Use\s+ONLY\s+names.*',
        r'If\s+this\s+is\s+an?\s+Excel\s+operation.*?(?=The system|$)',
        r'If\s+this\s+is\s+a\s+web.*?(?=The system|$)',
        r'If\s+this\s+is\s+a\s+LOGIN.*?(?=The system|$)',
        r'Describe\s+the\s+(?:function|exact|operation).*?(?=The system|$)',
        r'Which\s+columns.*?(?=The system|$)',
        r'What\s+menu\s+item.*?(?=The system|$)',
        r'Provide\s+the\s+exact.*?(?=The system|$)',
        
        # Model asking questions or refusing
        r'[Pp]lease\s+provide.*',
        r'I\s+need\s+more\s+information.*',
        r'[Cc]ould\s+you\s+(?:please\s+)?provide.*',
        r'I\s+cannot\s+determine.*',
        r'[Bb]ased\s+on\s+the\s+provided\s+information.*',
        r'I\s+would\s+need.*',
        r'[Tt]o\s+accurately.*',
        r'[Ww]ithout\s+more\s+context.*',
        r'[Uu]nfortunately.*',
        r'I\'m\s+(?:sorry|unable).*',
        
        # Bullet points that look like instructions
        r'^[\s]*[•\-\*]\s*(?:For|If|When|The|Write|Use|Describe).*$',
        
        # Common response prefixes to remove
        r'^(?:Sure|Certainly|Of course|Here\'s|Here is)[,!.]?\s*',
        r'^(?:Based on|Looking at|From|According to)\s+(?:the|this).*?(?=The system|The user|$)',
    ]
    
    cleaned = text
    for pattern in instruction_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
    
    # Remove multiple whitespace and newlines
    cleaned = re.sub(r'\n\s*\n+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # Remove step number prefixes (we add our own later)
    cleaned = re.sub(r'^(?:Step\s*\d+[:.]\s*)+', '', cleaned, flags=re.IGNORECASE)
    
    # Remove quotes around the entire text
    cleaned = cleaned.strip('"\'')
    
    # Final validation
    if len(cleaned) < 15:
        return ""
    
    # Check if it still looks like instructions
    lower = cleaned.lower()
    bad_starts = [
        'instructions', 'write ', 'describe ', 'if this is', 'for excel',
        'please ', 'i need', 'could you', 'would you', 'note:', 'important:',
        'rules:', 'format:'
    ]
    if any(lower.startswith(bad) for bad in bad_starts):
        return ""
    
    return cleaned


def _validate_step_quality(step_text: str) -> bool:
    """Check if a step description is valid and usable."""
    if not step_text:
        return False
    
    # Too short
    if len(step_text) < 20:
        return False
    
    # Contains obvious instruction fragments
    instruction_indicators = [
        'write 2-4', 'write in third person', 'use only names',
        'describe the function', 'if this is a', 'which columns',
        'please provide', 'i need more', 'could you provide',
        'based on the provided', 'instructions:', 'rules:'
    ]
    lower = step_text.lower()
    if any(ind in lower for ind in instruction_indicators):
        return False
    
    # Should contain some action verb
    action_verbs = [
        'the system', 'the user', 'the automation', 'the process',
        'navigates', 'clicks', 'selects', 'enters', 'opens', 'closes',
        'performs', 'executes', 'applies', 'submits', 'saves', 'loads',
        'displays', 'shows', 'validates', 'verifies', 'processes'
    ]
    if not any(verb in lower for verb in action_verbs):
        return False
    
    return True


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
    FIXED: Sanitizes output to prevent prompt leakage.
    """
    if client is None:
        from pdd_no_audio.clients.text_llm import text_client
        client = text_client

    before_desc = transition.get("frame_before", {}).get("vision_description", "")
    after_desc = transition.get("frame_after", {}).get("vision_description", "")
    change_desc = transition.get("change_description", "")

    # Detect operations using delta-based detection
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

    # For high-confidence auth screens, use template + minimal enrichment
    if auth_info.get("is_auth") and auth_info.get("confidence", 0) >= 0.5:
        auth_template = get_auth_step_description(
            auth_info["auth_type"],
            auth_info.get("indicators", []),
            app_name
        )
        
        # Try to enrich with specific details from vision
        if change_desc and len(change_desc) > 30:
            # Extract any specific button names or field labels
            specific_details = _extract_specific_ui_elements(change_desc)
            if specific_details:
                auth_template = auth_template.replace(
                    "the login page",
                    f"the login page ({specific_details})"
                )
        
        return auth_template

    # Standard step synthesis with strict output format
    prompt = f"""Write a single process step for a PDD document.

CONTEXT:
Screen before: {safe_sample(before_desc, 300)}
Screen after: {safe_sample(after_desc, 300)}
Action: {safe_sample(change_desc, 200)}

OUTPUT FORMAT:
Write exactly 2-3 sentences in third person describing what the automation does.
Start with "The system..." or "The automation..."
Be specific about application elements, field names, and button labels mentioned above.

STEP:"""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        call_name=f"SynthStep_{step_index}",
        temperature=0.3  # Lower temperature for more consistent output
    )

    if response:
        # Apply strict sanitization
        step = _sanitize_step_response(response)
        
        # Validate quality
        if _validate_step_quality(step):
            return step
        
        # If sanitized result is bad, try to use change_desc directly
        print(f"    [Synthesize] Step {step_index}: LLM output failed validation, using fallback")

    # Fallback chain
    if change_desc and len(change_desc) > 20:
        # Try to use the vision change description
        sanitized_change = _sanitize_step_response(change_desc)
        if sanitized_change and len(sanitized_change) > 20:
            # Ensure it starts with proper subject
            if not sanitized_change.lower().startswith(('the ', 'a ', 'an ')):
                sanitized_change = f"The system {sanitized_change[0].lower()}{sanitized_change[1:]}"
            return sanitized_change
    
    # Check for operations to generate a description
    if operations:
        op_names = [op["display_name"] for op in operations[:2]]
        return f"The system performs the following operation: {', '.join(op_names)}."
    
    # Ultimate fallback
    return "The system proceeds to the next step in the process sequence."


def _extract_specific_ui_elements(text: str) -> str:
    """Extract specific UI element names from text for enrichment."""
    if not text:
        return ""
    
    # Look for quoted text or specific patterns
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return ', '.join(quoted[:3])
    
    # Look for button/field patterns
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


def _synth_worker(args: Tuple) -> Tuple[int, str, List[Dict]]:
    """Worker function for parallel step synthesis."""
    i, transition, change_info, app_name = args
    client = _create_worker_client()
    step_text = synthesize_single_step(
        transition, change_info, step_index=i,
        app_name=app_name, client=client
    )

    # Get delta-based operations
    before_ocr = transition.get("frame_before", {}).get("ocr_text", "")
    after_ocr = transition.get("frame_after", {}).get("ocr_text", "")
    change_desc = transition.get("change_description", "")

    operations = detect_operations_delta(before_ocr, after_ocr, change_desc)
    
    return (i, step_text, operations)


def synthesize_pdd_steps(
    transitions: List[Dict],
    change_data: List[Dict] = None,
    app_name: str = ""
) -> List[Dict]:
    """
    Synthesize all PDD steps from transitions.
    Uses parallel workers for speed.
    Includes semantic deduplication to remove repeated steps.
    """
    if not transitions:
        return []

    start = time.time()
    total = len(transitions)
    workers = min(llm_params.text_llm_workers, total)
    print(f"    [Synthesize] Generating {total} PDD steps ({workers} parallel workers)...")

    # Filter out minor transitions before synthesis
    filtered_transitions = []
    filtered_change_data = []
    
    for i, transition in enumerate(transitions):
        change_info = change_data[i] if change_data and i < len(change_data) else {}
        
        # Skip truly minor changes (unless it's auth-related)
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
    
    # Build work items
    work_items = []
    for i, transition in enumerate(filtered_transitions):
        change_info = filtered_change_data[i] if i < len(filtered_change_data) else None
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
                if done_count % 5 == 0 or done_count == len(work_items):
                    print(f"    [Synthesize] {done_count}/{len(work_items)} steps done")
            except Exception as e:
                print(f"    [Synthesize] Error on step {idx}: {e}")
                results[idx] = ("The system proceeds to the next operation.", [])

    # Assemble in order
    steps = []
    for i in range(len(filtered_transitions)):
        step_text, operations = results.get(i, ("The system proceeds.", []))
        transition = filtered_transitions[i]
        change_info = filtered_change_data[i] if i < len(filtered_change_data) else {}

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

    # Semantic deduplication
    unique_steps = _deduplicate_pdd_steps(steps)

    timed(f"Synthesis ({len(unique_steps)} steps)", start)
    return unique_steps


def _simple_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple word-based similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    
    # Normalize and tokenize
    words1 = set(re.findall(r'[a-z]+', text1.lower()))
    words2 = set(re.findall(r'[a-z]+', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def _deduplicate_pdd_steps(steps: List[Dict]) -> List[Dict]:
    """
    Remove near-duplicate consecutive steps using semantic similarity.
    Also removes generic fallback steps if surrounded by meaningful content.
    """
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
        
        # Check if current step is too generic
        is_generic = any(phrase in curr_desc.lower() for phrase in generic_phrases)
        
        # Calculate similarity
        similarity = _simple_text_similarity(prev_desc, curr_desc)
        
        # Skip if:
        # 1. Very similar (>85%) to previous step
        # 2. Generic step when previous was meaningful
        if similarity > 0.85:
            print(f"    [Synthesize] Merged similar step {step['number']}: {similarity:.0%} similar")
            continue
        
        if is_generic and len(prev_desc) > 50:
            print(f"    [Synthesize] Skipped generic step {step['number']}")
            continue
        
        # Check for same operation type repetition
        ops_prev = set(o["operation"] for o in unique[-1].get("operations_detected", []))
        ops_curr = set(o["operation"] for o in step.get("operations_detected", []))
        
        if ops_prev and ops_prev == ops_curr and similarity > 0.6:
            print(f"    [Synthesize] Merged repeated operation step {step['number']}")
            continue
        
        unique.append(step)

    # Renumber
    for i, step in enumerate(unique):
        step["number"] = i + 1

    if len(unique) < len(steps):
        print(f"    [Synthesize] Deduplicated: {len(steps)} → {len(unique)} steps")

    return unique