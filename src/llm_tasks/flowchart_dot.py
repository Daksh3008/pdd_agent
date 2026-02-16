# src/llm_tasks/flowchart_dot.py

"""
DOT flowchart code generation.
Two-phase approach:
1. Classify steps deterministically in code (PROCESS/DECISION/LOOP/END)
2. Send classified steps to LLM for DOT structure generation
3. Deterministic fallback if LLM fails

This ensures consistent flowcharts because the LLM receives
clean, classified input — not raw transcript.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

from llm_client import llm_client
from llm_tasks.utils import _timed, _filter_conversation_steps
from llm_tasks.requirements import get_interface_requirements
from llm_tasks.entity_extraction import extract_entities_and_project


# ============================================================
# STEP CLASSIFICATION — Deterministic, no LLM
# ============================================================

# Keywords that indicate a decision/conditional step
DECISION_KEYWORDS = [
    'if ', 'whether', 'check if', 'verify if', 'determine if',
    'validate if', 'confirm if', 'eligible', 'meets criteria',
    'qualifies', 'is valid', 'is active', 'is inactive',
    'has been', 'found', 'not found', 'exists', 'does not exist',
    'successful', 'failed', 'pass', 'fail', 'approved', 'rejected',
    'matches', 'does not match', 'compliant', 'non-compliant'
]

# Keywords that indicate iteration/looping
LOOP_KEYWORDS = [
    'each', 'every', 'all ', 'iterate', 'repeat', 'loop',
    'next record', 'next item', 'next entry', 'next user',
    'next account', 'next server', 'next group',
    'remaining', 'batch', 'one by one', 'for all',
    'processes each', 'validates each', 'checks each',
    'reviews each', 'handles each'
]

# Keywords that indicate end/reporting steps
END_PHASE_KEYWORDS = [
    'final report', 'final status', 'execution status',
    'completion', 'summary report', 'audit log',
    'logs out', 'closes', 'terminates', 'ends',
    'overall status', 'execution outcome'
]


def classify_steps(steps: List[str]) -> List[Dict]:
    """
    Classify each step as PROCESS, DECISION, LOOP, or END_PHASE.
    Pure code logic — no LLM involved.

    Returns list of dicts:
    {
        "index": 1,
        "text": "Original step text",
        "type": "PROCESS" | "DECISION" | "LOOP" | "END_PHASE",
        "short_label": "Shortened label for flowchart node"
    }
    """
    classified = []

    for i, step in enumerate(steps):
        step_lower = step.lower()
        step_type = "PROCESS"

        # Check decision
        if any(kw in step_lower for kw in DECISION_KEYWORDS):
            step_type = "DECISION"

        # Check loop (overrides decision if both match)
        elif any(kw in step_lower for kw in LOOP_KEYWORDS):
            step_type = "LOOP"

        # Check end phase
        elif any(kw in step_lower for kw in END_PHASE_KEYWORDS):
            step_type = "END_PHASE"

        # Generate short label for flowchart
        short_label = _shorten_label(step)

        classified.append({
            "index": i + 1,
            "text": step,
            "type": step_type,
            "short_label": short_label
        })

    # Log classification
    type_counts = {}
    for c in classified:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
    print(f"    [Classify] {len(classified)} steps: {type_counts}")

    return classified


def _shorten_label(text: str, max_words: int = 6) -> str:
    """Shorten step text to a flowchart-friendly label."""
    # Remove common prefixes
    text = re.sub(
        r'^(the system|the automation|the bot|the solution|it)\s+',
        '', text, flags=re.IGNORECASE
    ).strip()

    # Remove trailing periods
    text = text.rstrip('.')

    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    # Truncate to max words
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words]) + '...'

    return text


# ============================================================
# FORMAT CLASSIFIED STEPS FOR LLM
# ============================================================

def _format_steps_for_prompt(classified: List[Dict]) -> str:
    """
    Format classified steps into a clear prompt section.
    The LLM sees step type annotations to guide flowchart structure.
    """
    lines = []
    for c in classified:
        type_tag = c["type"]
        if type_tag == "DECISION":
            lines.append(
                f"{c['index']}. [DECISION] {c['text']}"
            )
        elif type_tag == "LOOP":
            lines.append(
                f"{c['index']}. [LOOP - repeats for multiple items] {c['text']}"
            )
        elif type_tag == "END_PHASE":
            lines.append(
                f"{c['index']}. [END PHASE] {c['text']}"
            )
        else:
            lines.append(
                f"{c['index']}. {c['text']}"
            )
    return "\n".join(lines)


# ============================================================
# LLM DOT GENERATION — from classified steps
# ============================================================

def generate_dot_and_apps(process_steps, transcript, entities=None):
    """Generate DOT flowchart code and interface requirements."""
    start = time.time()
    if entities is None:
        entities, _ = extract_entities_and_project(transcript)
    dot_code = generate_dot_from_classified_steps(process_steps)
    apps = get_interface_requirements(transcript, entities)
    _timed("DOT+Apps", start)
    return dot_code, apps


def generate_dot_from_classified_steps(steps: List[str]) -> str:
    """
    Main flowchart generation pipeline:
    1. Filter conversation steps
    2. Classify steps deterministically
    3. Send classified steps to LLM for DOT generation
    4. Fall back to deterministic DOT if LLM fails
    """
    if not steps:
        return _deterministic_dot([{
            "index": 1, "text": "Process",
            "type": "PROCESS", "short_label": "Process"
        }])

    # Step 1: Filter
    filtered = _filter_conversation_steps(steps)

    # Step 2: Classify
    classified = classify_steps(filtered[:15])

    if not classified:
        return _deterministic_dot([{
            "index": 1, "text": "Process",
            "type": "PROCESS", "short_label": "Process"
        }])

    # Step 3: LLM generation
    dot_code = _llm_generate_dot(classified)

    if dot_code:
        return dot_code

    # Step 4: Deterministic fallback
    print("    [DOT] LLM failed, using deterministic generation")
    return _deterministic_dot(classified)


def _llm_generate_dot(classified: List[Dict]) -> Optional[str]:
    """
    Send classified steps to LLM for DOT code generation.
    The LLM receives a CLEAN, STRUCTURED input — not raw transcript.
    """
    formatted_steps = _format_steps_for_prompt(classified)

    prompt = f"""You are a senior business analyst. You have received all the classified steps for an automation process. Generate a Graphviz DOT flowchart for this process.

The steps have been pre-classified:
- Regular steps → box nodes
- [DECISION] steps → diamond nodes with Yes/No branches
- [LOOP] steps → must have a loop-back arrow (a "More Items?" diamond that loops back)
- [END PHASE] steps → these come near the end before the End node

CLASSIFIED STEPS:
{formatted_steps}

FLOWCHART RULES:
1. Start with a Start node (oval, green) and end with an End node (oval, red)
2. Every node MUST have: label="Short Description" (max 5-6 words)
3. Regular steps: shape=box, fillcolor=lightblue
4. Decisions: shape=diamond, fillcolor=gold, with Yes/No edge labels
5. After a [LOOP] step, add a decision diamond "More Items?" that loops back to the loop step on "Yes" and continues forward on "No"
6. Edges between regular steps have NO labels
7. Every path must reach the End node
8. Use node IDs like Step1, Step2, Decision1, etc.

OUTPUT FORMAT — output ONLY the DOT code:
```dot
digraph ProcessFlow {{
    rankdir=TB;
    node [fontname="Arial", fontsize=10, style=filled];
    edge [fontname="Arial", fontsize=9];

    Start [label="Start", shape=oval, fillcolor=lightgreen];
    Step1 [label="...", shape=box, fillcolor=lightblue];
    Decision1 [label="...?", shape=diamond, fillcolor=gold];
    ...
    End [label="End", shape=oval, fillcolor=lightcoral];

    Start -> Step1;
    Step1 -> Step2;
    ...
}}
```"""

    response = llm_client.generate(
        prompt, temperature=0.2,
        call_name="DOTGeneration"
    )

    if not response:
        return None

    dot = _extract_dot(response)
    if not dot:
        return None

    # Validate
    has_labels = re.search(r'\w+\s*\[.*?label\s*=', dot, re.IGNORECASE)
    if not has_labels:
        print("    [DOT] LLM output has no node labels")
        return None

    # Ensure End node exists
    if not re.search(r'End\s*\[', dot, re.IGNORECASE):
        dot = _ensure_end_node(dot)

    # Ensure all paths reach End
    if not re.search(r'->\s*End', dot, re.IGNORECASE):
        last_brace = dot.rfind('}')
        if last_brace != -1:
            # Find last step node
            last_step = None
            for m in re.finditer(r'(Step\d+|Decision\d+|MoreItems\d*)', dot):
                last_step = m.group(1)
            if last_step:
                dot = (
                    dot[:last_brace] +
                    f'    {last_step} -> End;\n' +
                    dot[last_brace:]
                )

    return dot


# ============================================================
# DETERMINISTIC DOT GENERATION — guaranteed consistent output
# ============================================================

def _deterministic_dot(classified: List[Dict]) -> str:
    """
    Generate DOT code deterministically from classified steps.
    Same input always produces same output.
    No LLM involved — pure code logic.
    """
    lines = [
        'digraph ProcessFlow {',
        '    rankdir=TB;',
        '    node [fontname="Arial", fontsize=10, style=filled];',
        '    edge [fontname="Arial", fontsize=9];',
        '',
        '    Start [label="Start", shape=oval, fillcolor=lightgreen];'
    ]

    # Track nodes for edge building
    nodes = ['Start']  # List of node IDs in order
    edges = []         # List of (from, to, label) tuples
    loop_targets = []  # (loop_decision_id, loop_back_to_id)

    step_counter = 0
    decision_counter = 0

    for c in classified:
        step_type = c["type"]
        label = c["short_label"].replace('"', '\\"')

        # Word-wrap long labels
        if len(label) > 30:
            words = label.split()
            mid = len(words) // 2
            label = ' '.join(words[:mid]) + '\\n' + ' '.join(words[mid:])

        if step_type == "DECISION":
            decision_counter += 1
            node_id = f'Decision{decision_counter}'

            # Convert statement to question for diamond label
            question_label = _make_question(label)

            lines.append(
                f'    {node_id} [label="{question_label}", '
                f'shape=diamond, fillcolor=gold];'
            )
            nodes.append(node_id)

            # Decision creates a Yes path (continue) and No path
            # The No path will skip to the next non-decision step
            # For now, mark it — edges built after all nodes

        elif step_type == "LOOP":
            step_counter += 1
            node_id = f'Step{step_counter}'

            lines.append(
                f'    {node_id} [label="{label}", '
                f'shape=box, fillcolor=lightblue];'
            )
            nodes.append(node_id)

            # Add loop decision diamond after this step
            decision_counter += 1
            loop_decision_id = f'LoopCheck{decision_counter}'
            lines.append(
                f'    {loop_decision_id} [label="More Items?", '
                f'shape=diamond, fillcolor=gold];'
            )
            nodes.append(loop_decision_id)
            loop_targets.append((loop_decision_id, node_id))

        else:
            # PROCESS or END_PHASE
            step_counter += 1
            node_id = f'Step{step_counter}'

            lines.append(
                f'    {node_id} [label="{label}", '
                f'shape=box, fillcolor=lightblue];'
            )
            nodes.append(node_id)

    # Add End node
    lines.append(
        '    End [label="End", shape=oval, fillcolor=lightcoral];'
    )
    nodes.append('End')
    lines.append('')

    # ── Build edges ──
    i = 0
    while i < len(nodes) - 1:
        current = nodes[i]
        next_node = nodes[i + 1]

        # Check if current is a loop decision
        is_loop_decision = False
        for loop_dec_id, loop_back_id in loop_targets:
            if current == loop_dec_id:
                # Yes → loop back
                lines.append(
                    f'    {current} -> {loop_back_id} [label="Yes"];'
                )
                # No → continue to next
                lines.append(
                    f'    {current} -> {next_node} [label="No"];'
                )
                is_loop_decision = True
                break

        if not is_loop_decision:
            # Check if current is a regular decision
            if current.startswith('Decision'):
                # Yes → next step
                lines.append(
                    f'    {current} -> {next_node} [label="Yes"];'
                )
                # No → skip to the step after next (or End)
                skip_to = nodes[i + 2] if i + 2 < len(nodes) else 'End'
                lines.append(
                    f'    {current} -> {skip_to} [label="No"];'
                )
            else:
                # Regular edge
                lines.append(f'    {current} -> {next_node};')

        i += 1

    lines.append('}')
    return '\n'.join(lines)


def _make_question(label: str) -> str:
    """Convert a statement label into a question for decision diamonds."""
    label = label.rstrip('.')

    # Already a question
    if label.endswith('?'):
        return label

    # Common patterns
    lower = label.lower()

    if 'valid' in lower:
        return 'Valid?'
    if 'eligible' in lower:
        return 'Eligible?'
    if 'active' in lower:
        return 'Active?'
    if 'found' in lower:
        return 'Found?'
    if 'exist' in lower:
        return 'Exists?'
    if 'success' in lower:
        return 'Successful?'
    if 'fail' in lower:
        return 'Failed?'
    if 'match' in lower:
        return 'Matches?'
    if 'approv' in lower:
        return 'Approved?'
    if 'compli' in lower:
        return 'Compliant?'
    if 'meets criteria' in lower:
        return 'Meets Criteria?'

    # Generic: shorten and add ?
    words = label.split()
    if len(words) > 4:
        label = ' '.join(words[:4])
    return label + '?'


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _ensure_end_node(dot_code: str) -> str:
    """Add End node declaration if missing."""
    if not re.search(r'End\s*\[', dot_code, re.IGNORECASE):
        end_decl = (
            '    End [label="End", shape=oval, '
            'fillcolor=lightcoral];\n'
        )
        last_brace = dot_code.rfind('}')
        if last_brace != -1:
            dot_code = (
                dot_code[:last_brace] + end_decl +
                dot_code[last_brace:]
            )
    return dot_code


def _extract_dot(response: str) -> Optional[str]:
    """Extract DOT code from LLM response."""
    # Try code block first
    m = re.search(
        r'```(?:dot|graphviz|)?\s*\n?(.*?)```',
        response, re.DOTALL
    )
    if m and 'digraph' in m.group(1):
        return m.group(1).strip()

    # Try raw digraph block
    m = re.search(
        r'(digraph\s+\w*\s*\{.*\})',
        response, re.DOTALL
    )
    if m:
        return m.group(1).strip()

    # Try to reconstruct
    if 'digraph' in response and '->' in response:
        text = response.strip()
        if not text.startswith('digraph'):
            text = 'digraph ProcessFlow {\n' + text
        if not text.endswith('}'):
            text += '\n}'
        return text
    return None