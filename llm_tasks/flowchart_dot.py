# llm/flowchart_dot.py

"""
DOT flowchart code generation.
Two approaches:
1. From transcript (audio pipeline) - classify and generate
2. From PDD steps (video pipeline) - direct step-to-node mapping
"""

import re
import time
from typing import Dict, List, Optional, Tuple

from core.gemini_client import gemini_client
from core.config import config
from core.utils import timed, filter_conversation_steps
from llm_tasks.system_prompts import get_system_prompt
from llm_tasks.requirements import get_interface_requirements
from llm_tasks.entity_extraction import extract_entities_and_project


# ============================================================
# Step Classification (Deterministic)
# ============================================================

DECISION_KEYWORDS = [
    'if ', 'whether', 'check if', 'verify if', 'determine if',
    'validate if', 'confirm if', 'eligible', 'meets criteria',
    'qualifies', 'is valid', 'is active', 'is inactive',
    'found', 'not found', 'exists', 'does not exist',
    'successful', 'failed', 'pass', 'fail', 'approved', 'rejected',
]

LOOP_KEYWORDS = [
    'each', 'every', 'all ', 'iterate', 'repeat', 'loop',
    'next record', 'next item', 'next entry', 'next user',
    'for all', 'processes each', 'validates each',
]

END_PHASE_KEYWORDS = [
    'final report', 'final status', 'execution status',
    'completion', 'summary report', 'audit log',
    'logs out', 'closes', 'terminates', 'ends',
]


def classify_steps(steps: List[str]) -> List[Dict]:
    """Classify each step as PROCESS, DECISION, LOOP, or END_PHASE."""
    classified = []

    for i, step in enumerate(steps):
        step_lower = step.lower()
        step_type = "PROCESS"

        if any(kw in step_lower for kw in DECISION_KEYWORDS):
            step_type = "DECISION"
        elif any(kw in step_lower for kw in LOOP_KEYWORDS):
            step_type = "LOOP"
        elif any(kw in step_lower for kw in END_PHASE_KEYWORDS):
            step_type = "END_PHASE"

        short_label = _shorten_label(step)
        classified.append({
            "index": i + 1,
            "text": step,
            "type": step_type,
            "short_label": short_label
        })

    type_counts = {}
    for c in classified:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
    print(f"    [Classify] {len(classified)} steps: {type_counts}")

    return classified


def _shorten_label(text: str, max_words: int = 6) -> str:
    """Shorten step text to a flowchart-friendly label."""
    text = re.sub(
        r'^(the system|the automation|the bot|the solution|it)\s+',
        '', text, flags=re.IGNORECASE
    ).strip()
    text = text.rstrip('.')
    if text:
        text = text[0].upper() + text[1:]
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words]) + '...'
    return text


# ============================================================
# Audio Pipeline: From Transcript
# ============================================================

def generate_dot_and_apps(
    process_steps: List[str],
    transcript: str,
    entities: Dict = None
) -> Tuple[str, List[Dict]]:
    """Generate DOT flowchart code and interface requirements from transcript."""
    start = time.time()
    if entities is None:
        entities, _ = extract_entities_and_project(transcript)
    dot_code = _generate_dot_from_steps_list(process_steps)
    apps = get_interface_requirements(transcript, entities)
    timed("DOT+Apps", start)
    return dot_code, apps


def _generate_dot_from_steps_list(steps: List[str]) -> str:
    """Generate DOT code from a list of step strings."""
    if not steps:
        return _deterministic_dot([{
            "index": 1, "text": "Process",
            "type": "PROCESS", "short_label": "Process"
        }])

    filtered = filter_conversation_steps(steps)
    classified = classify_steps(filtered[:15])

    if not classified:
        return _deterministic_dot([{
            "index": 1, "text": "Process",
            "type": "PROCESS", "short_label": "Process"
        }])

    return _deterministic_dot(classified)


# ============================================================
# Video Pipeline: From PDD Steps (dicts)
# ============================================================

def generate_flowchart_dot_from_steps(
    pdd_steps: List[Dict],
    project_name: str
) -> str:
    """Generate DOT flowchart code from PDD step dicts."""
    start = time.time()

    if not pdd_steps:
        return ""

    # Extract step descriptions
    descriptions = [s.get("description", "") for s in pdd_steps]
    
    # Classify
    classified = classify_steps(descriptions[:20])

    if not classified:
        timed("Flowchart", start)
        return ""

    # Determine layout based on step count
    num_steps = len(classified)
    if num_steps > 20:
        rankdir = "LR"
        size_hint = 'size="12,8"'
    else:
        rankdir = "TB"
        size_hint = 'size="8,10"'

    dot_code = _deterministic_dot(classified, rankdir=rankdir, size_hint=size_hint)
    timed("Flowchart", start)
    return dot_code


# ============================================================
# Deterministic DOT Generation
# ============================================================

def _deterministic_dot(
    classified: List[Dict],
    rankdir: str = "TB",
    size_hint: str = 'size="8,10"'
) -> str:
    """Generate DOT code deterministically from classified steps."""
    lines = [
        'digraph ProcessFlow {',
        f'    rankdir={rankdir};',
        f'    {size_hint};',
        '    dpi=300;',
        '    node [fontname="Arial", fontsize=10, style=filled];',
        '    edge [fontname="Arial", fontsize=9];',
        '',
        '    Start [label="Start", shape=oval, fillcolor=lightgreen];'
    ]

    nodes = ['Start']
    loop_targets = []
    step_counter = 0
    decision_counter = 0

    for c in classified:
        step_type = c["type"]
        label = c["short_label"].replace('"', '\\"')

        if len(label) > 30:
            words = label.split()
            mid = len(words) // 2
            label = ' '.join(words[:mid]) + '\\n' + ' '.join(words[mid:])

        if step_type == "DECISION":
            decision_counter += 1
            node_id = f'Decision{decision_counter}'
            question_label = _make_question(label)
            lines.append(
                f'    {node_id} [label="{question_label}", '
                f'shape=diamond, fillcolor=gold];'
            )
            nodes.append(node_id)

        elif step_type == "LOOP":
            step_counter += 1
            node_id = f'Step{step_counter}'
            lines.append(
                f'    {node_id} [label="{label}", '
                f'shape=box, fillcolor=lightblue];'
            )
            nodes.append(node_id)

            decision_counter += 1
            loop_decision_id = f'LoopCheck{decision_counter}'
            lines.append(
                f'    {loop_decision_id} [label="More Items?", '
                f'shape=diamond, fillcolor=gold];'
            )
            nodes.append(loop_decision_id)
            loop_targets.append((loop_decision_id, node_id))

        else:
            step_counter += 1
            node_id = f'Step{step_counter}'
            lines.append(
                f'    {node_id} [label="{label}", '
                f'shape=box, fillcolor=lightblue];'
            )
            nodes.append(node_id)

    lines.append('    End [label="End", shape=oval, fillcolor=lightcoral];')
    nodes.append('End')
    lines.append('')

    # Build edges
    i = 0
    while i < len(nodes) - 1:
        current = nodes[i]
        next_node = nodes[i + 1]

        is_loop_decision = False
        for loop_dec_id, loop_back_id in loop_targets:
            if current == loop_dec_id:
                lines.append(f'    {current} -> {loop_back_id} [label="Yes"];')
                lines.append(f'    {current} -> {next_node} [label="No"];')
                is_loop_decision = True
                break

        if not is_loop_decision:
            if current.startswith('Decision'):
                lines.append(f'    {current} -> {next_node} [label="Yes"];')
                skip_to = nodes[i + 2] if i + 2 < len(nodes) else 'End'
                lines.append(f'    {current} -> {skip_to} [label="No"];')
            else:
                lines.append(f'    {current} -> {next_node};')

        i += 1

    lines.append('}')
    return '\n'.join(lines)


def _make_question(label: str) -> str:
    """Convert a statement label into a question for decision diamonds."""
    label = label.rstrip('.')

    if label.endswith('?'):
        return label

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

    words = label.split()
    if len(words) > 4:
        label = ' '.join(words[:4])
    return label + '?'