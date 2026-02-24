# pdd_no_audio/llm_tasks/sop_sections.py

"""
PDD document section generation using text LLM (qwen2.5:14b).
"""

import re
import time
from typing import List, Dict

from pdd_no_audio.clients.text_llm import text_client
from pdd_no_audio.llm_tasks.system_prompts import PDD_SYSTEM_PROMPT
from pdd_no_audio.utils import timed, safe_sample, parse_numbered_steps


def generate_document_purpose(
    project_name: str,
    app_name: str,
    step_summaries: List[str]
) -> str:
    start = time.time()
    steps_text = "\n".join([f"- {s[:120]}" for s in step_summaries[:10]])

    prompt = f"""Write the "Purpose of this Document" section for a Process Definition Document (PDD).

Project: "{project_name}"
Application: "{app_name}"

The process involves these steps:
{steps_text}

Write 1-2 paragraphs covering:
- What this document defines (objectives, scope, requirements)
- What process is being documented for automation
- That it covers current manual state and future automated state
- That it serves as the basis for designing and deploying the solution

Use ONLY names from the information above. Formal business English. Third person.

PURPOSE:"""

    response = text_client.generate(prompt=prompt, system_prompt=PDD_SYSTEM_PROMPT, call_name="DocumentPurpose")
    timed("Purpose", start)

    if response and len(response) > 50:
        return response.strip()

    return (
        f"This Process Definition Document (PDD) defines the objectives, scope, "
        f"and detailed process requirements for the {project_name} automation initiative. "
        f"It documents the current manual process performed in {app_name or 'the application'} "
        f"and specifies the future automated state for implementation."
    )


def generate_overview_justification(
    project_name: str,
    app_name: str,
    step_summaries: List[str]
) -> Dict[str, str]:
    start = time.time()
    steps_text = "\n".join([f"- {s[:100]}" for s in step_summaries[:8]])

    prompt = f"""Write two sections for a Process Definition Document.

Project: "{project_name}"
Application: "{app_name}"

Process steps:
{steps_text}

===OVERVIEW===
Write an 'Overview and Objective' section:
- One paragraph stating the primary objective
- Then 4-6 bullet points of what the automation achieves
- Use: 'Ensure...', 'Standardize...', 'Reduce...', 'Improve...'

===JUSTIFICATION===
Write a 'Business Justification' section:
- Opening sentence about operational benefits
- Then 4-6 numbered items with **bold title** and description

Use ONLY names from above. Formal English.

===OVERVIEW==="""

    response = text_client.generate(prompt=prompt, system_prompt=PDD_SYSTEM_PROMPT, call_name="OverviewJustification")
    timed("Overview+Justification", start)

    result = {"overview": "", "justification": ""}
    if response:
        ov = re.search(r'===OVERVIEW===\s*(.*?)(?====JUSTIFICATION===|$)', response, re.DOTALL)
        jf = re.search(r'===JUSTIFICATION===\s*(.*?)$', response, re.DOTALL)
        if ov:
            result["overview"] = ov.group(1).strip()
        if jf:
            result["justification"] = jf.group(1).strip()

    if not result["overview"]:
        result["overview"] = f"The primary objective is to automate the {project_name} process to ensure consistency, accuracy, and compliance."
    if not result["justification"]:
        result["justification"] = f"The {project_name} delivers operational efficiency and governance control."
    return result


def generate_as_is_process(
    project_name: str,
    app_name: str,
    step_summaries: List[str]
) -> str:
    start = time.time()
    steps_text = "\n".join([f"- {s[:120]}" for s in step_summaries[:10]])

    prompt = f"""Document the CURRENT STATE ("As Is") for this process.

Project: "{project_name}"
Application: "{app_name}"

Observed process steps:
{steps_text}

Write 4-8 numbered steps. Each step:
- **Bold title**
- Description: what the person manually does
- Tools Used: applications used (ONLY from the information above)

Then add 'Business Challenges' with 4-6 bullet points about manual process issues.

CURRENT STATE:"""

    response = text_client.generate(prompt=prompt, system_prompt=PDD_SYSTEM_PROMPT, call_name="AsIsProcess")
    timed("As-Is", start)

    if response and len(response) > 100:
        return response.strip()
    return f"The current {project_name} process is performed manually in {app_name or 'the application'}."


def generate_to_be_process(
    project_name: str,
    app_name: str,
    step_summaries: List[str]
) -> str:
    start = time.time()
    steps_text = "\n".join([f"- {s[:120]}" for s in step_summaries[:10]])

    prompt = f"""Write the "To Be" / future automated state description.

Project: "{project_name}"
Application: "{app_name}"

Observed process steps:
{steps_text}

Write 2-3 paragraphs describing how the automation handles this process end-to-end:
- Write as if the automation already exists
- Use: 'The system will...', 'The automation will automatically...'
- Cover: trigger → connection → data handling → processing → validation → action → reporting
- Mention specific operations (data filtering, lookups, validations) from the steps
- End with audit readiness and compliance

FUTURE STATE:"""

    response = text_client.generate(prompt=prompt, system_prompt=PDD_SYSTEM_PROMPT, call_name="ToBeProcess")
    timed("To-Be", start)

    if response and len(response) > 100:
        return response.strip()
    return f"The {project_name} will use an automation solution to handle the end-to-end process in {app_name or 'the application'}."


def generate_prerequisites(
    project_name: str,
    app_name: str,
    vision_descriptions: List[str]
) -> List[Dict]:
    start = time.time()
    screen_context = "\n".join([f"Screen {i+1}: {safe_sample(d, 200)}" for i, d in enumerate(vision_descriptions[:5])])

    prompt = f"""Identify input requirements for this automation project.

Project: "{project_name}"
Application: "{app_name}"

Screens observed:
{screen_context}

List 3-8 inputs the automation needs. Consider:
- Credentials/access for applications
- Source data (files, databases, portals)
- Configuration parameters
- File paths or network locations

FORMAT (one per line):
INPUT: Parameter Name | DESCRIPTION: What it is and why needed

INPUTS:"""

    response = text_client.generate(prompt=prompt, system_prompt=PDD_SYSTEM_PROMPT, call_name="InputRequirements")
    timed("Inputs", start)

    inputs = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line and 'INPUT' in line.upper():
                parts = line.split('|')
                param, desc = "", ""
                for part in parts:
                    part = part.strip()
                    if part.upper().startswith('INPUT'):
                        param = re.sub(r'^INPUT[S]?\s*[:]\s*', '', part, flags=re.IGNORECASE).strip()
                    elif part.upper().startswith('DESC'):
                        desc = re.sub(r'^DESCRIPTION\s*[:]\s*', '', part, flags=re.IGNORECASE).strip()
                if param:
                    inputs.append({"parameter": param, "description": desc or "Required."})

    if not inputs:
        inputs = [
            {"parameter": "Application Credentials", "description": f"Authorized credentials for {app_name or 'the application'}."},
            {"parameter": "Source Data", "description": "Input data files or database access."},
            {"parameter": "Automation Configuration", "description": "Configured workflow parameters."},
        ]
    return inputs


def generate_exception_handling(
    project_name: str,
    app_name: str,
    step_descriptions: List[str]
) -> List[Dict]:
    start = time.time()
    steps_context = "\n".join([f"- {s[:80]}" for s in step_descriptions[:10]])

    prompt = f"""Write exception handling scenarios for this automation.

Project: "{project_name}"
Application: "{app_name}"

Process steps:
{steps_context}

List 5-8 technical exception scenarios and how the system handles each.
Consider: login failures, missing data, file not found, formula errors,
invalid data, processing errors, application timeout, network issues.

FORMAT (one per line):
EXCEPTION: Scenario title | HANDLING: What the system does

EXCEPTIONS:"""

    response = text_client.generate(prompt=prompt, system_prompt=PDD_SYSTEM_PROMPT, call_name="ExceptionHandling")
    timed("Exceptions", start)

    exceptions = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line and 'EXCEPTION' in line.upper():
                parts = line.split('|')
                exc = {"exception": "", "handling": ""}
                for part in parts:
                    part = part.strip()
                    if part.upper().startswith('EXCEPTION'):
                        exc["exception"] = re.sub(r'^EXCEPTION:\s*', '', part, flags=re.IGNORECASE).strip()
                    elif part.upper().startswith('HANDLING'):
                        exc["handling"] = re.sub(r'^HANDLING:\s*', '', part, flags=re.IGNORECASE).strip()
                if exc["exception"]:
                    exceptions.append(exc)

    if not exceptions:
        exceptions = [
            {"exception": "Application Login Failure", "handling": "Stop execution, log error, notify support."},
            {"exception": "Source File Not Found", "handling": "Log failure, send alert, halt process."},
            {"exception": "Invalid Data / Formula Error", "handling": "Log error details, skip affected record, continue."},
            {"exception": "Record Not Found", "handling": "Log details, skip, continue processing."},
            {"exception": "System Exception", "handling": "Capture details in error log for audit."},
        ]
    return exceptions


def generate_interface_requirements(
    app_name: str,
    vision_descriptions: List[str]
) -> List[Dict]:
    start = time.time()
    screen_context = "\n".join([f"Screen {i+1}: {safe_sample(d, 150)}" for i, d in enumerate(vision_descriptions[:5])])

    prompt = f"""Identify application interfaces needed for this automation.

Application observed: "{app_name}"

Screens observed:
{screen_context}

ONLY list applications or systems visible in the screen descriptions.

FORMAT (one per line):
APP: Name | INTERFACE: Web/Desktop/API/Database | PURPOSE: Why needed

INTERFACES:"""

    response = text_client.generate(prompt=prompt, system_prompt=PDD_SYSTEM_PROMPT, call_name="InterfaceRequirements")
    timed("Interfaces", start)

    apps = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line and 'APP' in line.upper():
                app = {"application": "", "interface": "", "url": "", "purpose": ""}
                parts = line.split('|')
                for part in parts:
                    part = part.strip()
                    p_up = part.upper()
                    if p_up.startswith('APP'):
                        app["application"] = re.sub(r'^APP(?:LICATION)?:\s*', '', part, flags=re.IGNORECASE).strip()
                    elif p_up.startswith('INTERFACE'):
                        app["interface"] = part.split(':', 1)[-1].strip()
                    elif p_up.startswith('PURPOSE'):
                        app["purpose"] = part.split(':', 1)[-1].strip()
                if app["application"]:
                    apps.append(app)

    if not apps and app_name:
        apps = [{"application": app_name, "interface": "Desktop/Web", "url": "N/A", "purpose": "Primary application for process execution"}]
    if not apps:
        apps = [{"application": "N/A", "interface": "N/A", "url": "N/A", "purpose": "N/A"}]
    return apps


def generate_flowchart_dot(
    steps: List[Dict],
    project_name: str
) -> str:
    """Generate DOT flowchart code from PDD steps."""
    start = time.time()

    # For long step lists, summarize to keep prompt manageable
    if len(steps) > 25:
        # Take first 10, middle 5, last 10
        mid = len(steps) // 2
        display_steps = steps[:10] + steps[mid-2:mid+3] + steps[-10:]
    else:
        display_steps = steps[:25]

    steps_text = "\n".join([
        f"{s['number']}. {s['description'][:80]}" for s in display_steps
    ])

    prompt = f"""Generate a Graphviz DOT flowchart for this process.

Process: "{project_name}"

Steps:
{steps_text}

FLOWCHART RULES:
1. Start with a Start node (oval, green) and end with an End node (oval, red)
2. Every node MUST have: label="Short Description" (max 5-6 words)
3. Steps: shape=box, fillcolor=lightblue
4. Use node IDs like Step1, Step2, etc.
5. Connect steps in order with arrows
6. Use style=filled for all nodes
7. Keep node count under 25 — group related steps if needed

OUTPUT ONLY the DOT code:
```dot
digraph ProcessFlow {{
    rankdir=TB;
    dpi=300;
    node [fontname="Arial", fontsize=11, style=filled];

    Start [label="Start", shape=oval, fillcolor=lightgreen];
    Step1 [label="...", shape=box, fillcolor=lightblue];
    ...
    End [label="End", shape=oval, fillcolor=lightcoral];

    Start -> Step1;
    Step1 -> Step2;
    ...
}}
```"""

    response = text_client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.2,
        call_name="FlowchartDOT"
    )
    timed("Flowchart", start)

    if response:
        dot = _extract_dot(response)
        if dot:
            # Ensure high DPI in DOT code
            dot = _ensure_high_dpi(dot)
            return dot

    return _build_simple_dot(steps, project_name)


def _extract_dot(response: str) -> str:
    m = re.search(r'```(?:dot|graphviz|)?\s*\n?(.*?)```', response, re.DOTALL)
    if m and 'digraph' in m.group(1):
        return m.group(1).strip()
    m = re.search(r'(digraph\s+\w*\s*\{.*\})', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _ensure_high_dpi(dot_code: str) -> str:
    """Ensure DOT code has high DPI setting."""
    if 'dpi' not in dot_code.lower():
        # Insert dpi after opening brace
        match = re.search(r'(digraph\s+\w*\s*\{)', dot_code)
        if match:
            insert_pos = match.end()
            dot_code = dot_code[:insert_pos] + '\n    dpi=300;' + dot_code[insert_pos:]
    else:
        # Replace existing dpi with 300
        dot_code = re.sub(r'dpi\s*=\s*\d+', 'dpi=300', dot_code)
    return dot_code


def _build_simple_dot(steps: List[Dict], project_name: str) -> str:
    # Group steps if too many
    display_steps = steps
    if len(steps) > 25:
        # Keep first 8, middle 4, last 8
        mid = len(steps) // 2
        display_steps = steps[:8] + steps[mid-2:mid+2] + steps[-8:]

    lines = [
        'digraph ProcessFlow {',
        '    rankdir=TB;',
        '    dpi=300;',
        '    node [fontname="Arial", fontsize=11, style=filled];',
        '    edge [fontname="Arial", fontsize=9];',
        '',
        '    Start [label="Start", shape=oval, fillcolor=lightgreen];'
    ]
    for i, s in enumerate(display_steps):
        label = s["description"][:40].replace('"', '\\"')
        lines.append(f'    Step{i+1} [label="{label}", shape=box, fillcolor=lightblue];')
    lines.append('    End [label="End", shape=oval, fillcolor=lightcoral];')
    lines.append('')
    lines.append('    Start -> Step1;')
    for i in range(len(display_steps) - 1):
        lines.append(f'    Step{i+1} -> Step{i+2};')
    lines.append(f'    Step{len(display_steps)} -> End;')
    lines.append('}')
    return '\n'.join(lines)