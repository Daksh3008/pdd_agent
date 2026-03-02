# pdd_no_audio/llm_tasks/sop_sections.py

"""
PDD document section generation using text LLM (qwen2.5:14b).
Includes adaptive flowchart layout and parallel section generation.
"""

import re
import time
import concurrent.futures
from typing import List, Dict, Any

from pdd_no_audio.clients.text_llm import TextLLMClient
from pdd_no_audio.config import text_config, llm_params
from pdd_no_audio.llm_tasks.system_prompts import PDD_SYSTEM_PROMPT
from pdd_no_audio.utils import timed, safe_sample, parse_numbered_steps


def _worker_client() -> TextLLMClient:
    """Create a fresh TextLLMClient for parallel workers."""
    return TextLLMClient()


# ============================================================
# Individual section generators
# ============================================================

def generate_document_purpose(
    project_name: str,
    app_name: str,
    step_summaries: List[str],
    client: TextLLMClient = None
) -> str:
    if client is None:
        from pdd_no_audio.clients.text_llm import text_client
        client = text_client

    steps_text = "\n".join(f"- {s[:100]}" for s in step_summaries[:15])

    prompt = f"""Write a "Purpose of this Document" section for a Process Definition Document.

Project: "{project_name}"
Application: "{app_name}"
Key Steps:
{steps_text}

Write 2-3 professional paragraphs explaining:
1. What this document defines
2. Who should use it (developers, BA, QA)
3. Scope of the automation

Do NOT use bullet points. Write in formal third person."""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="DocumentPurpose"
    )
    return response.strip() if response else ""


def generate_overview_justification(
    project_name: str,
    app_name: str,
    step_summaries: List[str],
    client: TextLLMClient = None
) -> Dict[str, str]:
    if client is None:
        from pdd_no_audio.clients.text_llm import text_client
        client = text_client

    steps_text = "\n".join(f"- {s[:80]}" for s in step_summaries[:12])

    prompt = f"""Write TWO sections for a PDD:

Project: "{project_name}"
Application: "{app_name}"
Steps:
{steps_text}

SECTION 1 — OVERVIEW AND OBJECTIVE:
Write 3-5 bullet points starting with "•" describing what the automation does.

SECTION 2 — BUSINESS JUSTIFICATION:
Write 2-3 paragraphs explaining why this process should be automated (efficiency, accuracy, compliance).

Separate the two sections clearly with headers."""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="OverviewJustification"
    )

    result = {"overview": "", "justification": ""}
    if response:
        parts = re.split(
            r'(?:SECTION\s*2|BUSINESS\s*JUSTIFICATION|Justification)',
            response, maxsplit=1, flags=re.IGNORECASE
        )
        if len(parts) >= 2:
            result["overview"] = parts[0].strip()
            result["justification"] = parts[1].strip()
        else:
            result["overview"] = response.strip()

    for key in result:
        result[key] = re.sub(
            r'^(?:SECTION\s*\d|OVERVIEW|OBJECTIVE)[:\-—]*\s*',
            '', result[key], flags=re.IGNORECASE
        ).strip()

    return result


def generate_as_is_process(
    project_name: str,
    app_name: str,
    step_summaries: List[str],
    client: TextLLMClient = None
) -> str:
    if client is None:
        from pdd_no_audio.clients.text_llm import text_client
        client = text_client

    steps_text = "\n".join(f"- {s[:80]}" for s in step_summaries[:10])

    prompt = f"""Write the "As Is" (current manual process) section for a PDD.

Project: "{project_name}"
Application: "{app_name}"
Automated Steps:
{steps_text}

Describe how this process is CURRENTLY done MANUALLY:
- What manual steps does a human perform?
- What tools do they use?
- What are the pain points (slow, error-prone, repetitive)?

Write 2-3 paragraphs in formal tone."""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="AsIsProcess"
    )
    return response.strip() if response else ""


def generate_to_be_process(
    project_name: str,
    app_name: str,
    step_summaries: List[str],
    client: TextLLMClient = None
) -> str:
    if client is None:
        from pdd_no_audio.clients.text_llm import text_client
        client = text_client

    steps_text = "\n".join(f"- {s[:80]}" for s in step_summaries[:10])

    prompt = f"""Write the "To Be" (automated process) section for a PDD.

Project: "{project_name}"
Application: "{app_name}"
Automated Steps:
{steps_text}

Describe the AUTOMATED process:
- How the bot/automation executes each phase
- What triggers the process
- How exceptions are handled
- What outputs are produced

Write 2-3 paragraphs in formal tone."""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="ToBeProcess"
    )
    return response.strip() if response else ""


def generate_prerequisites(
    project_name: str,
    app_name: str,
    vision_descriptions: List[str],
    client: TextLLMClient = None
) -> List[Dict]:
    if client is None:
        from pdd_no_audio.clients.text_llm import text_client
        client = text_client

    desc_sample = "\n".join(safe_sample(d, 100) for d in vision_descriptions[:8])

    prompt = f"""List the INPUT REQUIREMENTS for automating this process.

Project: "{project_name}"
Application: "{app_name}"
Screens observed:
{desc_sample}

List each input as:
1. Parameter Name | Description
2. Parameter Name | Description

Include: credentials, file paths, URLs, config values, email recipients, etc.
List 5-10 inputs."""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="Prerequisites"
    )

    inputs = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|', 1)
                param = re.sub(r'^\d+[\.\)]\s*', '', parts[0]).strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                if param and len(param) > 2:
                    inputs.append({"parameter": param, "description": desc})
            elif re.match(r'^\d+[\.\)]\s*', line) and len(line) > 10:
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                if ':' in cleaned:
                    parts = cleaned.split(':', 1)
                    inputs.append({
                        "parameter": parts[0].strip(),
                        "description": parts[1].strip()
                    })
                else:
                    inputs.append({"parameter": cleaned, "description": ""})

    return inputs if inputs else [
        {"parameter": "Application URL", "description": f"URL for {app_name or 'the application'}"},
        {"parameter": "User Credentials", "description": "Username and password for authentication"},
    ]


def generate_exception_handling(
    project_name: str,
    app_name: str,
    step_descriptions: List[str],
    client: TextLLMClient = None
) -> List[Dict]:
    if client is None:
        from pdd_no_audio.clients.text_llm import text_client
        client = text_client

    steps_sample = "\n".join(f"- {s[:80]}" for s in step_descriptions[:10])

    prompt = f"""List exception handling scenarios for this automation.

Project: "{project_name}"
Application: "{app_name}"
Steps:
{steps_sample}

For each exception:
Exception: <scenario> | Handling: <action>

Include: login failures, timeouts, missing data, application errors, network issues, session expiry.
List 6-10 exceptions."""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="ExceptionHandling"
    )

    exceptions = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|', 1)
                exc = re.sub(
                    r'^(?:Exception|Scenario):\s*', '',
                    parts[0], flags=re.IGNORECASE
                ).strip()
                exc = re.sub(r'^\d+[\.\)]\s*', '', exc).strip()
                handling = re.sub(
                    r'^(?:Handling|Action):\s*', '',
                    parts[1], flags=re.IGNORECASE
                ).strip()
                if exc and len(exc) > 5:
                    exceptions.append({"exception": exc, "handling": handling})

    return exceptions if exceptions else [
        {
            "exception": "Application Login Failure",
            "handling": "Retry login up to 3 times. If still failing, stop execution and notify support."
        },
        {
            "exception": "Session Timeout / Expiry",
            "handling": "Re-authenticate and resume from last checkpoint."
        },
        {
            "exception": "Element Not Found",
            "handling": "Wait up to 30 seconds, retry. If not found, log error and skip."
        },
    ]


def generate_interface_requirements(
    app_name: str,
    vision_descriptions: List[str],
    client: TextLLMClient = None
) -> List[Dict]:
    if client is None:
        from pdd_no_audio.clients.text_llm import text_client
        client = text_client

    desc_sample = "\n".join(safe_sample(d, 80) for d in vision_descriptions[:6])

    prompt = f"""List the INTERFACE REQUIREMENTS (applications/systems) needed for this automation.

Application: "{app_name}"
Screens:
{desc_sample}

For each interface:
1. Application/System Name | Purpose/Role in automation

List 3-6 interfaces."""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="InterfaceReqs"
    )

    interfaces = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|', 1)
                app = re.sub(r'^\d+[\.\)]\s*', '', parts[0]).strip()
                purpose = parts[1].strip() if len(parts) > 1 else ""
                if app and len(app) > 2:
                    interfaces.append({"application": app, "purpose": purpose})

    return interfaces if interfaces else [
        {
            "application": app_name or "Target Application",
            "purpose": "Primary application for process execution"
        },
    ]


# ============================================================
# Parallel section generation
# ============================================================

def generate_all_sections_parallel(
    project_name: str,
    app_name: str,
    step_descriptions: List[str],
    vision_descriptions: List[str]
) -> Dict[str, Any]:
    """
    Generate all PDD sections in parallel using thread pool.
    Returns dict with all section results.
    """
    start = time.time()
    results = {}

    def _run_purpose():
        c = _worker_client()
        return generate_document_purpose(project_name, app_name, step_descriptions, client=c)

    def _run_overview():
        c = _worker_client()
        return generate_overview_justification(project_name, app_name, step_descriptions, client=c)

    def _run_as_is():
        c = _worker_client()
        return generate_as_is_process(project_name, app_name, step_descriptions, client=c)

    def _run_to_be():
        c = _worker_client()
        return generate_to_be_process(project_name, app_name, step_descriptions, client=c)

    def _run_prereqs():
        c = _worker_client()
        return generate_prerequisites(project_name, app_name, vision_descriptions, client=c)

    def _run_exceptions():
        c = _worker_client()
        return generate_exception_handling(project_name, app_name, step_descriptions, client=c)

    def _run_interfaces():
        c = _worker_client()
        return generate_interface_requirements(app_name, vision_descriptions, client=c)

    tasks = {
        "purpose": _run_purpose,
        "overview_justification": _run_overview,
        "as_is": _run_as_is,
        "to_be": _run_to_be,
        "prerequisites": _run_prereqs,
        "exceptions": _run_exceptions,
        "interfaces": _run_interfaces,
    }

    workers = min(llm_params.text_llm_workers, len(tasks))
    print(f"    [Sections] Generating {len(tasks)} sections ({workers} parallel workers)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
                print(f"    [Sections] ✓ {name}")
            except Exception as e:
                print(f"    [Sections] ✗ {name}: {e}")
                results[name] = None

    timed(f"All sections ({len(results)})", start)
    return results


# ============================================================
# Flowchart DOT generation
# ============================================================

def generate_flowchart_dot(
    steps: List[Dict],
    project_name: str
) -> str:
    """Generate DOT flowchart code with adaptive layout based on step count."""
    from pdd_no_audio.clients.text_llm import text_client

    start = time.time()
    num_steps = len(steps)

    if num_steps > 25:
        mid = num_steps // 2
        display_steps = steps[:10] + steps[mid-2:mid+3] + steps[-10:]
    else:
        display_steps = steps[:25]

    steps_text = "\n".join([
        f"{s['number']}. {s['description'][:80]}" for s in display_steps
    ])

    if num_steps > 20:
        rankdir = "LR"
        size_hint = "size=\"10,8\""
    elif num_steps > 10:
        rankdir = "TB"
        size_hint = "size=\"8,10\""
    else:
        rankdir = "TB"
        size_hint = "size=\"8,10\""

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
7. Layout: rankdir={rankdir}, {size_hint}
8. For LOGIN/LOGOUT steps, use fillcolor=lightyellow and shape=diamond

OUTPUT ONLY the DOT code:
```dot
digraph ProcessFlow {{
    rankdir={rankdir};
    {size_hint};
    dpi=300;
    node [fontname="Arial", fontsize=11, style=filled];
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
            dot = _ensure_layout(dot, rankdir, size_hint, num_steps)
            return dot

    return _build_simple_dot(steps, project_name, rankdir, size_hint)


def _extract_dot(response: str) -> str:
    m = re.search(r'```(?:dot|graphviz|)?\s*\n?(.*?)```', response, re.DOTALL)
    if m and 'digraph' in m.group(1):
        return m.group(1).strip()
    m = re.search(r'(digraph\s+\w*\s*\{.*\})', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _ensure_layout(dot_code: str, rankdir: str, size_hint: str, num_steps: int) -> str:
    if 'rankdir' not in dot_code:
        match = re.search(r'(digraph\s+\w*\s*\{)', dot_code)
        if match:
            insert_pos = match.end()
            dot_code = (dot_code[:insert_pos] +
                       f'\n    rankdir={rankdir};' +
                       dot_code[insert_pos:])

    if 'size=' not in dot_code:
        match = re.search(r'(digraph\s+\w*\s*\{)', dot_code)
        if match:
            insert_pos = match.end()
            dot_code = (dot_code[:insert_pos] +
                       f'\n    {size_hint};' +
                       dot_code[insert_pos:])

    return dot_code


def _build_simple_dot(steps: List[Dict], project_name: str,
                      rankdir: str = "TB",
                      size_hint: str = "size=\"8,10\"") -> str:
    display_steps = steps
    if len(steps) > 25:
        mid = len(steps) // 2
        display_steps = steps[:8] + steps[mid-2:mid+2] + steps[-8:]

    lines = [
        f'digraph ProcessFlow {{',
        f'    rankdir={rankdir};',
        f'    {size_hint};',
        '    dpi=300;',
        '    node [fontname="Arial", fontsize=11, style=filled];',
        '    edge [fontname="Arial", fontsize=9];',
        '',
        '    Start [label="Start", shape=oval, fillcolor=lightgreen];'
    ]
    for i, s in enumerate(display_steps):
        label = s["description"][:40].replace('"', '\\"')
        auth_info = s.get("auth_info", {})
        if auth_info and auth_info.get("is_auth"):
            lines.append(
                f'    Step{i+1} [label="{label}", '
                f'shape=diamond, fillcolor=lightyellow];'
            )
        else:
            lines.append(
                f'    Step{i+1} [label="{label}", '
                f'shape=box, fillcolor=lightblue];'
            )
    lines.append('    End [label="End", shape=oval, fillcolor=lightcoral];')
    lines.append('')
    lines.append('    Start -> Step1;')
    for i in range(len(display_steps) - 1):
        lines.append(f'    Step{i+1} -> Step{i+2};')
    lines.append(f'    Step{len(display_steps)} -> End;')
    lines.append('}')
    return '\n'.join(lines)