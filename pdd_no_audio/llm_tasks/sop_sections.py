# pdd_no_audio/llm_tasks/sop_sections.py

"""
PDD document section generation using text LLM (qwen2.5:14b).
Includes adaptive flowchart layout and parallel section generation.
FIXED: Improved prompts to prevent instruction leakage.
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


def _sanitize_section_output(text: str) -> str:
    """Remove any instruction echoes from section outputs."""
    if not text:
        return ""
    
    # Remove common instruction patterns
    patterns = [
        r'^Write\s+\d+-\d+\s+.*?(?=\n|$)',
        r'^Do\s+NOT\s+.*?(?=\n|$)',
        r'^INSTRUCTIONS?:.*?(?=\n\n|$)',
        r'^RULES?:.*?(?=\n\n|$)',
        r'^OUTPUT:?\s*',
        r'^SECTION\s*\d+[:\s]*',
        r'(?:^|\n)Note:\s*.*?(?=\n|$)',
    ]
    
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


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

Key process steps:
{steps_text}

Write 2-3 paragraphs explaining:
- What this document defines
- Who should use it (developers, business analysts, QA)
- Scope of the automation

Write in formal third person. Output only the section content."""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="DocumentPurpose"
    )
    return _sanitize_section_output(response) if response else ""


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

    prompt = f"""Write two sections for a PDD:

Project: "{project_name}"
Application: "{app_name}"

Process steps:
{steps_text}

=== OVERVIEW ===
Write 3-5 bullet points (using •) describing what this automation does.

=== JUSTIFICATION ===
Write 2-3 paragraphs explaining why this process should be automated.

Output both sections with the === headers."""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="OverviewJustification"
    )

    result = {"overview": "", "justification": ""}
    if response:
        # Parse the two sections
        overview_match = re.search(
            r'===\s*OVERVIEW\s*===\s*(.*?)(?====\s*JUSTIFICATION|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        justification_match = re.search(
            r'===\s*JUSTIFICATION\s*===\s*(.*?)$',
            response, re.DOTALL | re.IGNORECASE
        )
        
        if overview_match:
            result["overview"] = _sanitize_section_output(overview_match.group(1))
        if justification_match:
            result["justification"] = _sanitize_section_output(justification_match.group(1))
        
        # Fallback if no markers found
        if not result["overview"] and not result["justification"]:
            parts = re.split(r'\n\n+', response, maxsplit=1)
            if len(parts) >= 2:
                result["overview"] = _sanitize_section_output(parts[0])
                result["justification"] = _sanitize_section_output(parts[1])
            else:
                result["overview"] = _sanitize_section_output(response)

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

The automated process includes these steps:
{steps_text}

Describe how this process is CURRENTLY done MANUALLY before automation:
- What manual steps does a human perform?
- What tools do they use?
- What are the pain points?

Write 2-3 paragraphs. Output only the section content."""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="AsIsProcess"
    )
    return _sanitize_section_output(response) if response else ""


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

Automated steps:
{steps_text}

Describe the AUTOMATED process:
- How the bot executes each phase
- What triggers the process
- How exceptions are handled
- What outputs are produced

Write 2-3 paragraphs. Output only the section content."""

    response = client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="ToBeProcess"
    )
    return _sanitize_section_output(response) if response else ""


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

    prompt = f"""List the INPUT REQUIREMENTS for this automation.

Project: "{project_name}"
Application: "{app_name}"

Screens observed:
{desc_sample}

List each input in this format:
Parameter Name | Description

Include: credentials, file paths, URLs, config values, etc.
List 5-10 inputs. Output only the list."""

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

Process steps:
{steps_sample}

For each exception use this format:
Exception | Handling Action

Include: login failures, timeouts, missing data, application errors.
List 6-10 exceptions. Output only the list."""

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

    prompt = f"""List the INTERFACE REQUIREMENTS (applications/systems) for this automation.

Application: "{app_name}"

Screens observed:
{desc_sample}

For each interface use this format:
Application Name | Purpose

List 3-6 interfaces. Output only the list."""

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

    # For large step counts, sample steps
    if num_steps > 25:
        mid = num_steps // 2
        display_steps = steps[:10] + steps[mid-2:mid+3] + steps[-10:]
    else:
        display_steps = steps[:25]

    steps_text = "\n".join([
        f"{s['number']}. {s['description'][:80]}" for s in display_steps
    ])

    # Determine layout based on step count
    if num_steps > 20:
        rankdir = "LR"
        size_hint = "size=\"12,8\""
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

Requirements:
- Start with Start node (oval, green fill)
- End with End node (oval, red fill)
- Each step as a box with short label (max 6 words)
- Use node IDs: Step1, Step2, etc.
- Connect nodes with arrows in order
- Layout: rankdir={rankdir}

Output ONLY the DOT code starting with "digraph" and ending with "}}"."""

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
    """Extract DOT code from LLM response."""
    # Try to find code block
    m = re.search(r'```(?:dot|graphviz|)?\s*\n?(.*?)```', response, re.DOTALL)
    if m and 'digraph' in m.group(1):
        return m.group(1).strip()
    
    # Try to find raw digraph
    m = re.search(r'(digraph\s+\w*\s*\{.*\})', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    
    return ""


def _ensure_layout(dot_code: str, rankdir: str, size_hint: str, num_steps: int) -> str:
    """Ensure DOT code has proper layout settings."""
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
    """Build a simple DOT flowchart as fallback."""
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
        # Truncate label and escape quotes
        label = s["description"][:40].replace('"', '\\"').replace('\n', ' ')
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