# llm/document_sections.py

"""
Document section generation.
Purpose, Overview/Justification, As-Is Process, To-Be Process.
Supports both audio (transcript) and video (step summaries) pipelines.
Includes parallel generation for video pipeline.
"""

import re
import time
import concurrent.futures
from typing import Dict, List, Any

from core.gemini_client import gemini_client, GeminiClient
from core.config import config
from core.utils import timed, safe_sample
from llm_tasks.system_prompts import get_system_prompt, PDD_SYSTEM_PROMPT


def _sanitize_section_output(text: str) -> str:
    """Remove any instruction echoes from section outputs."""
    if not text:
        return ""
    patterns = [
        r'^Write\s+\d+-\d+\s+.*?(?=\n|$)',
        r'^Do\s+NOT\s+.*?(?=\n|$)',
        r'^INSTRUCTIONS?:.*?(?=\n\n|$)',
        r'^OUTPUT:?\s*',
        r'^SECTION\s*\d+[:\s]*',
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


# ============================================================
# Audio Pipeline Sections (from transcript)
# ============================================================

def get_document_purpose_text(
    transcript: str,
    project_name: str,
    entity_hint: str
) -> str:
    """Generate the Purpose of this Document section."""
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_small)
    doc_type = config.document.document_type
    doc_full = config.document.document_type_full

    if doc_type == "BRD":
        focus = (
            "- What business requirements this document captures\n"
            "- What business outcomes the automation must achieve\n"
            "- That it defines functional and non-functional requirements"
        )
    else:
        focus = (
            "- What this document defines (objectives, scope, requirements)\n"
            "- What process is being documented for automation\n"
            "- That it covers current manual state and future automated state"
        )

    prompt = f"""Write the "Purpose of this Document" section for a {doc_full}.

Project: "{project_name}". {entity_hint}

Write 1-2 paragraphs covering:
{focus}
- That it serves as the basis for designing and deploying the solution

Use ONLY names from the transcript. Formal business English. Third person.

TRANSCRIPT:
{sample}

PURPOSE:"""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="DocumentPurpose"
    )
    timed("Purpose", start)

    if response and len(response) > 50:
        return _sanitize_section_output(response)

    return (
        f"This {doc_full} ({doc_type}) defines the "
        f"objectives, scope, and detailed business requirements for the "
        f"{project_name} initiative."
    )


def get_overview_and_justification(
    transcript: str,
    project_name: str,
    entity_hint: str
) -> Dict[str, str]:
    """Generate Overview/Objective and Business Justification."""
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)
    doc_full = config.document.document_type_full
    doc_type = config.document.document_type

    if doc_type == "BRD":
        overview_instruction = (
            "Write an 'Overview and Objective' section:\n"
            "- One paragraph stating the business need and objective\n"
            "- Then 4-6 bullet points of business outcomes expected"
        )
        justification_instruction = (
            "Write a 'Business Justification' section:\n"
            "- Opening sentence about business value\n"
            "- Then 4-6 numbered items with **bold title** and description"
        )
    else:
        overview_instruction = (
            "Write an 'Overview and Objective' section:\n"
            "- One paragraph stating the primary objective\n"
            "- Then 4-6 bullet points of what the automation achieves"
        )
        justification_instruction = (
            "Write a 'Business Justification' section:\n"
            "- Opening sentence about operational benefits\n"
            "- Then 4-6 numbered items with **bold title** and description"
        )

    prompt = f"""Write two sections for a {doc_full}.

Project: "{project_name}". {entity_hint}

===OVERVIEW===
{overview_instruction}

===JUSTIFICATION===
{justification_instruction}

Use ONLY names from the transcript.

TRANSCRIPT:
{sample}"""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="OverviewJustification"
    )
    timed("Overview+Justification", start)

    result = {"overview": "", "justification": ""}
    if response:
        ov = re.search(
            r'===OVERVIEW===\s*(.*?)(?====JUSTIFICATION===|$)',
            response, re.DOTALL
        )
        jf = re.search(r'===JUSTIFICATION===\s*(.*?)$', response, re.DOTALL)
        if ov:
            result["overview"] = _sanitize_section_output(ov.group(1))
        if jf:
            result["justification"] = _sanitize_section_output(jf.group(1))

    if not result["overview"]:
        result["overview"] = (
            f"The primary objective is to automate the {project_name} "
            f"process to ensure consistency, accuracy, and compliance."
        )
    if not result["justification"]:
        result["justification"] = (
            f"The {project_name} delivers operational efficiency "
            f"and governance control."
        )
    return result


def get_as_is_process(
    transcript: str,
    project_name: str,
    entity_hint: str
) -> str:
    """Generate the current manual process (As Is state)."""
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)
    doc_type = config.document.document_type

    if doc_type == "BRD":
        instruction = (
            "Write 4-8 numbered items describing current business pain points:\n"
            "- **Bold title** for each pain point\n"
            "- Description of the business impact"
        )
    else:
        instruction = (
            "Write 4-8 numbered steps. Each step:\n"
            "- **Bold title**\n"
            "- Description: what the person manually does\n"
            "- Tools Used: applications used (ONLY from transcript)"
        )

    prompt = f"""Document the CURRENT STATE ("As Is").

Project: "{project_name}". {entity_hint}

{instruction}

Then add 'Business Challenges' with 4-6 bullet points.

Use ONLY names from the transcript.

TRANSCRIPT:
{sample}

CURRENT STATE:"""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="AsIsProcess"
    )
    timed("As-Is", start)

    if response and len(response) > 100:
        return _sanitize_section_output(response)
    return (
        f"The current {project_name} process is performed manually. "
        f"Details to be documented during implementation."
    )


def get_to_be_process(
    transcript: str,
    project_name: str,
    entity_hint: str
) -> str:
    """Generate the future automated process (To Be state)."""
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)
    doc_type = config.document.document_type

    if doc_type == "BRD":
        instruction = (
            "Write 2-3 paragraphs describing the desired future state:\n"
            "- What business outcomes the solution must deliver\n"
            "- What capabilities are required"
        )
    else:
        instruction = (
            "Write 2-3 paragraphs describing how the automation handles "
            "this process end-to-end:\n"
            "- Write as if the automation already exists\n"
            "- Cover: trigger → connection → data handling → processing "
            "→ validation → action → reporting → logging"
        )

    prompt = f"""Write the "To Be" / future state description.

Project: "{project_name}". {entity_hint}

{instruction}

Use ONLY names from the transcript.

TRANSCRIPT:
{sample}

FUTURE STATE:"""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="ToBeProcess"
    )
    timed("To-Be", start)

    if response and len(response) > 100:
        return _sanitize_section_output(response)
    return (
        f"The {project_name} will use an automation solution to "
        f"handle the end-to-end process."
    )


# ============================================================
# Video Pipeline Sections (from step summaries) - Parallel
# ============================================================

def _generate_purpose_video(
    project_name: str,
    app_name: str,
    step_summaries: List[str]
) -> str:
    """Generate purpose section from video step summaries."""
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

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="DocumentPurpose_Video"
    )
    return _sanitize_section_output(response) if response else ""


def _generate_overview_video(
    project_name: str,
    app_name: str,
    step_summaries: List[str]
) -> Dict[str, str]:
    """Generate overview and justification from video step summaries."""
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

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="OverviewJustification_Video"
    )

    result = {"overview": "", "justification": ""}
    if response:
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

    return result


def _generate_as_is_video(
    project_name: str,
    app_name: str,
    step_summaries: List[str]
) -> str:
    """Generate as-is section from video step summaries."""
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

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="AsIsProcess_Video"
    )
    return _sanitize_section_output(response) if response else ""


def _generate_to_be_video(
    project_name: str,
    app_name: str,
    step_summaries: List[str]
) -> str:
    """Generate to-be section from video step summaries."""
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

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="ToBeProcess_Video"
    )
    return _sanitize_section_output(response) if response else ""


def _generate_prerequisites_video(
    project_name: str,
    app_name: str,
    vision_descriptions: List[str]
) -> List[Dict]:
    """Generate prerequisites/inputs from video analysis."""
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

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="Prerequisites_Video"
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

    return inputs if inputs else [
        {"parameter": "Application URL", "description": f"URL for {app_name or 'the application'}"},
        {"parameter": "User Credentials", "description": "Username and password for authentication"},
    ]


def _generate_exceptions_video(
    project_name: str,
    app_name: str,
    step_descriptions: List[str]
) -> List[Dict]:
    """Generate exception handling from video analysis."""
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

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="ExceptionHandling_Video"
    )

    exceptions = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|', 1)
                exc = re.sub(r'^\d+[\.\)]\s*', '', parts[0]).strip()
                handling = parts[1].strip() if len(parts) > 1 else ""
                if exc and len(exc) > 5:
                    exceptions.append({"exception": exc, "handling": handling})

    return exceptions if exceptions else [
        {"exception": "Application Login Failure",
         "handling": "Retry login up to 3 times. If still failing, stop and notify."},
        {"exception": "Element Not Found",
         "handling": "Wait up to 30 seconds, retry. If not found, log error and skip."},
    ]


def _generate_interfaces_video(
    app_name: str,
    vision_descriptions: List[str]
) -> List[Dict]:
    """Generate interface requirements from video analysis."""
    desc_sample = "\n".join(safe_sample(d, 80) for d in vision_descriptions[:6])

    prompt = f"""List the INTERFACE REQUIREMENTS (applications/systems) for this automation.

Application: "{app_name}"

Screens observed:
{desc_sample}

For each interface use this format:
Application Name | Purpose

List 3-6 interfaces. Output only the list."""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=PDD_SYSTEM_PROMPT,
        temperature=0.3,
        call_name="InterfaceReqs_Video"
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
        {"application": app_name or "Target Application",
         "purpose": "Primary application for process execution"},
    ]


def generate_all_sections_parallel(
    project_name: str,
    app_name: str,
    step_descriptions: List[str],
    vision_descriptions: List[str]
) -> Dict[str, Any]:
    """
    Generate all PDD sections in parallel using thread pool.
    Used by the video pipeline for faster generation.
    """
    start = time.time()
    results = {}

    tasks = {
        "purpose": lambda: _generate_purpose_video(project_name, app_name, step_descriptions),
        "overview_justification": lambda: _generate_overview_video(project_name, app_name, step_descriptions),
        "as_is": lambda: _generate_as_is_video(project_name, app_name, step_descriptions),
        "to_be": lambda: _generate_to_be_video(project_name, app_name, step_descriptions),
        "prerequisites": lambda: _generate_prerequisites_video(project_name, app_name, vision_descriptions),
        "exceptions": lambda: _generate_exceptions_video(project_name, app_name, step_descriptions),
        "interfaces": lambda: _generate_interfaces_video(app_name, vision_descriptions),
    }

    workers = min(config.llm.max_workers, len(tasks))
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