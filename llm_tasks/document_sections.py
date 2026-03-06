# llm_tasks/document_sections.py

"""
Document section generation — used ONLY by the video pipeline now.
Audio pipeline uses the consolidated meeting_compact.py call.
Includes parallel generation for video pipeline sections.
"""

import re
import time
import concurrent.futures
from typing import Dict, List, Any, Optional

from core.gemini_client import gemini_client
from core.config import config
from core.utils import timed, safe_sample, enforce_tone, redact_pii_text
from llm_tasks.system_prompts import get_system_prompt, PDD_SYSTEM_PROMPT, TONE_RULES


def _sanitize_section_output(text: str) -> str:
    """Remove instruction echoes and apply tone + redaction."""
    if not text:
        return ""
    patterns = [
        r'^Write\s+\d+-\d+\s+.*?(?=\n|$)',
        r'^Do\s+NOT\s+.*?(?=\n|$)',
        r'^INSTRUCTIONS?:.*?(?=\n\n|$)',
        r'^OUTPUT:?\s*',
        r'^SECTION\s*\d+[:\s]*',
        r'^Sure[,!.]?\s*',
        r'^Certainly[,!.]?\s*',
        r'^Here\s+(?:is|are)\s+.*?(?=\n\n|$)',
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = enforce_tone(cleaned)
    cleaned = redact_pii_text(cleaned)
    return cleaned.strip()


# ============================================================
# Video Pipeline Sections (from step summaries) — Parallel
# ============================================================

_VIDEO_SECTION_PROMPT_BASE = f"""You are a senior Business Analyst creating a Process Definition Document.

{TONE_RULES}

CRITICAL: Output ONLY the requested section content. No headers, no instructions, no meta-commentary."""


def _format_clarification_context(clarification_qa: Optional[Dict[str, str]]) -> str:
    """Format user-provided Q/A clarifications for section prompts."""
    if not clarification_qa:
        return ""

    lines = []
    for i, (question, answer) in enumerate(clarification_qa.items(), start=1):
        q = redact_pii_text((question or "").strip())
        a = redact_pii_text((answer or "").strip())
        if q and a:
            lines.append(f"{i}. Q: {q}\n   A: {a}")

    if not lines:
        return ""

    return (
        "Human Clarifications (authoritative context):\n"
        + "\n".join(lines)
        + "\nUse this context whenever relevant to remove ambiguity.\n"
    )


def generate_section_clarification_questions(
    project_name: str,
    app_name: str,
    step_descriptions: List[str],
    vision_descriptions: List[str],
    max_questions: int = 5
) -> List[str]:
    """Generate clarifying questions before section generation."""
    steps_text = "\n".join(f"- {s[:120]}" for s in step_descriptions[:15])
    vision_text = "\n".join(f"- {safe_sample(v, 120)}" for v in vision_descriptions[:8])

    prompt = f"""You are preparing to draft a high-quality Process Definition Document.

Project: "{project_name}"
Application: "{app_name}"

Observed process steps:
{steps_text}

Observed screen/context notes:
{vision_text}

TASK:
Ask the minimum clarifying questions needed to improve document quality.

RULES:
- Ask at most {max_questions} questions.
- Ask only if the answer materially improves Purpose, As-Is, To-Be, requirements, exceptions, or interfaces.
- Keep each question specific and answerable in 1-3 sentences.
- Output only questions, one per line.
- Do not include numbering, headings, or explanations."""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=_VIDEO_SECTION_PROMPT_BASE,
        temperature=0.2,
        call_name="SectionClarificationQuestions_Video"
    )

    fallback_questions = [
        "What is the exact trigger that starts this automation?",
        "What inputs must be available before execution begins?",
        "What should the system do when validation fails or data is missing?",
    ]

    if not response:
        return fallback_questions[:max_questions]

    questions = []
    for line in response.split("\n"):
        q = re.sub(r"^\s*(?:\d+[\.)]\s*|[-*]\s*)", "", line).strip()
        q = re.sub(r'^"|"$', "", q).strip()
        if not q or len(q) <= 8:
            continue
        if "?" not in q:
            q = q.rstrip(".:") + "?"
        questions.append(q)

    deduped = []
    seen = set()
    for q in questions:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)

    if not deduped:
        return fallback_questions[:max_questions]

    return deduped[:max_questions]


def _generate_purpose_video(
    project_name: str,
    app_name: str,
    step_summaries: List[str],
    clarification_context: str = ""
) -> str:
    """Generate purpose section from video step summaries."""
    steps_text = "\n".join(f"- {s[:100]}" for s in step_summaries[:15])

    prompt = f"""Write the "Purpose of this Document" section for a Process Definition Document.

Project: "{project_name}"
Application: "{app_name}"

Key process steps observed:
{steps_text}

{clarification_context}

REQUIREMENTS:
- Write 2-3 substantive paragraphs (150-250 words total).
- Paragraph 1: Define what this document covers — objectives, scope, and requirements for the automation.
- Paragraph 2: Identify the intended audience — developers, QA engineers, business stakeholders.
- Paragraph 3: Describe the scope — which processes, applications, and data flows the document addresses.
- Write in third person, present tense, active voice.
- NEVER mention screenshots, recordings, or video analysis.

Output ONLY the section content."""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=_VIDEO_SECTION_PROMPT_BASE,
        temperature=0.3,
        call_name="DocumentPurpose_Video"
    )
    return _sanitize_section_output(response) if response else ""


def _generate_overview_video(
    project_name: str,
    app_name: str,
    step_summaries: List[str],
    clarification_context: str = ""
) -> Dict[str, str]:
    """Generate overview and justification from video step summaries."""
    steps_text = "\n".join(f"- {s[:80]}" for s in step_summaries[:12])

    prompt = f"""Write two sections for a Process Definition Document.

Project: "{project_name}"
Application: "{app_name}"

Process steps observed:
{steps_text}

{clarification_context}

=== OVERVIEW ===
Write 1 opening paragraph stating the primary business objective of this automation.
Then list 5-7 bullet points (using •) describing specific outcomes the automation achieves.
Each bullet must start with "The system..." and describe a concrete capability.

=== JUSTIFICATION ===
Write 1 opening paragraph about the operational value of automating this process.
Then list 5-7 numbered items, each with a **bold title** and a description explaining the business benefit.
Example: "1. **Reduced Processing Time** — The system completes the end-to-end process in minutes compared to hours of manual effort."

Output both sections with the === headers. Third person, present tense, active voice only."""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=_VIDEO_SECTION_PROMPT_BASE,
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
    step_summaries: List[str],
    clarification_context: str = ""
) -> str:
    """Generate as-is section from video step summaries."""
    steps_text = "\n".join(f"- {s[:80]}" for s in step_summaries[:10])

    prompt = f"""Write the "As Is" (current manual process) section for a Process Definition Document.

Project: "{project_name}"
Application: "{app_name}"

The automated process includes these steps:
{steps_text}

{clarification_context}

YOUR TASK:
Describe how this process is CURRENTLY performed MANUALLY before automation.

REQUIREMENTS:
- Write 3-4 paragraphs (200-300 words total).
- Paragraph 1: Describe the manual process flow — what steps a human operator performs.
- Paragraph 2: Identify the tools, applications, and data sources used manually.
- Paragraph 3: Describe pain points — time consumption, error rates, inconsistencies.
- End with "Business Challenges:" followed by 4-6 bullet points listing specific problems.
- Third person, present tense, active voice.

Output ONLY the section content."""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=_VIDEO_SECTION_PROMPT_BASE,
        temperature=0.3,
        call_name="AsIsProcess_Video"
    )
    return _sanitize_section_output(response) if response else ""


def _generate_to_be_video(
    project_name: str,
    app_name: str,
    step_summaries: List[str],
    clarification_context: str = ""
) -> str:
    """Generate to-be section from video step summaries."""
    steps_text = "\n".join(f"- {s[:80]}" for s in step_summaries[:10])

    prompt = f"""Write the "To Be" (automated process) section for a Process Definition Document.

Project: "{project_name}"
Application: "{app_name}"

Automated steps:
{steps_text}

{clarification_context}

YOUR TASK:
Describe the AUTOMATED process as if it is already operational.

REQUIREMENTS:
- Write 3-4 paragraphs (200-300 words total).
- Paragraph 1: Describe the automation trigger and initialization sequence.
- Paragraph 2: Describe the core processing — data extraction, validation, transformation.
- Paragraph 3: Describe exception handling, reporting, and completion procedures.
- Write as present-tense facts: "The system connects...", "The automation validates..."
- NEVER use future tense ("will", "would", "should").
- Third person, present tense, active voice.

Output ONLY the section content."""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=_VIDEO_SECTION_PROMPT_BASE,
        temperature=0.3,
        call_name="ToBeProcess_Video"
    )
    return _sanitize_section_output(response) if response else ""


def _generate_prerequisites_video(
    project_name: str,
    app_name: str,
    vision_descriptions: List[str],
    clarification_context: str = ""
) -> List[Dict]:
    """Generate prerequisites/inputs from video analysis."""
    desc_sample = "\n".join(safe_sample(d, 100) for d in vision_descriptions[:8])

    prompt = f"""List the INPUT REQUIREMENTS for this automation.

Project: "{project_name}"
Application: "{app_name}"

Screens observed:
{desc_sample}

{clarification_context}

YOUR TASK:
List each input parameter the automation requires to execute.

FORMAT (one per line, use | separator):
Parameter Name | Description of what it is and why the automation needs it

Include: credentials, file paths, URLs, configuration values, data sources, thresholds.
List 5-10 inputs. Output ONLY the list, no headers or explanations.
NEVER include actual credential values, personal names, or email addresses."""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=_VIDEO_SECTION_PROMPT_BASE,
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
                    inputs.append({
                        "parameter": redact_pii_text(param),
                        "description": redact_pii_text(desc)
                    })

    return inputs if inputs else [
        {"parameter": "Application URL", "description": f"URL for {app_name or 'the application'}"},
        {"parameter": "User Credentials", "description": "Username and password for authentication"},
        {"parameter": "Input Data Source", "description": "Path or location of the source data file"},
    ]


def _generate_exceptions_video(
    project_name: str,
    app_name: str,
    step_descriptions: List[str],
    clarification_context: str = ""
) -> List[Dict]:
    """Generate exception handling from video analysis."""
    steps_sample = "\n".join(f"- {s[:80]}" for s in step_descriptions[:10])

    prompt = f"""List exception handling scenarios for this automation.

Project: "{project_name}"
Application: "{app_name}"

Process steps:
{steps_sample}

{clarification_context}

YOUR TASK:
For each potential failure scenario, describe the exception and the system's handling action.

FORMAT (one per line, use | separator):
Exception Scenario | Handling Action (what the system does)

Include: login failures, element not found, timeouts, data validation errors, application crashes, network errors.
List 6-10 exceptions. Output ONLY the list.
Write handling actions in third person: "The system retries...", "The system logs..."."""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=_VIDEO_SECTION_PROMPT_BASE,
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
         "handling": "The system retries login up to 3 times. If authentication still fails, the system stops execution and sends an error notification."},
        {"exception": "Element Not Found on Screen",
         "handling": "The system waits up to 30 seconds for the element. If not found, the system captures a screenshot, logs the error, and skips to the next step."},
        {"exception": "Data Validation Failure",
         "handling": "The system flags the invalid record, logs the validation error details, and continues processing remaining records."},
    ]


def _generate_interfaces_video(
    app_name: str,
    vision_descriptions: List[str],
    clarification_context: str = ""
) -> List[Dict]:
    """Generate interface requirements from video analysis."""
    desc_sample = "\n".join(safe_sample(d, 80) for d in vision_descriptions[:6])

    prompt = f"""List the INTERFACE REQUIREMENTS (applications and systems) for this automation.

Application: "{app_name}"

Screens observed:
{desc_sample}

{clarification_context}

YOUR TASK:
List each application or system the automation interacts with and its purpose.

FORMAT (one per line, use | separator):
Application/System Name | Purpose of interaction

List 3-6 interfaces. Output ONLY the list.
NEVER include personal names or credentials."""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=_VIDEO_SECTION_PROMPT_BASE,
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
    vision_descriptions: List[str],
    clarification_qa: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Generate all PDD sections in parallel using thread pool.
    Used by the video pipeline for faster generation.
    """
    start = time.time()
    results = {}
    clarification_context = _format_clarification_context(clarification_qa)

    tasks = {
        "purpose": lambda: _generate_purpose_video(project_name, app_name, step_descriptions, clarification_context),
        "overview_justification": lambda: _generate_overview_video(project_name, app_name, step_descriptions, clarification_context),
        "as_is": lambda: _generate_as_is_video(project_name, app_name, step_descriptions, clarification_context),
        "to_be": lambda: _generate_to_be_video(project_name, app_name, step_descriptions, clarification_context),
        "prerequisites": lambda: _generate_prerequisites_video(project_name, app_name, vision_descriptions, clarification_context),
        "exceptions": lambda: _generate_exceptions_video(project_name, app_name, step_descriptions, clarification_context),
        "interfaces": lambda: _generate_interfaces_video(app_name, vision_descriptions, clarification_context),
    }

    workers = min(config.llm.max_workers, len(tasks))
    print(f"    [Sections] Generating {len(tasks)} sections ({workers} parallel workers)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
                print(f"    [Sections] {name}")
            except Exception as e:
                print(f"    [Sections] {name}: {e}")
                results[name] = None

    timed(f"All sections ({len(results)})", start)
    return results