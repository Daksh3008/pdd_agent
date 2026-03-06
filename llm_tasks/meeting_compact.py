# llm_tasks/meeting_compact.py

"""
Compact meeting transcript → PDD bundle extraction.

TWO consolidated LLM calls:
  Call 1: Document sections (purpose, overview, justification, as-is, to-be)
  Call 2: Process data (steps, detailed steps, inputs, interfaces, exceptions)

This avoids the single-giant-JSON problem where Gemini truncates or
malforms the response, causing silent parse failures.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from core.gemini_client import gemini_client
from core.config import config
from core.utils import safe_sample, timed, enforce_tone, redact_pii_text
from llm_tasks.system_prompts import get_system_prompt, TONE_RULES


# ============================================================
# JSON extraction helpers
# ============================================================

def _extract_json_object(text: str) -> Optional[str]:
    """Extract the first JSON object from a response."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _coerce_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        lines = [l.strip("•*- \t") for l in x.splitlines() if l.strip()]
        return [l for l in lines if len(l) > 5]
    return [str(x).strip()] if str(x).strip() else []


def _coerce_list_dict(x: Any, keys: Tuple[str, str]) -> List[Dict[str, str]]:
    k1, k2 = keys
    out: List[Dict[str, str]] = []
    if isinstance(x, list):
        for row in x:
            if isinstance(row, dict):
                v1 = str(row.get(k1, "")).strip()
                v2 = str(row.get(k2, "")).strip()
                if v1:
                    out.append({k1: v1, k2: v2})
            elif isinstance(row, str) and "|" in row:
                a, b = row.split("|", 1)
                a, b = a.strip(), b.strip()
                if a:
                    out.append({k1: a, k2: b})
    elif isinstance(x, str):
        for line in x.splitlines():
            if "|" in line:
                a, b = line.split("|", 1)
                a, b = a.strip(), b.strip()
                if a:
                    out.append({k1: a, k2: b})
    return out


def _apply_tone_and_redaction(text: str) -> str:
    """Apply tone enforcement and PII redaction."""
    if not text:
        return text
    text = enforce_tone(text)
    text = redact_pii_text(text)
    # Remove markdown bold markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    return text


def _strip_markdown(text: str) -> str:
    """Remove all markdown formatting from text."""
    if not text:
        return text
    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    # Italic
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Bullet normalization
    text = re.sub(r'^[-*]\s+', '- ', text, flags=re.MULTILINE)
    return text


def _format_clarification_context(clarification_qa: Optional[Dict[str, str]]) -> str:
    """Format human-provided clarifications for transcript prompts."""
    if not clarification_qa:
        return ""

    lines: List[str] = []
    for i, (q, a) in enumerate(clarification_qa.items(), start=1):
        q_clean = redact_pii_text((q or "").strip())
        a_clean = redact_pii_text((a or "").strip())
        if q_clean and a_clean:
            lines.append(f"{i}. Q: {q_clean}\n   A: {a_clean}")

    if not lines:
        return ""

    return (
        "Human Clarifications (authoritative context):\n"
        + "\n".join(lines)
        + "\nUse this context whenever relevant to remove ambiguity.\n"
    )


def generate_clarification_questions_from_transcript(
    transcript: str,
    project_name_hint: Optional[str] = None,
    max_questions: int = 5,
) -> List[str]:
    """Generate clarifying questions before consolidated transcript extraction."""
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)
    project_hint = project_name_hint or "(not provided)"

    prompt = f"""You are preparing to generate a high-quality Process Definition Document from a meeting transcript.

Project name hint: "{project_hint}"

TASK:
Ask only the minimum clarifying questions that would materially improve document quality.

RULES:
- Ask at most {max_questions} questions.
- Questions must help improve: purpose, overview, as-is, to-be, process steps, or requirements.
- Keep each question specific and answerable in 1-3 sentences.
- Output only questions, one per line.
- Do not include numbering, headings, or explanations.

TRANSCRIPT:
{sample}
"""

    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        temperature=0.2,
        max_output_tokens=1200,
        call_name="DocBundle_ClarificationQuestions",
        max_retries=3,
    )

    fallback_questions = [
        "What is the primary business outcome this automation must deliver?",
        "Which input sources and output destinations are in scope for the process?",
        "What are the top exception scenarios and expected handling actions?",
    ]

    if not resp:
        return fallback_questions[:max_questions]

    questions: List[str] = []
    for line in resp.splitlines():
        q = re.sub(r"^\s*(?:\d+[\.)]\s*|[-*]\s*)", "", line).strip()
        q = re.sub(r'^"|"$', "", q).strip()
        if not q or len(q) <= 8:
            continue
        if "?" not in q:
            q = q.rstrip(".:") + "?"
        questions.append(q)

    deduped: List[str] = []
    seen = set()
    for q in questions:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)

    if not deduped:
        return fallback_questions[:max_questions]

    return deduped[:max_questions]


# ============================================================
# Call 1: Document Sections
# ============================================================

def _generate_document_sections(
    transcript: str,
    project_name_hint: Optional[str] = None,
    clarification_qa: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    LLM Call 1: Extract project name + all narrative document sections.
    Returns dict with project_name, entities, and document sections.
    """
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)
    doc_type = config.document.document_type
    doc_full = config.document.document_type_full

    project_hint = f'Project name hint: "{project_name_hint}"' if project_name_hint else \
        "Derive a short descriptive project name (max 6 words) from the main process discussed."
    clarification_context = _format_clarification_context(clarification_qa)

    prompt = f"""You are a senior Business Analyst creating a {doc_full} ({doc_type}).

{project_hint}

TASK: Produce the following sections as a JSON object.

SECTION REQUIREMENTS:

1. "project_name": Short descriptive name (max 6 words).

2. "entities": Object with arrays: "companies", "applications", "systems", "departments".
   ONLY names explicitly mentioned in the transcript. Use empty array if none found.

3. "purpose": 2-3 paragraphs (150-250 words).
   - What this document defines (objectives, scope, requirements).
   - Intended audience (developers, QA, business stakeholders).
   - Scope of the automation being documented.

4. "overview": 1 paragraph + 5-7 bullet points.
   - Opening paragraph stating the primary business objective.
   - Bullet points listing specific outcomes. Each starts with "The system".

5. "justification": 1 paragraph + 5-7 numbered benefits.
   - Opening paragraph about operational value.
   - Numbered items with a title and description for each benefit.
   DO NOT use markdown bold (no ** markers). Use plain text only.

6. "as_is": 3-4 paragraphs (200-300 words).
   - Current manual process description.
   - Pain points, inefficiencies, risks.
   - End with "Business Challenges:" followed by 4-6 bullet points.

7. "to_be": 3-4 paragraphs (200-300 words).
   - Future automated process, written in present tense as if operational.
   - Cover trigger, data handling, validation, processing, reporting, logging.

CRITICAL RULES:
- Use ONLY names from the transcript. Never invent names.
- NEVER mention the meeting, transcript, recording, or speakers.
- NEVER include personal names, emails, or phone numbers.
- Write in THIRD PERSON, PRESENT TENSE, ACTIVE VOICE.
- DO NOT use markdown formatting (no **, no ##, no __).
- Output STRICT JSON only. No markdown fences, no commentary.

JSON FORMAT:
{{
  "project_name": "string",
  "entities": {{"companies": [], "applications": [], "systems": [], "departments": []}},
  "purpose": "string",
  "overview": "string",
  "justification": "string",
  "as_is": "string",
  "to_be": "string"
}}

TRANSCRIPT:
{sample}

{clarification_context}
"""

    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        temperature=0.3,
        max_output_tokens=config.llm.max_output_tokens,
        call_name="DocBundle_Sections",
        max_retries=3
    )

    result = {
        "project_name": project_name_hint or "Process Automation Project",
        "entities": {"companies": [], "applications": [], "systems": [], "departments": []},
        "document": {"purpose": "", "overview": "", "justification": "", "as_is": "", "to_be": ""}
    }

    json_text = _extract_json_object(resp or "")
    if json_text:
        try:
            data = json.loads(json_text)

            pn = str(data.get("project_name", "")).strip()
            if pn and len(pn) > 3:
                result["project_name"] = pn

            ent = data.get("entities", {})
            if isinstance(ent, dict):
                for key in ["companies", "applications", "systems", "departments"]:
                    items = ent.get(key, [])
                    if isinstance(items, list):
                        result["entities"][key] = [str(i).strip() for i in items if str(i).strip()]

            for key in ["purpose", "overview", "justification", "as_is", "to_be"]:
                val = str(data.get(key, "")).strip()
                if val:
                    val = _strip_markdown(val)
                    val = _apply_tone_and_redaction(val)
                    result["document"][key] = val

        except json.JSONDecodeError as e:
            print(f"    [DocBundle_Sections] JSON parse failed: {e}")
            # Try to extract sections from raw text
            result["document"] = _fallback_parse_sections(resp)
        except Exception as e:
            print(f"    [DocBundle_Sections] Error: {type(e).__name__}: {e}")
    else:
        print("    [DocBundle_Sections] No JSON found in response, using fallback parser")
        result["document"] = _fallback_parse_sections(resp)

    # Ensure all sections have content
    pn = result["project_name"]
    doc = result["document"]
    doc_full_name = config.document.document_type_full

    if not doc["purpose"] or len(doc["purpose"]) < 50:
        doc["purpose"] = (
            f"This {doc_full_name} defines the objectives, scope, and detailed "
            f"requirements for the {pn} automation initiative. The document serves "
            f"as the primary reference for development teams, quality assurance personnel, "
            f"and business stakeholders involved in the design, implementation, and "
            f"validation of the automated solution.\n\n"
            f"The scope encompasses the complete end-to-end process flow, including "
            f"system interactions, data handling procedures, validation rules, exception "
            f"handling scenarios, and expected outputs. The document captures both the "
            f"current manual state and the target automated state of the {pn} process.\n\n"
            f"This document applies to all phases of the automation lifecycle, from "
            f"initial development through testing, deployment, and ongoing maintenance."
        )

    if not doc["overview"] or len(doc["overview"]) < 50:
        doc["overview"] = (
            f"The primary objective of the {pn} automation is to streamline "
            f"and standardize the existing manual process, ensuring consistency, "
            f"accuracy, and compliance with established business rules.\n\n"
            f"- The system automates repetitive manual tasks to reduce processing time.\n"
            f"- The system enforces standardized validation rules across all records.\n"
            f"- The system generates comprehensive audit trails for compliance tracking.\n"
            f"- The system reduces human error through automated data handling.\n"
            f"- The system provides real-time status reporting and exception notifications."
        )

    if not doc["justification"] or len(doc["justification"]) < 50:
        doc["justification"] = (
            f"Automating the {pn} process delivers measurable operational value "
            f"by reducing manual effort, improving accuracy, and ensuring consistent "
            f"execution across all processing cycles.\n\n"
            f"1. Reduced Processing Time: The system completes the end-to-end process "
            f"in minutes compared to hours of manual effort.\n"
            f"2. Improved Accuracy: The system eliminates manual data entry errors "
            f"through automated validation and processing.\n"
            f"3. Enhanced Compliance: The system maintains detailed audit logs for "
            f"regulatory and governance requirements.\n"
            f"4. Consistent Execution: The system applies identical business rules "
            f"to every record, eliminating subjective variation.\n"
            f"5. Scalability: The system handles increased volume without additional "
            f"manual resources."
        )

    if not doc["as_is"] or len(doc["as_is"]) < 50:
        doc["as_is"] = (
            f"The current {pn} process relies on manual execution by trained operators. "
            f"The operator logs into the required applications, navigates to the relevant "
            f"modules, and performs data extraction, validation, and processing tasks "
            f"by hand.\n\n"
            f"The manual process requires significant time investment for each processing "
            f"cycle. The operator must review individual records, cross-reference data "
            f"across multiple sources, and manually update system records based on the "
            f"validation results.\n\n"
            f"Business Challenges:\n"
            f"- High processing time due to manual record-by-record handling.\n"
            f"- Risk of human error in data entry and validation.\n"
            f"- Inconsistent application of business rules across operators.\n"
            f"- Limited audit trail for compliance verification.\n"
            f"- Difficulty scaling with increased processing volume."
        )

    if not doc["to_be"] or len(doc["to_be"]) < 50:
        doc["to_be"] = (
            f"The {pn} automation executes the end-to-end process through a structured "
            f"sequence of system actions. The automation initiates upon a configured "
            f"trigger, establishes connections to required applications, and processes "
            f"data according to defined business rules.\n\n"
            f"The system validates each record against the established criteria, performs "
            f"the required actions for compliant records, and logs exceptions for "
            f"non-compliant entries. The automation handles error scenarios through "
            f"configured retry logic and exception routing.\n\n"
            f"Upon completion, the system generates a comprehensive execution report "
            f"detailing processed records, exceptions encountered, and overall "
            f"processing statistics. The system updates the process status and "
            f"archives execution logs for audit purposes."
        )

    return result


def _fallback_parse_sections(raw_text: str) -> Dict[str, str]:
    """Parse sections from non-JSON LLM response as fallback."""
    sections = {"purpose": "", "overview": "", "justification": "", "as_is": "", "to_be": ""}
    if not raw_text:
        return sections

    raw_text = _strip_markdown(raw_text)

    # Try to find labeled sections
    section_patterns = {
        "purpose": [r'(?:purpose|document purpose)[:\s]*(.*?)(?=overview|justification|as.is|to.be|\Z)'],
        "overview": [r'(?:overview|objective)[:\s]*(.*?)(?=justification|as.is|to.be|\Z)'],
        "justification": [r'(?:justification|business justification)[:\s]*(.*?)(?=as.is|to.be|\Z)'],
        "as_is": [r'(?:as.is|current state|current process)[:\s]*(.*?)(?=to.be|future|\Z)'],
        "to_be": [r'(?:to.be|future state|automated process)[:\s]*(.*?)$'],
    }

    for key, patterns in section_patterns.items():
        for pattern in patterns:
            m = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
            if m and len(m.group(1).strip()) > 30:
                sections[key] = _apply_tone_and_redaction(m.group(1).strip())
                break

    return sections


# ============================================================
# Call 2: Process Data (steps, requirements)
# ============================================================

def _generate_process_data(
    transcript: str,
    project_name: str,
    entities: Dict,
    clarification_qa: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    LLM Call 2: Extract process steps, detailed steps, and all requirements tables.
    """
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)

    apps_hint = ""
    if entities.get("applications"):
        apps_hint = f"Applications mentioned: {', '.join(entities['applications'])}"
    clarification_context = _format_clarification_context(clarification_qa)

    prompt = f"""You are a senior Business Analyst extracting automation process data.

Project: "{project_name}"
{apps_hint}

TASK: Extract ALL process steps and requirements from this transcript.

OUTPUT STRICT JSON with this exact structure:

{{
  "process_steps": [
    "Step description starting with a verb (8-15 items)"
  ],
  "detailed_steps": [
    "Detailed screen-level action description (15-25 items)"
  ],
  "input_requirements": [
    {{"parameter": "Parameter Name", "description": "What it is and why needed"}}
  ],
  "interface_requirements": [
    {{"application": "App Name", "purpose": "Why the automation uses it"}}
  ],
  "exception_handling": [
    {{"exception": "Error scenario", "handling": "What the system does"}}
  ]
}}

DETAILED INSTRUCTIONS:

process_steps (8-15 items):
- High-level automation steps, each starting with an action verb.
- Examples: "Connect to the application using credentials",
  "Navigate to the data module", "Extract records from source",
  "Validate records against criteria", "Generate execution report"

detailed_steps (15-25 items):
- Screen-level actions for PDD Section 2.4.
- Each describes ONE specific action: login, navigate, click, enter data,
  download, process, validate, export, report, logout.
- Each starts with "The system" or "The automation".
- Examples: "The system opens the target application login page",
  "The system enters the configured credentials and clicks Sign In",
  "The system navigates to the Reports tab in the main menu"

input_requirements (5-10 items):
- Parameters needed: credentials, URLs, file paths, config values.
- NEVER include actual passwords or personal data.

interface_requirements (3-8 items):
- Applications and systems the automation interacts with.
- ONLY names from the transcript.

exception_handling (6-10 items):
- Error scenarios with system response.
- Include: login failure, timeout, missing data, validation error, system crash.
- Handling must be in third person: "The system retries...", "The system logs..."

CRITICAL RULES:
- Use ONLY application names from the transcript.
- NEVER mention the meeting, transcript, or speakers.
- NEVER include personal names, emails, or phone numbers.
- Third person, present tense, active voice.
- DO NOT use markdown formatting.
- Output STRICT JSON only.

TRANSCRIPT:
{sample}

{clarification_context}
"""

    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        temperature=0.3,
        max_output_tokens=config.llm.max_output_tokens,
        call_name="DocBundle_ProcessData",
        max_retries=3
    )

    result = {
        "process_steps": [],
        "detailed_steps": [],
        "input_requirements": [],
        "interface_requirements": [],
        "exception_handling": [],
    }

    json_text = _extract_json_object(resp or "")
    if json_text:
        try:
            data = json.loads(json_text)

            result["process_steps"] = _coerce_list_str(data.get("process_steps"))
            result["detailed_steps"] = _coerce_list_str(data.get("detailed_steps"))
            result["input_requirements"] = _coerce_list_dict(
                data.get("input_requirements"), ("parameter", "description")
            )
            result["interface_requirements"] = _coerce_list_dict(
                data.get("interface_requirements"), ("application", "purpose")
            )
            result["exception_handling"] = _coerce_list_dict(
                data.get("exception_handling"), ("exception", "handling")
            )

        except json.JSONDecodeError as e:
            print(f"    [DocBundle_ProcessData] JSON parse failed: {e}")
            result = _fallback_parse_process_data(resp)
        except Exception as e:
            print(f"    [DocBundle_ProcessData] Error: {type(e).__name__}: {e}")
    else:
        print("    [DocBundle_ProcessData] No JSON found, using fallback parser")
        result = _fallback_parse_process_data(resp)

    # Redact PII from all text
    result["process_steps"] = [redact_pii_text(s) for s in result["process_steps"]]
    result["detailed_steps"] = [redact_pii_text(s) for s in result["detailed_steps"]]

    # Apply fallbacks for empty sections
    if not result["process_steps"]:
        apps = ', '.join(entities.get('applications', [])) or 'the target application'
        result["process_steps"] = [
            f"The system connects to {apps} using authorized credentials.",
            f"The system navigates to the relevant processing module within {apps}.",
            "The system extracts the required data from the configured source.",
            "The system filters records based on the defined business criteria.",
            "The system validates each record against the established rules.",
            f"The system performs the required processing actions in {apps}.",
            "The system captures the updated status for each processed item.",
            "The system generates a comprehensive execution report.",
            "The system updates the process status and logs execution details.",
        ]
        print("    [DocBundle_ProcessData] Using fallback process steps")

    if not result["detailed_steps"]:
        result["detailed_steps"] = [
            "The system opens the target application and navigates to the login page.",
            "The system enters the configured credentials and authenticates.",
            "The system navigates to the main processing module.",
            "The system selects the appropriate data source or report.",
            "The system applies the configured filters and criteria.",
            "The system extracts the matching records from the source.",
            "The system validates each record against the defined business rules.",
            "The system flags records that fail validation checks.",
            "The system processes each validated record according to the workflow.",
            "The system updates the record status after processing.",
            "The system captures processing results and timestamps.",
            "The system generates a summary report of all processed records.",
            "The system exports the report to the configured output location.",
            "The system logs all execution details for audit purposes.",
            "The system closes the application and terminates the session.",
        ]
        print("    [DocBundle_ProcessData] Using fallback detailed steps")

    if not result["input_requirements"]:
        result["input_requirements"] = [
            {"parameter": "Application URL", "description": "Web address of the target application portal."},
            {"parameter": "User Credentials", "description": "Username and password for secure application access."},
            {"parameter": "Input Data Source", "description": "File path or database connection for source data."},
            {"parameter": "Processing Criteria", "description": "Business rules and filters for record selection."},
            {"parameter": "Output Location", "description": "File path or destination for generated reports."},
        ]

    if not result["interface_requirements"]:
        result["interface_requirements"] = [
            {"application": "Target Application", "purpose": "Primary application for process execution."},
        ]

    if not result["exception_handling"]:
        result["exception_handling"] = [
            {"exception": "Application Login Failure",
             "handling": "The system retries login up to 3 times. If authentication fails, the system stops and sends a notification."},
            {"exception": "Element Not Found",
             "handling": "The system waits up to 30 seconds. If not found, the system captures a screenshot and logs the error."},
            {"exception": "Data Validation Error",
             "handling": "The system flags the invalid record, logs details, and continues processing remaining items."},
            {"exception": "Application Timeout",
             "handling": "The system retries the operation after a configured wait period and logs the timeout event."},
            {"exception": "System Exception",
             "handling": "The system captures error details, saves a diagnostic screenshot, and terminates gracefully."},
        ]

    return result


def _fallback_parse_process_data(raw_text: str) -> Dict[str, Any]:
    """Parse process data from non-JSON response."""
    result = {
        "process_steps": [],
        "detailed_steps": [],
        "input_requirements": [],
        "interface_requirements": [],
        "exception_handling": [],
    }
    if not raw_text:
        return result

    raw_text = _strip_markdown(raw_text)

    # Try to extract numbered lists
    lines = raw_text.split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        lower = line.lower()

        # Detect section headers
        if 'process_step' in lower or 'process step' in lower:
            current_section = 'process_steps'
            continue
        elif 'detailed_step' in lower or 'detailed step' in lower:
            current_section = 'detailed_steps'
            continue
        elif 'input_req' in lower or 'input req' in lower:
            current_section = 'input_requirements'
            continue
        elif 'interface_req' in lower or 'interface req' in lower:
            current_section = 'interface_requirements'
            continue
        elif 'exception' in lower:
            current_section = 'exception_handling'
            continue

        if not line or not current_section:
            continue

        # Parse numbered/bulleted items
        cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line).strip()
        cleaned = re.sub(r'^[-•*]\s*', '', cleaned).strip()
        cleaned = cleaned.strip('"')

        if len(cleaned) < 10:
            continue

        if current_section in ('process_steps', 'detailed_steps'):
            result[current_section].append(cleaned)
        elif current_section == 'input_requirements' and '|' in cleaned:
            parts = cleaned.split('|', 1)
            result[current_section].append({
                "parameter": parts[0].strip(),
                "description": parts[1].strip() if len(parts) > 1 else ""
            })
        elif current_section == 'interface_requirements' and '|' in cleaned:
            parts = cleaned.split('|', 1)
            result[current_section].append({
                "application": parts[0].strip(),
                "purpose": parts[1].strip() if len(parts) > 1 else ""
            })
        elif current_section == 'exception_handling' and '|' in cleaned:
            parts = cleaned.split('|', 1)
            result[current_section].append({
                "exception": parts[0].strip(),
                "handling": parts[1].strip() if len(parts) > 1 else ""
            })

    return result


# ============================================================
# Public API
# ============================================================

def generate_doc_bundle_from_transcript(
    transcript: str,
    project_name_hint: Optional[str] = None,
    clarification_qa: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Two consolidated LLM calls to extract ALL PDD content.
    Call 1: Document narrative sections
    Call 2: Process steps and requirements tables
    """
    start = time.time()

    # Call 1: Sections
    print("    [DocBundle] Call 1/2: Document sections...")
    sections_result = _generate_document_sections(
        transcript,
        project_name_hint,
        clarification_qa=clarification_qa,
    )

    project_name = sections_result["project_name"]
    entities = sections_result["entities"]

    # Call 2: Process data
    print("    [DocBundle] Call 2/2: Process steps & requirements...")
    process_result = _generate_process_data(
        transcript,
        project_name,
        entities,
        clarification_qa=clarification_qa,
    )

    # Combine
    result = {
        "project_name": project_name,
        "entities": entities,
        "document": sections_result["document"],
        "process": {
            "process_steps": process_result["process_steps"],
            "detailed_steps": process_result["detailed_steps"],
        },
        "requirements": {
            "input_requirements": process_result["input_requirements"],
            "interface_requirements": process_result["interface_requirements"],
            "exception_handling": process_result["exception_handling"],
        }
    }

    timed("DocBundle_Combined", start)
    return result


def generate_dot_from_transcript(
    transcript: str,
    project_name: str,
    process_steps: Optional[List[str]] = None
) -> str:
    """
    Single LLM call to generate DOT flowchart code.
    Enforces short labels (max 5 words per node).
    """
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)

    steps_block = ""
    if process_steps:
        steps_block = "PROCESS STEPS (use these as nodes):\n" + "\n".join(
            f"{i+1}. {s[:80]}" for i, s in enumerate(process_steps[:18])
        )

    max_words = config.flowchart.max_label_words

    prompt = f"""Generate a Graphviz DOT flowchart for this automation process.

Process: "{project_name}"

{steps_block}

STRICT FORMATTING RULES:
1. Output ONLY valid DOT code. No markdown, no explanation, no code fences.
2. Must start with: digraph ProcessFlow {{
3. Must include Start (oval, green) and End (oval, red) nodes.
4. Use rankdir=TB.
5. Every node label MUST be MAX {max_words} WORDS. Short verb+object phrases only.
   GOOD: "Login to Portal", "Validate Records", "Export Report"
   BAD: "The system validates each record against the defined criteria"
6. Use diamond shape for decisions, box for steps.
7. Colors: process=lightblue, decision=gold, start=lightgreen, end=lightcoral.
8. All nodes must have style=filled.

TRANSCRIPT (context only):
{sample[:3000]}
"""

    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        temperature=0.2,
        call_name="DOT_FromTranscript",
        max_retries=3
    )

    dot = (resp or "").strip()
    dot = re.sub(r'^```(?:dot|graphviz)?\s*', '', dot)
    dot = re.sub(r'\s*```$', '', dot)
    dot = dot.strip()

    m = re.search(r"(digraph\s+\w*\s*\{.*\})", dot, re.DOTALL)
    if m:
        dot = m.group(1).strip()

    dot = _enforce_short_labels(dot, max_words)

    timed("DOT_FromTranscript", start)
    return dot


def _enforce_short_labels(dot_code: str, max_words: int = 5) -> str:
    """Post-process DOT code to enforce short labels."""
    if not dot_code:
        return dot_code

    def _shorten(match):
        prefix = match.group(1)
        label = match.group(2)
        suffix = match.group(3)

        label = re.sub(
            r'\b(the|a|an|to|of|for|in|on|at|by|with|and|is|are|was|were|been|being)\b',
            ' ', label, flags=re.IGNORECASE
        )
        label = re.sub(r'\s+', ' ', label).strip()

        words = label.split()
        if len(words) > max_words:
            label = ' '.join(words[:max_words])

        if label:
            label = label[0].upper() + label[1:]

        return f'{prefix}"{label}"{suffix}'

    result = re.sub(
        r'(label\s*=\s*)"([^"]+)"(\s*[,\]])',
        _shorten, dot_code
    )
    return result