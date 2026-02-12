# src/llm_tasks.py

"""
LLM-based tasks for PDD/BRD generation.
Universal prompts that work for ANY business process meeting.

PROMPT DESIGN:
1. System prompt establishes ROLE and THINKING MODE
2. Each prompt uses ROLE → CONTEXT → TASK → FORMAT pattern
3. All examples use generic placeholders [Application], [System], etc.
4. No product names, technology names, or specific architectures in examples
5. Technology-neutral: works for RPA, scripts, APIs, workflows, etc.

ANTI-HALLUCINATION:
- Every prompt reminds the LLM to use ONLY transcript content
- Entity verification removes names not found in transcript
- No product names in examples that could leak into output
- Prompt sizes are kept within safe limits for the model
"""

import re
import time
from typing import Optional, Dict, List, Tuple
from llm_client import llm_client
from config import llm_params, doc_config, ACTION_KEYWORDS


MAX_SAMPLE = llm_params.max_sample_text
MAX_SAMPLE_SMALL = llm_params.max_sample_small
MAX_SAMPLE_ENTITY = llm_params.max_sample_entity
CHUNK_SIZE = llm_params.chunk_size
MAX_CHUNKS = llm_params.max_chunks
OVERLAP_SIZE = llm_params.overlap_size


# ============================================================
# UNIVERSAL SYSTEM PROMPT
# Technology-neutral, anti-hallucination
# ============================================================

SYSTEM_PROMPT = """You are a senior Business Analyst who creates Process Definition Documents and Business Requirement Documents for automation projects.

You receive transcripts of meetings where business teams discuss a process they want to automate. Your job is to understand the UNDERLYING BUSINESS PROCESS and document it professionally.

RULES:

1. SEPARATE CONVERSATION FROM PROCESS
   Extract only PROCESS ACTIONS. Ignore conversation, greetings, scheduling, and small talk.

2. TRANSLATE HUMAN ACTIONS TO SYSTEM ACTIONS
   Convert manual human descriptions into professional automation language.
   Human says: "I open the file and check each row"
   You write: "The system opens the source data file and iterates through each record for validation."

3. RECONSTRUCT THE LOGICAL SEQUENCE
   Meetings are messy. Reconstruct the complete end-to-end process in logical order.

4. USE PROFESSIONAL LANGUAGE
   Third person. Automation terminology. No first person.
   "The system establishes a connection..."
   "Validates each record against..."
   "Updates the execution status..."

5. ANTI-HALLUCINATION — MANDATORY:
   - Use ONLY names, applications, and systems EXPLICITLY mentioned in the provided text
   - NEVER invent or substitute names from your training data
   - If a name is unclear, use generic terms: "the application", "the portal", "the system"
   - NEVER mention the meeting, transcript, or recording
   - NEVER write "they discussed" or "the team agreed"
   - NEVER use first person"""


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _timed(name: str, start: float):
    """Log elapsed time for a task."""
    print(f"    [{name}] done in {time.time() - start:.1f}s")


def _safe_sample(transcript: str, max_len: int = None) -> str:
    """Take a safe-sized sample. Beginning + End."""
    max_len = max_len or MAX_SAMPLE
    if len(transcript) <= max_len:
        return transcript
    first = int(max_len * 0.6)
    last = max_len - first - 30
    return transcript[:first] + "\n[...]\n" + transcript[-last:]


def split_into_chunks(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            bp = text.rfind('. ', start + CHUNK_SIZE - 300, end)
            if bp != -1:
                end = bp + 1
        chunks.append(text[start:end].strip())
        start = end - OVERLAP_SIZE
        if start <= 0 and chunks:
            break
    return chunks


def _build_entity_hint(entities: Dict) -> str:
    """Build a hint string from extracted entities."""
    parts = []
    if entities.get("companies"):
        parts.append(f"Companies: {', '.join(entities['companies'])}")
    if entities.get("applications"):
        parts.append(f"Applications: {', '.join(entities['applications'])}")
    if entities.get("systems"):
        parts.append(f"Systems: {', '.join(entities['systems'])}")
    if parts:
        return "Entities from transcript: " + "; ".join(parts)
    return ""


def _verify_entities_against_transcript(entities: Dict, transcript: str) -> Dict:
    """
    Remove entity names that don't appear in the transcript.
    Catches LLM hallucinations.
    """
    transcript_lower = transcript.lower()

    def _appears(name: str) -> bool:
        name_lower = name.lower().strip()
        if name_lower in transcript_lower:
            return True
        # Multi-word: all significant words must appear
        words = name_lower.split()
        if len(words) > 1:
            significant = [w for w in words if len(w) > 3]
            if significant and all(w in transcript_lower for w in significant):
                return True
        # Prefix match (4+ chars)
        if len(name_lower) >= 4 and name_lower[:4] in transcript_lower:
            return True
        # No-space match
        no_space = name_lower.replace(" ", "")
        if len(no_space) >= 4 and no_space in transcript_lower.replace(" ", ""):
            return True
        return False

    verified = {}
    for key, items in entities.items():
        if isinstance(items, list):
            verified_items = []
            for item in items:
                if _appears(item):
                    verified_items.append(item)
                else:
                    print(f"    [Entities] ⚠ Removed hallucinated: '{item}'")
            verified[key] = verified_items
        else:
            verified[key] = items
    return verified


# ============================================================
# CALL 1: Entities + Project Name
# ============================================================

def extract_entities_and_project(transcript: str) -> Tuple[Dict[str, List[str]], str]:
    """Extract named entities and project name from transcript."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=MAX_SAMPLE_ENTITY)

    prompt = f"""Extract factual information from this meeting transcript.

RULES:
- ONLY extract names EXPLICITLY mentioned in the text
- Do NOT guess or correct names — write them exactly as they appear
- If no names are mentioned for a category, write "None"
- For the project name: if not stated, create a short descriptive name from the main process discussed

OUTPUT FORMAT:
COMPANIES: name1, name2
APPLICATIONS: name1, name2
SYSTEMS: name1, name2
DEPARTMENTS: name1, name2
PROJECT_NAME: Short Descriptive Name

TRANSCRIPT:
{sample}"""

    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)

    entities = {
        "companies": [], "applications": [], "systems": [],
        "departments": [], "processes": []
    }
    project_name = "Process Automation Project"

    if response:
        for line in response.split('\n'):
            line = line.strip()
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            key = key.strip().upper()
            items = [
                x.strip() for x in value.split(',')
                if x.strip() and x.strip().lower() not in
                ['none', 'n/a', '', 'not mentioned', 'none mentioned']
            ]
            if 'COMPAN' in key:
                entities["companies"] = items
            elif 'APPLIC' in key:
                entities["applications"] = items
            elif 'SYSTEM' in key:
                entities["systems"] = items
            elif 'DEPART' in key:
                entities["departments"] = items
            elif 'PROJECT' in key:
                name = value.strip().strip('"\'')
                if name and name.lower() not in ['none', 'n/a', 'not mentioned']:
                    project_name = ' '.join(name.split()[:7])

    entities = _verify_entities_against_transcript(entities, transcript)
    _timed("Entities+Project", start)
    return entities, project_name


# ============================================================
# CALL 2: Document Purpose
# ============================================================

def get_document_purpose_text(transcript: str, project_name: str, entity_hint: str) -> str:
    """Generate the Purpose of this Document section."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=MAX_SAMPLE_SMALL)

    prompt = f"""Write the "Purpose of this Document" section for a {doc_config.document_type_full}.

Project: "{project_name}". {entity_hint}

Write 1-2 paragraphs covering:
- What this document defines (objectives, scope, requirements)
- What process is being documented for automation
- That it covers current manual state and future automated state
- That it serves as the basis for designing and deploying the solution

Use ONLY names from the transcript. Formal business English. Third person.

TRANSCRIPT:
{sample}

PURPOSE:"""

    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
    _timed("Purpose", start)

    if response and len(response) > 50:
        return response.strip()

    return (
        f"This {doc_config.document_type_full} ({doc_config.document_type}) defines the "
        f"objectives, scope, and detailed business requirements for the {project_name} "
        f"initiative. It describes the current manual process and the future automated "
        f"state required to design, develop, and deploy the automation solution."
    )


# ============================================================
# CALL 3: Overview & Objective + Business Justification
# ============================================================

def get_overview_and_justification(
    transcript: str, project_name: str, entity_hint: str
) -> Dict[str, str]:
    """Generate Overview/Objective and Business Justification."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=MAX_SAMPLE)

    prompt = f"""Write two sections for a {doc_config.document_type_full}.

Project: "{project_name}". {entity_hint}

Use ONLY names from the transcript.

===OVERVIEW===
Write an "Overview and Objective" section:
- One paragraph stating the primary objective
- Then 4-6 bullet points of what the automation achieves
- Use action phrases: "Ensure...", "Standardize...", "Reduce...", "Improve..."

===JUSTIFICATION===
Write a "Business Justification" section:
- Opening sentence about operational benefits
- Then 4-6 numbered items with **bold title** and description

TRANSCRIPT:
{sample}"""

    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
    _timed("Overview+Justification", start)

    result = {"overview": "", "justification": ""}
    if response:
        ov = re.search(
            r'===OVERVIEW===\s*(.*?)(?====JUSTIFICATION===|$)',
            response, re.DOTALL
        )
        jf = re.search(r'===JUSTIFICATION===\s*(.*?)$', response, re.DOTALL)
        if ov:
            result["overview"] = ov.group(1).strip()
        if jf:
            result["justification"] = jf.group(1).strip()

    if not result["overview"]:
        result["overview"] = (
            f"The primary objective is to automate the {project_name} process "
            f"to ensure consistency, accuracy, and compliance."
        )
    if not result["justification"]:
        result["justification"] = (
            f"The {project_name} delivers operational efficiency "
            f"and governance control."
        )
    return result


# ============================================================
# CALL 4: "As Is" Process
# ============================================================

def get_as_is_process(
    transcript: str, project_name: str, entity_hint: str
) -> str:
    """Generate the current manual process (As Is state)."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=MAX_SAMPLE)

    prompt = f"""Document the CURRENT MANUAL PROCESS ("As Is" state).

Project: "{project_name}". {entity_hint}

Use ONLY names from the transcript.

Write 4-8 numbered steps. Each step:
- **Bold title**
- Description: what the person manually does
- Tools Used: applications used (ONLY from transcript)

Then add "Business Challenges" with 4-6 bullet points.

TRANSCRIPT:
{sample}

CURRENT MANUAL PROCESS:"""

    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
    _timed("As-Is", start)

    if response and len(response) > 100:
        return response.strip()
    return (
        f"The current {project_name} process is performed manually. "
        f"Details to be documented during implementation."
    )


# ============================================================
# CALL 5: "To Be" Process
# ============================================================

def get_to_be_process(
    transcript: str, project_name: str, entity_hint: str
) -> str:
    """Generate the future automated process (To Be state)."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=MAX_SAMPLE)

    prompt = f"""Write the "To Be" automated process description.

Project: "{project_name}". {entity_hint}

Use ONLY names from the transcript.

Write 2-3 paragraphs describing how the automation handles this process end-to-end:
- Write as if the automation already exists
- Use: "The system will...", "The automation will automatically..."
- Cover: trigger → connection → data handling → processing → validation → action → reporting → logging
- End with audit readiness and compliance

TRANSCRIPT:
{sample}

TO-BE PROCESS:"""

    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
    _timed("To-Be", start)

    if response and len(response) > 100:
        return response.strip()
    return (
        f"The {project_name} will use an automation solution to "
        f"handle the end-to-end process."
    )


# ============================================================
# CALL 6: Process Steps (Technology-Neutral)
# ============================================================

def extract_process_steps(transcript: str, entities: Dict = None) -> List[str]:
    """
    Extract automation process steps.
    Technology-neutral — works for any automation type.
    Uses 3 strategies with increasingly simple prompts.
    """
    start = time.time()

    if entities is None:
        entities, _ = extract_entities_and_project(transcript)

    entity_hint = _build_entity_hint(entities)
    all_steps = []

    # Strategy 1: Full extraction
    print("    [Steps] Strategy 1: Full extraction...")
    sample = _safe_sample(transcript, max_len=MAX_SAMPLE)

    prompt = f"""Extract the automation process steps from this meeting transcript.

{entity_hint}

Use ONLY names from the transcript.

Write 8-15 numbered steps describing what the AUTOMATED SYSTEM does.

CRITICAL: Extract PROCESS ACTIONS, not meeting conversation.

WRONG (meeting summaries):
- "Team discussed downloading data"
- "Agreed to validate records"

CORRECT (system actions):
- "Connects to [Application] using authorized credentials."
- "Extracts relevant data from the source system."
- "Filters records based on defined business rules."
- "Validates each record against the defined criteria."
- "Performs the required action for eligible records."
- "Generates a report containing processed records and outcomes."
- "Updates the execution status based on results."
- "Logs execution details for audit and tracking."

Each step should:
- Start with a verb: Connects, Extracts, Validates, Navigates, Generates, Updates, Logs
- Describe a SYSTEM action
- Be 1-2 sentences

TRANSCRIPT:
{sample}

PROCESS STEPS:
1."""

    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)

    if response:
        if not response.strip().startswith("1"):
            response = "1. " + response
        all_steps = _parse_numbered_steps(response)
        print(f"    [Steps] Strategy 1: {len(all_steps)} steps")

    # Strategy 2: Simplified
    if len(all_steps) < 3:
        print("    [Steps] Strategy 2: Simplified...")
        prompt = f"""What steps will an automated system perform for this process?

{entity_hint}

Use ONLY names from the transcript.
Write 8-12 steps. Start each with a verb.

{_safe_sample(transcript, MAX_SAMPLE_SMALL)}

Steps:
1."""
        response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
        if response:
            if not response.strip().startswith("1"):
                response = "1. " + response
            s2 = _parse_numbered_steps(response)
            if len(s2) > len(all_steps):
                all_steps = s2
            print(f"    [Steps] Strategy 2: {len(s2)} steps")

    # Strategy 3: Template fallback
    if len(all_steps) < 3:
        print("    [Steps] Strategy 3: Template...")
        all_steps = _template_steps(entities)

    unique = _deduplicate_steps(all_steps)
    if len(unique) > 20:
        mid = len(unique) // 2
        unique = unique[:8] + unique[mid-2:mid+2] + unique[-8:]
    if not unique:
        unique = _template_steps(entities)

    _timed(f"Steps ({len(unique)})", start)
    return unique


def _template_steps(entities: Dict) -> List[str]:
    """Generate template steps when LLM extraction fails. Technology-neutral."""
    apps = ', '.join(entities.get('applications', [])) or 'the target application'
    return [
        f"Connects to {apps} using authorized credentials.",
        f"Extracts relevant data from {apps}.",
        "Filters and processes records based on defined business rules.",
        "Validates each record against the defined criteria.",
        f"Performs the required actions for eligible records in {apps}.",
        "Captures updated status after processing each record.",
        "Generates a report containing processed records and outcomes.",
        "Updates the execution status based on results.",
        "Logs execution details for audit and tracking.",
    ]


def _parse_numbered_steps(text: str) -> List[str]:
    """Parse numbered steps from LLM response."""
    steps = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line).strip()
        cleaned = re.sub(r'^[-•*➤]\s*', '', cleaned).strip()
        cleaned = cleaned.strip('"')
        if not cleaned or len(cleaned) < 10:
            continue
        skip = [
            'here are', 'following', 'process steps', 'transcript',
            'note:', 'section', 'based on', 'the above', 'these are',
            'below', 'i have', 'let me', 'sure,', 'certainly',
            'wrong', 'correct', 'critical', 'example', '❌', '✅',
            'important', 'context', 'the meeting', 'discussed',
            'use only', 'names from'
        ]
        if any(cleaned.lower().startswith(p) for p in skip):
            continue
        if cleaned.startswith('WRONG') or cleaned.startswith('RIGHT'):
            continue
        if cleaned.isupper() or cleaned.endswith(':'):
            continue
        steps.append(cleaned)
    return steps


def _deduplicate_steps(steps: List[str]) -> List[str]:
    """Remove near-duplicate steps."""
    seen = set()
    unique = []
    for s in steps:
        key = re.sub(r'[^a-z]', '', s.lower())[:40]
        if key not in seen and len(key) > 5:
            seen.add(key)
            unique.append(s)
    return unique


# ============================================================
# CALL 7: Input Requirements
# ============================================================

def get_input_requirements(
    transcript: str, project_name: str, entity_hint: str
) -> List[Dict]:
    """Extract input requirements for the automation."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=MAX_SAMPLE_SMALL)

    prompt = f"""Identify input requirements for this automation project.

Project: "{project_name}". {entity_hint}

Use ONLY names from the transcript.

List 3-8 inputs the automation needs. Consider:
- Credentials/access for applications
- Source data (files, databases, portals)
- Configuration parameters
- Identifiers for lookups

FORMAT (one per line):
INPUT: Parameter Name | DESCRIPTION: What it is and why needed

TRANSCRIPT:
{sample}

INPUTS:"""

    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
    _timed("Inputs", start)

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
                        param = re.sub(
                            r'^INPUT[S]?\s*[:]\s*', '', part,
                            flags=re.IGNORECASE
                        ).strip()
                    elif part.upper().startswith('DESC'):
                        desc = re.sub(
                            r'^DESCRIPTION\s*[:]\s*', '', part,
                            flags=re.IGNORECASE
                        ).strip()
                if param:
                    inputs.append({
                        "parameter": param,
                        "description": desc or "Required for execution."
                    })

    if not inputs:
        inputs = [
            {"parameter": "Application Credentials",
             "description": "Authorized credentials for secure access."},
            {"parameter": "Source Data",
             "description": "Data from source applications."},
            {"parameter": "Automation Configuration",
             "description": "Configured workflow for execution."},
        ]
    return inputs


# ============================================================
# CALL 8: Detailed Process Steps
# ============================================================

def get_detailed_process_steps(
    transcript: str, project_name: str, entity_hint: str
) -> List[Dict]:
    """
    Extract detailed step-by-step process for Section 2.4.
    Each step becomes a subsection with optional screenshot.
    """
    start = time.time()
    sample = _safe_sample(transcript, max_len=MAX_SAMPLE)

    prompt = f"""Write detailed step-by-step instructions for the automated process.

Project: "{project_name}". {entity_hint}

Use ONLY names from the transcript.

List 10-25 detailed steps describing specific screen-level actions:
- Logging into applications
- Navigating to specific pages/tabs
- Clicking buttons or menu items
- Entering data in fields
- Downloading or exporting files
- Processing or validating records
- Generating reports
- Updating status

Numbered list, each step 1-2 sentences.

TRANSCRIPT:
{sample}

STEPS:
1."""

    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
    _timed("Detailed Steps", start)

    detailed = []
    if response:
        if not response.strip().startswith("1"):
            response = "1. " + response
        parsed = _parse_numbered_steps(response)
        for i, step in enumerate(parsed):
            detailed.append({
                "number": f"2.4.{i+1}",
                "description": step
            })

    if not detailed:
        detailed = [
            {"number": "2.4.1",
             "description": "Log in to the target application."},
            {"number": "2.4.2",
             "description": "Navigate to the relevant section."},
            {"number": "2.4.3",
             "description": "Extract and process the required data."},
            {"number": "2.4.4",
             "description": "Validate records against defined criteria."},
            {"number": "2.4.5",
             "description": "Perform required actions for validated records."},
            {"number": "2.4.6",
             "description": "Generate reports and update execution status."},
        ]
    return detailed


# ============================================================
# CALL 9: Interface Requirements
# ============================================================

def get_interface_requirements(transcript: str, entities: Dict) -> List[Dict]:
    """Extract interface/application requirements."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=MAX_SAMPLE_SMALL)
    apps_hint = ""
    if entities.get('applications'):
        apps_hint = (
            f"Applications from transcript: "
            f"{', '.join(entities['applications'])}"
        )

    prompt = f"""Identify application interfaces needed for this automation.

{apps_hint}

ONLY list applications explicitly mentioned in the transcript.

FORMAT (one per line):
APP: Name | INTERFACE: Web/Desktop/API/Database | PURPOSE: Why needed

TRANSCRIPT:
{sample}

INTERFACES:"""

    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
    _timed("Interfaces", start)

    apps = []
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if '|' in line and 'APP' in line.upper():
                app = {
                    "application": "", "interface": "",
                    "url": "", "purpose": ""
                }
                parts = line.split('|')
                for part in parts:
                    part = part.strip()
                    p_up = part.upper()
                    if p_up.startswith('APP'):
                        app["application"] = re.sub(
                            r'^APP(?:LICATION)?:\s*', '', part,
                            flags=re.IGNORECASE
                        ).strip()
                    elif p_up.startswith('INTERFACE'):
                        app["interface"] = part.split(':', 1)[-1].strip()
                    elif p_up.startswith('PURPOSE'):
                        app["purpose"] = part.split(':', 1)[-1].strip()
                if app["application"]:
                    apps.append(app)

    if not apps:
        apps = [{
            "application": "N/A", "interface": "N/A",
            "url": "N/A", "purpose": "N/A"
        }]
    return apps


# ============================================================
# CALL 10: Exception Handling
# ============================================================

def get_exception_handling(
    transcript: str, project_name: str, entity_hint: str
) -> List[Dict]:
    """Generate exception handling scenarios."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=MAX_SAMPLE_SMALL)

    prompt = f"""Write exception handling scenarios for this automation.

Project: "{project_name}". {entity_hint}

List 5-8 exceptions and how the system handles each.

FORMAT (one per line):
EXCEPTION: Scenario title | HANDLING: What the system does

Consider: login failures, missing data, records not found, processing errors, validation failures, system errors.

TRANSCRIPT:
{sample}

EXCEPTIONS:"""

    response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
    _timed("Exceptions", start)

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
                        exc["exception"] = re.sub(
                            r'^EXCEPTION:\s*', '', part,
                            flags=re.IGNORECASE
                        ).strip()
                    elif part.upper().startswith('HANDLING'):
                        exc["handling"] = re.sub(
                            r'^HANDLING:\s*', '', part,
                            flags=re.IGNORECASE
                        ).strip()
                if exc["exception"]:
                    exceptions.append(exc)

    if not exceptions:
        exceptions = [
            {"exception": "Application Login Failure",
             "handling": "Stop execution, log error, notify support."},
            {"exception": "Missing or Invalid Input Data",
             "handling": "Log failure, skip affected record."},
            {"exception": "Record Not Found",
             "handling": "Log details, skip, continue processing."},
            {"exception": "Processing Error",
             "handling": "Log error, mark failed, continue with others."},
            {"exception": "System Exception",
             "handling": "Capture details in error log for audit."},
        ]
    return exceptions


# ============================================================
# DOT Code Generation — Labels on NODES
# ============================================================

def generate_dot_and_apps(process_steps, transcript, entities=None):
    """Generate DOT flowchart code and interface requirements."""
    start = time.time()
    if entities is None:
        entities, _ = extract_entities_and_project(transcript)
    dot_code = _generate_dot_from_steps(process_steps)
    apps = get_interface_requirements(transcript, entities)
    _timed("DOT+Apps", start)
    return dot_code, apps


def _generate_dot_from_steps(steps):
    """
    Generate DOT code from process steps.
    Labels on NODES, simple edges, decisions with Yes/No.
    """
    if not steps:
        return _manual_dot(["Start", "Process", "End"])
    chart = steps[:12]
    text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(chart)])

    prompt = f"""Convert these steps into Graphviz DOT flowchart code.

RULES:
1. Descriptive text goes in NODE labels: Step1 [label="Connect to App"]
2. Edges are simple arrows: Step1 -> Step2;
3. ONLY decision edges get labels: Decision1 -> Step3 [label="Yes"];
4. Start/End: shape=oval. Steps: shape=box. Decisions: shape=diamond.
5. Keep labels short (max 6-8 words).
6. Every node MUST have label="..."

Output ONLY DOT code.

Steps:
{text}

digraph ProcessFlow {{"""

    response = llm_client.generate(prompt, temperature=0.2)
    if response:
        dot = _extract_dot(response)
        if dot and re.search(r'\w+\s*\[.*?label\s*=', dot, re.IGNORECASE):
            return dot
        else:
            print("    [DOT] LLM output invalid, using manual generation")
    return _manual_dot(steps)


def _extract_dot(response):
    """Extract DOT code from LLM response."""
    m = re.search(r'```(?:dot|graphviz|)?\s*\n?(.*?)```', response, re.DOTALL)
    if m and 'digraph' in m.group(1):
        return m.group(1).strip()
    m = re.search(r'(digraph\s+\w*\s*\{.*\})', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    if 'digraph' in response and '->' in response:
        text = response.strip()
        if not text.startswith('digraph'):
            text = 'digraph ProcessFlow {\n' + text
        if not text.endswith('}'):
            text += '\n}'
        return text
    return None


def _manual_dot(steps):
    """Generate DOT code manually. Labels on nodes, simple edges."""
    lines = [
        'digraph ProcessFlow {',
        '    rankdir=TB;',
        '    node [fontname="Arial", fontsize=10, style=filled];',
        '    edge [fontname="Arial", fontsize=9];',
        '',
        '    Start [label="Start", shape=oval, fillcolor=lightgreen];'
    ]
    ids = ['Start']

    for i, s in enumerate(steps[:15]):
        nid = f'Step{i+1}'
        label = s.replace('[DECISION]', '').strip().replace('"', '\\"')
        words = label.split()
        if len(words) > 8:
            label = ' '.join(words[:8]) + '...'
        if len(label) > 35:
            w = label.split()
            m = len(w) // 2
            label = ' '.join(w[:m]) + '\\n' + ' '.join(w[m:])

        is_decision = '[DECISION]' in s
        shape = 'diamond' if is_decision else 'box'
        color = 'gold' if is_decision else 'lightblue'
        lines.append(
            f'    {nid} [label="{label}", shape={shape}, fillcolor={color}];'
        )
        ids.append(nid)

    lines.append('    End [label="End", shape=oval, fillcolor=lightcoral];')
    ids.append('End')
    lines.append('')

    for i in range(len(ids) - 1):
        lines.append(f'    {ids[i]} -> {ids[i+1]};')

    lines.append('}')
    return '\n'.join(lines)


# ============================================================
# Timestamps + Paraphrase
# ============================================================

def identify_key_timestamps(transcript, transcript_path):
    """Identify key action timestamps from transcript."""
    start = time.time()
    lines = []
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(
                    r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(.*)',
                    line.strip()
                )
                if m:
                    lines.append({
                        "timestamp": float(m.group(1)),
                        "text": m.group(3).strip()
                    })
    except Exception:
        return []
    if not lines:
        return []

    step = max(1, len(lines) // 20)
    sampled = lines[::step][:20]
    text = "\n".join(
        [f"[{l['timestamp']:.1f}] {l['text']}" for l in sampled]
    )

    prompt = f"""Which lines show a PROCESS ACTION (opening app, clicking, typing, navigating)?
NOT: talking, explaining, greeting.

Per line: [time] YES action  OR  [time] NO

{text}

Answers:"""

    response = llm_client.generate(prompt)
    moments = []
    if response:
        for line in response.split('\n'):
            m = re.search(
                r'\[?(\d+\.?\d*)\]?\s*YES\s*[-:.]?\s*(.*)',
                line, re.IGNORECASE
            )
            if m:
                ts = float(m.group(1))
                desc = m.group(2).strip()
                if not desc:
                    for sl in sampled:
                        if abs(sl["timestamp"] - ts) < 1.0:
                            desc = sl["text"]
                            break
                moments.append({
                    "timestamp": ts,
                    "description": desc or "Process action"
                })

    # Fallback: keyword matching
    if len(moments) < 3:
        all_kw = set()
        for kl in ACTION_KEYWORDS.values():
            for kw in kl:
                all_kw.add(kw.lower())
        for tl in lines:
            if any(kw in tl["text"].lower() for kw in all_kw):
                moments.append({
                    "timestamp": tl["timestamp"],
                    "description": tl["text"]
                })

    # Deduplicate
    if moments:
        deduped = [moments[0]]
        for km in moments[1:]:
            if abs(km["timestamp"] - deduped[-1]["timestamp"]) > 5.0:
                deduped.append(km)
        moments = deduped

    if len(moments) > 15:
        s = len(moments) // 15
        moments = moments[::s][:15]

    _timed(f"Timestamps ({len(moments)})", start)
    return moments


def paraphrase_batch(texts, batch_size=5):
    """Paraphrase frame descriptions into professional process steps."""
    if not texts:
        return []
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        numbered = "\n".join(
            [f"{j+1}. {t[:100]}" for j, t in enumerate(batch)]
        )

        prompt = f"""Rewrite each as a professional process step description.
What the SYSTEM does at this point. Third person, 1 sentence each.
Use ONLY names from the original text.

{numbered}

Rewritten:
1."""

        response = llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
        batch_results = []
        if response:
            if not response.strip().startswith("1"):
                response = "1. " + response
            batch_results = _parse_numbered_steps(response)
        for j in range(len(batch)):
            results.append(
                batch_results[j] if j < len(batch_results)
                else batch[j][:120]
            )
    return results


# ============================================================
# Backward compat wrappers
# ============================================================

def get_purpose_summary_io(transcript, project_name, entities=None):
    if entities is None:
        entities, _ = extract_entities_and_project(transcript)
    eh = _build_entity_hint(entities)
    return {
        "purpose": get_document_purpose_text(transcript, project_name, eh),
        "summary": get_to_be_process(transcript, project_name, eh),
        "inputs_outputs": get_as_is_process(transcript, project_name, eh),
        "overview": "", "justification": "", "as_is": "", "to_be": ""
    }


def extract_entities(t):
    return extract_entities_and_project(t)[0]

def get_project_name(t):
    return extract_entities_and_project(t)[1]

def get_document_purpose(t, pn):
    return get_document_purpose_text(t, pn, "")

def get_process_summary(t, e=None):
    return get_to_be_process(t, "Process", "")

def get_inputs_outputs(t, e=None):
    return get_as_is_process(t, "Process", "")

def generate_dot_code(t, e=None):
    s = extract_process_steps(t, e)
    d, _ = generate_dot_and_apps(s, t, e)
    return d

def get_applications_table(t, e=None):
    s = extract_process_steps(t, e)
    _, a = generate_dot_and_apps(s, t, e)
    return a