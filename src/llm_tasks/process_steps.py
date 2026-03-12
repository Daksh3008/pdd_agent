# src/llm_tasks/process_steps.py

"""
Calls 6, 8: Process step extraction.
High-level automation steps and detailed screen-level steps.
Adapts for PDD (how) vs BRD (what).
"""

import time
from typing import Dict, List

from llm_client import llm_client
from config import doc_config, llm_params
from llm_tasks.system_prompt import get_system_prompt
from llm_tasks.utils import (
    _timed, _safe_sample, _build_entity_hint,
    _filter_conversation_steps, parse_numbered_steps, deduplicate_steps
)
from llm_tasks.entity_extraction import extract_entities_and_project


def extract_process_steps(transcript: str, entities: Dict = None) -> List[str]:
    """
    Extract automation process steps.
    PDD: screen-level system actions
    BRD: functional requirements
    """
    start = time.time()

    if entities is None:
        entities, _ = extract_entities_and_project(transcript)

    entity_hint = _build_entity_hint(entities)
    doc_type = doc_config.document_type
    all_steps = []

    # Strategy 1: Full extraction
    print("    [Steps] Strategy 1: Full extraction...")
    sample = _safe_sample(transcript, max_len=llm_params.max_sample_text)

    if doc_type == "BRD":
        task_instruction = (
            "Write 8-15 numbered FUNCTIONAL REQUIREMENTS describing "
            "what the solution must do.\n"
            "Use 'The system shall...' or 'The solution must...' format.\n"
            "Focus on WHAT is needed, not HOW technically."
        )
        examples = (
            "- \"The system shall authenticate users via secure credentials.\"\n"
            "- \"The solution must extract and validate data from the source system.\"\n"
            "- \"The system shall generate audit-ready reports upon completion.\""
        )
    else:
        task_instruction = (
            "Write 8-15 numbered steps describing what the "
            "AUTOMATED SYSTEM does.\n"
            "Start each with a verb: Connects, Extracts, Validates, "
            "Navigates, Generates, Updates, Logs."
        )
        examples = (
            "- \"Connects to [Application] using authorized credentials.\"\n"
            "- \"Extracts relevant data from the source system.\"\n"
            "- \"Validates each record against the defined criteria.\"\n"
            "- \"Generates a report containing processed records and outcomes.\""
        )

    prompt = f"""Extract the process from this meeting transcript.

{entity_hint}

Use ONLY names from the transcript.

{task_instruction}

CRITICAL: Extract PROCESS ACTIONS, not meeting conversation.

WRONG (do NOT include):
- "Team discussed downloading data"
- "Coordinate with team member"
- "Schedule follow-up meeting"

CORRECT:
{examples}

TRANSCRIPT:
{sample}

STEPS:
1."""

    response = llm_client.generate(
        prompt, system_prompt=get_system_prompt(),
        call_name="ProcessSteps_S1"
    )

    if response:
        if not response.strip().startswith("1"):
            response = "1. " + response
        all_steps = parse_numbered_steps(response)
        all_steps = _filter_conversation_steps(all_steps)
        print(f"    [Steps] Strategy 1: {len(all_steps)} steps")

    # Strategy 2: Simplified
    if len(all_steps) < 3:
        print("    [Steps] Strategy 2: Simplified...")
        prompt = f"""What steps will an automated system perform for this process?

{entity_hint}

Use ONLY names from the transcript.
Write 8-12 steps. Start each with a verb.
Do NOT include conversations, scheduling, or coordination.

{_safe_sample(transcript, llm_params.max_sample_small)}

Steps:
1."""
        response = llm_client.generate(
            prompt, system_prompt=get_system_prompt(),
            call_name="ProcessSteps_S2"
        )
        if response:
            if not response.strip().startswith("1"):
                response = "1. " + response
            s2 = parse_numbered_steps(response)
            s2 = _filter_conversation_steps(s2)
            if len(s2) > len(all_steps):
                all_steps = s2
            print(f"    [Steps] Strategy 2: {len(s2)} steps")

    # Strategy 3: Template
    if len(all_steps) < 3:
        print("    [Steps] Strategy 3: Template...")
        all_steps = _template_steps(entities)

    unique = deduplicate_steps(all_steps)
    if len(unique) > 20:
        mid = len(unique) // 2
        unique = unique[:8] + unique[mid-2:mid+2] + unique[-8:]
    if not unique:
        unique = _template_steps(entities)

    _timed(f"Steps ({len(unique)})", start)
    return unique


def get_detailed_process_steps(
    transcript: str, project_name: str, entity_hint: str
) -> List[Dict]:
    """
    Extract detailed step-by-step process for Section 2.4.
    PDD: screen-level actions with screenshot alignment
    BRD: detailed functional requirements with acceptance criteria
    """
    start = time.time()
    sample = _safe_sample(transcript, max_len=llm_params.max_sample_text)
    doc_type = doc_config.document_type

    if doc_type == "BRD":
        instruction = (
            "List 10-25 detailed FUNCTIONAL REQUIREMENTS:\n"
            "- Each requirement describes a specific capability\n"
            "- Use 'The system shall...' format\n"
            "- Include validation rules, data handling, error conditions\n"
            "- Focus on business logic, not UI interactions"
        )
    else:
        instruction = (
            "List 10-25 detailed steps describing specific screen-level actions:\n"
            "- Logging into applications\n"
            "- Navigating to specific pages/tabs\n"
            "- Clicking buttons or menu items\n"
            "- Entering data in fields\n"
            "- Downloading or exporting files\n"
            "- Processing or validating records\n"
            "- Generating reports\n"
            "- Updating status"
        )

    prompt = f"""Write detailed instructions for the automated process.

Project: "{project_name}". {entity_hint}

Use ONLY names from the transcript.

{instruction}

Do NOT include steps about meetings, conversations, or coordinating with people.

Numbered list, each step 1-2 sentences.

TRANSCRIPT:
{sample}

STEPS:
1."""

    response = llm_client.generate(
        prompt, system_prompt=get_system_prompt(),
        call_name="DetailedSteps"
    )
    _timed("Detailed Steps", start)

    detailed = []
    if response:
        if not response.strip().startswith("1"):
            response = "1. " + response
        parsed = parse_numbered_steps(response)
        parsed = _filter_conversation_steps(parsed)
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


def _template_steps(entities: Dict) -> List[str]:
    """Generate template steps when LLM fails. Technology-neutral."""
    apps = (
        ', '.join(entities.get('applications', []))
        or 'the target application'
    )
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