# llm/requirements.py

"""
Requirements extraction from transcript.
Input requirements, interface requirements, exception handling.
Used by the audio pipeline.
"""

import re
import time
from typing import Dict, List

from core.gemini_client import gemini_client
from core.config import config
from core.utils import timed, safe_sample
from llm_tasks.system_prompts import get_system_prompt


def get_input_requirements(
    transcript: str,
    project_name: str,
    entity_hint: str
) -> List[Dict]:
    """Extract input requirements from transcript."""
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_small)
    doc_type = config.document.document_type

    if doc_type == "BRD":
        context = (
            "List 3-8 business inputs and data sources the solution requires.\n"
            "Focus on business data, not technical credentials."
        )
    else:
        context = (
            "List 3-8 inputs the automation needs. Consider:\n"
            "- Credentials/access for applications\n"
            "- Source data (files, databases, portals)\n"
            "- Configuration parameters"
        )

    prompt = f"""Identify input requirements for this project.

Project: "{project_name}". {entity_hint}

{context}

Use ONLY names from the transcript.

FORMAT (one per line):
INPUT: Parameter Name | DESCRIPTION: What it is and why needed

TRANSCRIPT:
{sample}

INPUTS:"""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="InputRequirements"
    )
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


def get_interface_requirements(
    transcript: str,
    entities: Dict
) -> List[Dict]:
    """Extract interface/application requirements from transcript."""
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_small)
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

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="InterfaceRequirements"
    )
    timed("Interfaces", start)

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


def get_exception_handling(
    transcript: str,
    project_name: str,
    entity_hint: str
) -> List[Dict]:
    """Generate exception handling scenarios from transcript."""
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_small)
    doc_type = config.document.document_type

    if doc_type == "BRD":
        context = (
            "List 5-8 business exception scenarios and required system behavior.\n"
            "Focus on business impact and recovery requirements."
        )
    else:
        context = (
            "List 5-8 technical exception scenarios and how the system handles each.\n"
            "Consider: login failures, missing data, records not found, "
            "processing errors, validation failures, system errors."
        )

    prompt = f"""Write exception handling scenarios.

Project: "{project_name}". {entity_hint}

{context}

FORMAT (one per line):
EXCEPTION: Scenario title | HANDLING: What the system does

TRANSCRIPT:
{sample}

EXCEPTIONS:"""

    response = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        call_name="ExceptionHandling"
    )
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