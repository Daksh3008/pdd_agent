# src/llm_tasks/document_sections.py

"""
Calls 2-5: Document section generation.
Purpose, Overview/Justification, As-Is Process, To-Be Process.
Adapts language for PDD vs BRD.
"""

import re
import time
from typing import Dict

from llm_client import llm_client
from config import doc_config, llm_params
from llm_tasks.system_prompt import get_system_prompt
from llm_tasks.utils import _timed, _safe_sample


def get_document_purpose_text(
    transcript: str, project_name: str, entity_hint: str
) -> str:
    """Generate the Purpose of this Document section."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=llm_params.max_sample_small)
    doc_type = doc_config.document_type
    doc_full = doc_config.document_type_full

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

    response = llm_client.generate(
        prompt, system_prompt=get_system_prompt(),
        call_name="DocumentPurpose"
    )
    _timed("Purpose", start)

    if response and len(response) > 50:
        return response.strip()

    return (
        f"This {doc_full} ({doc_type}) defines the "
        f"objectives, scope, and detailed business requirements for the "
        f"{project_name} initiative."
    )


def get_overview_and_justification(
    transcript: str, project_name: str, entity_hint: str
) -> Dict[str, str]:
    """Generate Overview/Objective and Business Justification."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=llm_params.max_sample_text)
    doc_full = doc_config.document_type_full
    doc_type = doc_config.document_type

    if doc_type == "BRD":
        overview_instruction = (
            "Write an 'Overview and Objective' section:\n"
            "- One paragraph stating the business need and objective\n"
            "- Then 4-6 bullet points of business outcomes expected\n"
            "- Use: 'Enable...', 'Ensure...', 'Provide...', 'Reduce...'"
        )
        justification_instruction = (
            "Write a 'Business Justification' section:\n"
            "- Opening sentence about business value\n"
            "- Then 4-6 numbered items with **bold title** and description\n"
            "- Focus on ROI, cost savings, compliance, risk reduction"
        )
    else:
        overview_instruction = (
            "Write an 'Overview and Objective' section:\n"
            "- One paragraph stating the primary objective\n"
            "- Then 4-6 bullet points of what the automation achieves\n"
            "- Use: 'Ensure...', 'Standardize...', 'Reduce...', 'Improve...'"
        )
        justification_instruction = (
            "Write a 'Business Justification' section:\n"
            "- Opening sentence about operational benefits\n"
            "- Then 4-6 numbered items with **bold title** and description"
        )

    prompt = f"""Write two sections for a {doc_full}.

Project: "{project_name}". {entity_hint}

Use ONLY names from the transcript.

===OVERVIEW===
{overview_instruction}

===JUSTIFICATION===
{justification_instruction}

TRANSCRIPT:
{sample}"""

    response = llm_client.generate(
        prompt, system_prompt=get_system_prompt(),
        call_name="OverviewJustification"
    )
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
    transcript: str, project_name: str, entity_hint: str
) -> str:
    """Generate the current manual process (As Is state)."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=llm_params.max_sample_text)
    doc_type = doc_config.document_type

    if doc_type == "BRD":
        instruction = (
            "Write 4-8 numbered items describing current business pain points:\n"
            "- **Bold title** for each pain point\n"
            "- Description of the business impact\n"
            "- Current tools/systems involved (ONLY from transcript)\n\n"
            "Then add 'Business Gaps' with 4-6 bullet points."
        )
    else:
        instruction = (
            "Write 4-8 numbered steps. Each step:\n"
            "- **Bold title**\n"
            "- Description: what the person manually does\n"
            "- Tools Used: applications used (ONLY from transcript)\n\n"
            "Then add 'Business Challenges' with 4-6 bullet points."
        )

    prompt = f"""Document the CURRENT STATE ("As Is").

Project: "{project_name}". {entity_hint}

Use ONLY names from the transcript.

{instruction}

TRANSCRIPT:
{sample}

CURRENT STATE:"""

    response = llm_client.generate(
        prompt, system_prompt=get_system_prompt(),
        call_name="AsIsProcess"
    )
    _timed("As-Is", start)

    if response and len(response) > 100:
        return response.strip()
    return (
        f"The current {project_name} process is performed manually. "
        f"Details to be documented during implementation."
    )


def get_to_be_process(
    transcript: str, project_name: str, entity_hint: str
) -> str:
    """Generate the future automated process (To Be state)."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=llm_params.max_sample_text)
    doc_type = doc_config.document_type

    if doc_type == "BRD":
        instruction = (
            "Write 2-3 paragraphs describing the desired future state:\n"
            "- What business outcomes the solution must deliver\n"
            "- What capabilities are required\n"
            "- Use: 'The solution shall...', 'The system must provide...'\n"
            "- End with success criteria and expected business impact"
        )
    else:
        instruction = (
            "Write 2-3 paragraphs describing how the automation handles "
            "this process end-to-end:\n"
            "- Write as if the automation already exists\n"
            "- Use: 'The system will...', 'The automation will automatically...'\n"
            "- Cover: trigger → connection → data handling → processing "
            "→ validation → action → reporting → logging\n"
            "- End with audit readiness and compliance"
        )

    prompt = f"""Write the "To Be" / future state description.

Project: "{project_name}". {entity_hint}

Use ONLY names from the transcript.

{instruction}

TRANSCRIPT:
{sample}

FUTURE STATE:"""

    response = llm_client.generate(
        prompt, system_prompt=get_system_prompt(),
        call_name="ToBeProcess"
    )
    _timed("To-Be", start)

    if response and len(response) > 100:
        return response.strip()
    return (
        f"The {project_name} will use an automation solution to "
        f"handle the end-to-end process."
    )