# src/llm_tasks/entity_extraction.py

"""
Call 1: Entity extraction and project name detection.
"""

import time
from typing import Dict, List, Tuple

from llm_client import llm_client
from config import llm_params
from llm_tasks.system_prompt import get_system_prompt
from llm_tasks.utils import (
    _timed, _safe_sample, _verify_entities_against_transcript
)


def extract_entities_and_project(transcript: str) -> Tuple[Dict[str, List[str]], str]:
    """Extract named entities and project name from transcript."""
    start = time.time()
    sample = _safe_sample(transcript, max_len=llm_params.max_sample_entity)

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

    response = llm_client.generate(
        prompt, system_prompt=get_system_prompt(),
        call_name="EntityExtraction"
    )

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