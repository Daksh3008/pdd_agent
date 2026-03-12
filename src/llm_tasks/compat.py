# src/llm_tasks/compat.py

"""
Backward compatibility wrappers.
These maintain the old single-function API for any code
that imports from llm_tasks directly.
"""

from llm_tasks.entity_extraction import extract_entities_and_project
from llm_tasks.document_sections import (
    get_document_purpose_text,
    get_as_is_process,
    get_to_be_process
)
from llm_tasks.process_steps import extract_process_steps
from llm_tasks.flowchart_dot import generate_dot_and_apps
from llm_tasks.utils import _build_entity_hint


def get_purpose_summary_io(transcript, project_name, entities=None):
    """Legacy wrapper."""
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
    """Legacy wrapper."""
    return extract_entities_and_project(t)[0]


def get_project_name(t):
    """Legacy wrapper."""
    return extract_entities_and_project(t)[1]


def get_document_purpose(t, pn):
    """Legacy wrapper."""
    return get_document_purpose_text(t, pn, "")


def get_process_summary(t, e=None):
    """Legacy wrapper."""
    return get_to_be_process(t, "Process", "")


def get_inputs_outputs(t, e=None):
    """Legacy wrapper."""
    return get_as_is_process(t, "Process", "")


def generate_dot_code(t, e=None):
    """Legacy wrapper."""
    s = extract_process_steps(t, e)
    d, _ = generate_dot_and_apps(s, t, e)
    return d


def get_applications_table(t, e=None):
    """Legacy wrapper."""
    s = extract_process_steps(t, e)
    _, a = generate_dot_and_apps(s, t, e)
    return a