# src/llm_tasks/__init__.py

"""
LLM-based tasks for PDD/BRD generation.
Split into focused modules for maintainability.

Public API — import everything from here:
    from llm_tasks import extract_entities_and_project, extract_process_steps, ...
"""

# Utilities (used by other modules and by pdd_agent.py)
from llm_tasks.utils import (
    _build_entity_hint,
    _safe_sample,
    split_into_chunks,
    _filter_conversation_steps
)

# Call 1: Entities
from llm_tasks.entity_extraction import extract_entities_and_project

# Calls 2-5: Document sections
from llm_tasks.document_sections import (
    get_document_purpose_text,
    get_overview_and_justification,
    get_as_is_process,
    get_to_be_process
)

# Calls 6, 8: Process steps
from llm_tasks.process_steps import (
    extract_process_steps,
    get_detailed_process_steps
)

# Calls 7, 9, 10: Requirements
from llm_tasks.requirements import (
    get_input_requirements,
    get_interface_requirements,
    get_exception_handling
)

# DOT flowchart generation
from llm_tasks.flowchart_dot import generate_dot_and_apps

# Timestamps + paraphrase
from llm_tasks.timestamps import (
    identify_key_timestamps,
    paraphrase_batch
)

# Backward compatibility
from llm_tasks.compat import (
    get_purpose_summary_io,
    extract_entities,
    get_project_name,
    get_document_purpose,
    get_process_summary,
    get_inputs_outputs,
    generate_dot_code,
    get_applications_table
)