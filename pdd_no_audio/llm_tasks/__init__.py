# pdd_no_audio/llm_tasks/__init__.py

"""
LLM tasks for PDD generation from silent screen recordings.
"""

from pdd_no_audio.llm_tasks.step_synthesizer import synthesize_pdd_steps, synthesize_single_step
from pdd_no_audio.llm_tasks.sop_sections import (
    generate_document_purpose,
    generate_overview_justification,
    generate_as_is_process,
    generate_to_be_process,
    generate_prerequisites,
    generate_exception_handling,
    generate_interface_requirements,
    generate_flowchart_dot,
    generate_all_sections_parallel
)
from pdd_no_audio.llm_tasks.system_prompts import PDD_SYSTEM_PROMPT, SOP_VISION_PROMPT