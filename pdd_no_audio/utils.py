# pdd_no_audio/utils.py

"""
Shared utility functions for PDD Agent (no-audio).
Text processing, operation detection, sampling, step parsing.
"""

import re
import time
from typing import List, Set, Dict

from pdd_no_audio.config import (
    EXCEL_OPERATIONS, WEB_OPERATIONS, GENERAL_OPERATIONS
)


def timed(name: str, start: float):
    """Log elapsed time for a task."""
    print(f"    [{name}] done in {time.time() - start:.1f}s")


def safe_sample(text: str, max_len: int = 3000) -> str:
    """Take a safe-sized sample from text. Beginning + End."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    first = int(max_len * 0.6)
    last = max_len - first - 30
    return text[:first] + "\n[...]\n" + text[-last:]


def parse_numbered_steps(text: str) -> List[str]:
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
        skip_prefixes = [
            'here are', 'following', 'process steps', 'note:',
            'section', 'based on', 'the above', 'these are',
            'below', 'i have', 'let me', 'sure,', 'certainly',
            'wrong', 'correct', 'critical', 'example',
            'important', 'context', 'use only', 'names from'
        ]
        if any(cleaned.lower().startswith(p) for p in skip_prefixes):
            continue
        if cleaned.isupper() or cleaned.endswith(':'):
            continue
        steps.append(cleaned)
    return steps


def deduplicate_steps(steps: List[str]) -> List[str]:
    """Remove near-duplicate steps."""
    seen: Set[str] = set()
    unique = []
    for s in steps:
        key = re.sub(r'[^a-z]', '', s.lower())[:40]
        if key not in seen and len(key) > 5:
            seen.add(key)
            unique.append(s)
    return unique


def clean_vision_response(text: str) -> str:
    """Clean up vision model response text."""
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)
    return '\n'.join(cleaned).strip()


def extract_application_name(vision_descriptions: List[str]) -> str:
    """Try to extract application name from vision descriptions."""
    if not vision_descriptions:
        return ""
    app_patterns = [
        r'(?:application|app|software|website|portal|system)\s+(?:is\s+)?["\']?([A-Z][a-zA-Z0-9\s]+)',
        r'(?:shows?|displays?|using)\s+["\']?([A-Z][a-zA-Z0-9\s]{2,20})',
    ]
    for pattern in app_patterns:
        match = re.search(pattern, ' '.join(vision_descriptions[:3]))
        if match:
            name = match.group(1).strip()
            if len(name) > 2 and len(name) < 30:
                return name
    return ""


# ============================================================
# Operation Detection
# ============================================================

def detect_operations(
    ocr_text: str,
    vision_description: str,
    change_description: str = ""
) -> List[Dict]:
    """
    Detect specific operations from OCR text and vision descriptions.
    Returns list of detected operations with category and confidence.
    """
    combined_text = f"{ocr_text} {vision_description} {change_description}".lower()
    detected = []

    # Check Excel operations
    for op_name, keywords in EXCEL_OPERATIONS.items():
        for kw in keywords:
            if kw in combined_text:
                detected.append({
                    "category": "Excel",
                    "operation": op_name,
                    "keyword_matched": kw,
                    "display_name": _format_operation_name(op_name)
                })
                break

    # Check Web operations
    for op_name, keywords in WEB_OPERATIONS.items():
        for kw in keywords:
            if kw in combined_text:
                detected.append({
                    "category": "Web",
                    "operation": op_name,
                    "keyword_matched": kw,
                    "display_name": _format_operation_name(op_name)
                })
                break

    # Check General operations
    for op_name, keywords in GENERAL_OPERATIONS.items():
        for kw in keywords:
            if kw in combined_text:
                detected.append({
                    "category": "General",
                    "operation": op_name,
                    "keyword_matched": kw,
                    "display_name": _format_operation_name(op_name)
                })
                break

    return detected


def _format_operation_name(op_name: str) -> str:
    """Convert operation key to display name."""
    display_map = {
        "vlookup": "VLOOKUP Formula",
        "hlookup": "HLOOKUP Formula",
        "filter": "Data Filter",
        "sort": "Data Sort",
        "pivot_table": "Pivot Table",
        "duplicate_removal": "Duplicate Removal",
        "conditional_formatting": "Conditional Formatting",
        "formula": "Formula/Calculation",
        "copy_paste": "Copy & Paste",
        "find_replace": "Find & Replace",
        "freeze_panes": "Freeze Panes",
        "merge_cells": "Merge Cells",
        "insert_delete": "Insert/Delete Rows or Columns",
        "chart": "Chart/Graph Creation",
        "macro": "Macro Execution",
        "import_export": "Import/Export Data",
        "text_to_columns": "Text to Columns",
        "data_validation": "Data Validation",
        "subtotal": "Subtotal/Grouping",
        "concatenate": "Text Concatenation",
        "login": "User Authentication",
        "navigate": "Page Navigation",
        "search": "Search/Query",
        "form_fill": "Form Data Entry",
        "upload": "File Upload",
        "download": "File Download",
        "submit": "Form Submission",
        "select": "Selection",
        "scroll": "Page Scroll",
        "expand_collapse": "Expand/Collapse Section",
        "table_interaction": "Table Interaction",
        "modal_dialog": "Dialog/Popup Interaction",
        "refresh": "Page Refresh",
        "logout": "User Logout",
        "open_application": "Application Launch",
        "close_application": "Application Close",
        "switch_window": "Window Switch",
        "copy_data": "Data Copy",
        "paste_data": "Data Paste",
        "email": "Email Operation",
        "file_operation": "File Management",
    }
    return display_map.get(op_name, op_name.replace('_', ' ').title())


def build_operation_context(operations: List[Dict]) -> str:
    """Build a context string from detected operations for LLM prompts."""
    if not operations:
        return ""

    lines = ["Detected operations on this screen:"]
    for op in operations:
        lines.append(f"- {op['category']}: {op['display_name']}")

    return "\n".join(lines)