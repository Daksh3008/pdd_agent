# pdd_no_audio/utils.py

"""
Shared utility functions for PDD Agent (no-audio).
Text processing, operation detection, sampling, step parsing.
Enhanced with auth/login detection and parameter extraction.
"""

import re
import time
from typing import List, Set, Dict, Optional

from pdd_no_audio.config import (
    EXCEL_OPERATIONS, WEB_OPERATIONS, GENERAL_OPERATIONS,
    AUTH_VISUAL_INDICATORS
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
# Auth/Login Screen Detection
# ============================================================

def detect_auth_screen(ocr_text: str, vision_text: str = "") -> Dict:
    """
    Detect if a frame shows a login/logout/auth screen.
    Returns dict with auth_type and confidence.
    """
    combined = f"{ocr_text} {vision_text}".lower()
    if not combined.strip():
        return {"is_auth": False, "auth_type": "", "confidence": 0.0, "indicators": []}

    found_indicators = []
    for indicator in AUTH_VISUAL_INDICATORS:
        if indicator in combined:
            found_indicators.append(indicator)

    if not found_indicators:
        return {"is_auth": False, "auth_type": "", "confidence": 0.0, "indicators": []}

    # Classify auth type
    auth_type = "unknown_auth"
    login_words = {"sign in", "log in", "login", "signin", "username", "password",
                   "credentials", "welcome", "sso", "authenticate"}
    logout_words = {"sign out", "log out", "logout", "signout", "signed out",
                    "goodbye", "end session", "session expired"}
    password_words = {"change password", "reset password", "forgot password",
                      "new password", "password expired", "update password"}
    mfa_words = {"mfa", "otp", "verification code", "two-factor", "2fa",
                 "multi-factor", "captcha"}

    login_hits = sum(1 for i in found_indicators if i in login_words or
                     any(lw in i for lw in login_words))
    logout_hits = sum(1 for i in found_indicators if i in logout_words or
                      any(lw in i for lw in logout_words))
    password_hits = sum(1 for i in found_indicators if i in password_words or
                        any(pw in i for pw in password_words))
    mfa_hits = sum(1 for i in found_indicators if i in mfa_words or
                   any(mw in i for mw in mfa_words))

    if logout_hits > 0 and logout_hits >= login_hits:
        auth_type = "logout"
    elif password_hits > 0 and password_hits >= login_hits:
        auth_type = "password_change"
    elif mfa_hits > 0:
        auth_type = "mfa_verification"
    elif login_hits > 0:
        auth_type = "login"

    # Confidence based on number of indicators
    confidence = min(1.0, len(found_indicators) / 3.0)

    # Boost confidence if multiple categories of indicators found
    has_field = any(i in ["username", "user name", "password", "email", "user id", "userid"]
                    for i in found_indicators)
    has_action = any(i in ["sign in", "log in", "login", "submit", "continue",
                           "sign out", "log out", "logout"]
                     for i in found_indicators)
    if has_field and has_action:
        confidence = min(1.0, confidence + 0.3)

    return {
        "is_auth": confidence >= 0.3,
        "auth_type": auth_type,
        "confidence": confidence,
        "indicators": found_indicators
    }


def get_auth_step_description(auth_type: str, indicators: List[str],
                               app_name: str = "") -> str:
    """
    Generate a detailed auth step description for PDD.
    Used as fallback when LLM doesn't capture auth actions.
    """
    app = app_name or "the application"

    descriptions = {
        "login": (
            f"The system navigates to the login page of {app}. "
            f"The automation enters the configured username/user ID into the "
            f"username field and the corresponding password into the password field. "
            f"The 'Sign In' / 'Log In' button is clicked to authenticate and "
            f"gain access to the application. The system waits for the home page "
            f"or dashboard to load confirming successful authentication."
        ),
        "logout": (
            f"The system initiates the logout process from {app}. "
            f"The automation clicks on the user profile menu or logout option "
            f"and selects 'Sign Out' / 'Log Out' to end the current session. "
            f"The system confirms the session has been terminated and the "
            f"login page is displayed."
        ),
        "mfa_verification": (
            f"The system encounters a multi-factor authentication (MFA) prompt "
            f"in {app}. The automation handles the verification step by "
            f"entering the OTP/verification code or completing the required "
            f"authentication challenge. The system waits for successful "
            f"verification before proceeding."
        ),
        "password_change": (
            f"The system navigates to the password management section of {app}. "
            f"The automation enters the current password, followed by the new "
            f"password and confirmation. The 'Update Password' / 'Change Password' "
            f"button is clicked and the system verifies the password change "
            f"was successful."
        ),
        "unknown_auth": (
            f"The system interacts with an authentication screen in {app}. "
            f"The automation handles the required credential entry and "
            f"authentication steps to proceed with the process."
        ),
    }

    return descriptions.get(auth_type, descriptions["unknown_auth"])


# ============================================================
# Operation Detection with Parameter Extraction
# ============================================================

def extract_operation_parameters(text: str, operation: str) -> Dict:
    """
    Extract parameters for specific operations using regex.
    Returns dict with extracted fields.
    """
    params = {}
    text_lower = text.lower()
    if operation == "vlookup":
        match = re.search(
            r'vlookup\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*(\d+)\s*,?\s*([^)]*)\s*\)',
            text_lower, re.IGNORECASE
        )
        if match:
            params["lookup_value"] = match.group(1).strip()
            params["table_array"] = match.group(2).strip()
            params["col_index"] = match.group(3).strip()
            if match.group(4):
                params["range_lookup"] = match.group(4).strip()
    elif operation == "filter":
        match = re.search(
            r'filter\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,?\s*([^)]*)\s*\)',
            text_lower, re.IGNORECASE
        )
        if match:
            params["array"] = match.group(1).strip()
            params["include"] = match.group(2).strip()
    return params


def detect_operations(
    ocr_text: str,
    vision_description: str,
    change_description: str = ""
) -> List[Dict]:
    """
    Detect specific operations from OCR text and vision descriptions.
    Returns list of detected operations with category, confidence, and parameters.
    """
    combined_text = f"{ocr_text} {vision_description} {change_description}".lower()
    detected = []

    # Check Excel operations
    for op_name, keywords in EXCEL_OPERATIONS.items():
        for kw in keywords:
            if kw in combined_text:
                params = extract_operation_parameters(combined_text, op_name)
                detected.append({
                    "category": "Excel",
                    "operation": op_name,
                    "keyword_matched": kw,
                    "display_name": _format_operation_name(op_name),
                    "parameters": params
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
                    "display_name": _format_operation_name(op_name),
                    "parameters": {}
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
                    "display_name": _format_operation_name(op_name),
                    "parameters": {}
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
        # Auth operations
        "login": "User Login/Authentication",
        "logout": "User Logout/Sign-Out",
        "session_management": "Session Management",
        "password_management": "Password Management",
        "user_profile": "User Profile Access",
        # Other web
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
        # General
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
        op_line = f"- {op['category']}: {op['display_name']}"
        if op.get("parameters"):
            param_str = ", ".join(f"{k}={v}" for k, v in op["parameters"].items())
            op_line += f" ({param_str})"
        lines.append(op_line)

    return "\n".join(lines)