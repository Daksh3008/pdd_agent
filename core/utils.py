# core/utils.py

"""
Shared utility functions used across both pipelines.
Text processing, sampling, entity verification, operation detection, auth detection.
"""

import re
import time
from typing import List, Set, Dict, Optional

from core.config import (
    config, EXCEL_OPERATIONS, WEB_OPERATIONS,
    GENERAL_OPERATIONS, AUTH_VISUAL_INDICATORS
)


# ============================================================
# Timing
# ============================================================

def timed(name: str, start: float):
    """Log elapsed time for a task."""
    print(f"    [{name}] done in {time.time() - start:.1f}s")


# ============================================================
# Text Sampling
# ============================================================

def safe_sample(text: str, max_len: int = None) -> str:
    """Take a safe-sized sample. Beginning + End."""
    max_len = max_len or config.llm.max_sample_text
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    first = int(max_len * 0.6)
    last = max_len - first - 30
    return text[:first] + "\n[...]\n" + text[-last:]


def split_into_chunks(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    chunk_size = config.llm.chunk_size
    max_chunks = config.llm.max_chunks
    overlap = config.llm.overlap_size

    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(text))
        if end < len(text):
            bp = text.rfind('. ', start + chunk_size - 300, end)
            if bp != -1:
                end = bp + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start <= 0 and chunks:
            break
    return chunks


# ============================================================
# Entity Helpers
# ============================================================

def build_entity_hint(entities: Dict) -> str:
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


def verify_entities_against_transcript(entities: Dict, transcript: str) -> Dict:
    """Remove entity names that don't appear in the transcript."""
    transcript_lower = transcript.lower()

    def _appears(name: str) -> bool:
        name_lower = name.lower().strip()
        if name_lower in transcript_lower:
            return True
        words = name_lower.split()
        if len(words) > 1:
            significant = [w for w in words if len(w) > 3]
            if significant and all(w in transcript_lower for w in significant):
                return True
        if len(name_lower) >= 4 and name_lower[:4] in transcript_lower:
            return True
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
# Step Parsing and Filtering
# ============================================================

CONVERSATION_PHRASES = [
    'coordinate with', 'talk to', 'discuss with', 'meet with',
    'call ', 'email to', 'notify person', 'inform team',
    'schedule meeting', 'follow up with', 'follow-up with',
    'check with person', 'ask team', 'agreed to', 'decided to',
    'team will', 'we will', 'stakeholder meeting', 'agenda',
    'let me know', 'get back to', 'circle back', 'touch base',
    'set up a call', 'send invite', 'book a meeting'
]


def filter_conversation_steps(steps: List[str]) -> List[str]:
    """Remove steps that describe conversations/meetings, not process actions."""
    filtered = []
    for s in steps:
        s_lower = s.lower()
        if not any(phrase in s_lower for phrase in CONVERSATION_PHRASES):
            filtered.append(s)
        else:
            print(f"    [Filter] Removed conversation step: '{s[:60]}...'")
    return filtered if filtered else steps[:8]


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
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines).strip()


# ============================================================
# Auth / Login Detection
# ============================================================

def detect_auth_screen(ocr_text: str, vision_text: str = "") -> Dict:
    """Detect if a frame shows a login/logout/auth screen."""
    combined = f"{ocr_text} {vision_text}".lower()
    if not combined.strip():
        return {"is_auth": False, "auth_type": "", "confidence": 0.0, "indicators": []}

    found_indicators = []
    for indicator in AUTH_VISUAL_INDICATORS:
        if indicator in combined:
            found_indicators.append(indicator)

    if not found_indicators:
        return {"is_auth": False, "auth_type": "", "confidence": 0.0, "indicators": []}

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

    auth_type = "unknown_auth"
    if logout_hits > 0 and logout_hits >= login_hits:
        auth_type = "logout"
    elif password_hits > 0 and password_hits >= login_hits:
        auth_type = "password_change"
    elif mfa_hits > 0:
        auth_type = "mfa_verification"
    elif login_hits > 0:
        auth_type = "login"

    confidence = min(1.0, len(found_indicators) / 3.0)
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


def get_auth_step_description(auth_type: str, indicators: List[str] = None,
                               app_name: str = "") -> str:
    """Generate a detailed auth step description for PDD."""
    app = app_name or "the application"
    descriptions = {
        "login": (
            f"The system navigates to the login page of {app}. "
            f"The automation enters the configured username/user ID and password. "
            f"The 'Sign In' button is clicked to authenticate."
        ),
        "logout": (
            f"The system initiates the logout process from {app}. "
            f"The automation clicks the logout option to end the session."
        ),
        "mfa_verification": (
            f"The system handles a multi-factor authentication prompt "
            f"in {app} by completing the verification challenge."
        ),
        "password_change": (
            f"The system navigates to the password management section of {app} "
            f"and submits the password change."
        ),
        "unknown_auth": (
            f"The system interacts with an authentication screen in {app}."
        ),
    }
    return descriptions.get(auth_type, descriptions["unknown_auth"])


# ============================================================
# Operation Detection (Delta-based)
# ============================================================

def _extract_words(text: str) -> List[str]:
    """Extract meaningful words from text."""
    if not text:
        return []
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from',
        'are', 'was', 'not', 'but', 'all', 'can', 'will',
        'has', 'have', 'had', 'its', 'than', 'then', 'which',
        'would', 'could', 'should', 'into', 'over', 'under'
    }
    return [w for w in words if w not in stopwords]


def detect_operations_delta(
    ocr_before: str, ocr_after: str,
    change_description: str = ""
) -> List[Dict]:
    """Detect operations based on what CHANGED between frames."""
    detected = []
    words_before = set(_extract_words(ocr_before))
    words_after = set(_extract_words(ocr_after))
    added_words = words_after - words_before
    removed_words = words_before - words_after

    delta_text = ' '.join(added_words | removed_words)
    action_text = change_description.lower() if change_description else ""

    all_ops = {
        "Excel": EXCEL_OPERATIONS,
        "Web": WEB_OPERATIONS,
        "General": GENERAL_OPERATIONS,
    }

    for category, ops_dict in all_ops.items():
        for op_name, keywords in ops_dict.items():
            for kw in keywords:
                if kw in action_text:
                    detected.append({
                        "category": category,
                        "operation": op_name,
                        "keyword_matched": kw,
                        "display_name": format_operation_name(op_name),
                        "source": "vision_action",
                        "confidence": 0.9
                    })
                    break
                elif kw in delta_text.lower():
                    detected.append({
                        "category": category,
                        "operation": op_name,
                        "keyword_matched": kw,
                        "display_name": format_operation_name(op_name),
                        "source": "delta",
                        "confidence": 0.7
                    })
                    break

    # Deduplicate
    seen_ops = {}
    for op in detected:
        key = (op["category"], op["operation"])
        if key not in seen_ops or op["confidence"] > seen_ops[key]["confidence"]:
            seen_ops[key] = op
    return list(seen_ops.values())


def format_operation_name(op_name: str) -> str:
    """Convert operation key to display name."""
    display_map = {
        "vlookup": "VLOOKUP Formula", "hlookup": "HLOOKUP Formula",
        "filter": "Data Filter", "sort": "Data Sort",
        "pivot_table": "Pivot Table", "duplicate_removal": "Duplicate Removal",
        "formula": "Formula/Calculation", "copy_paste": "Copy & Paste",
        "find_replace": "Find & Replace", "chart": "Chart/Graph",
        "macro": "Macro Execution", "import_export": "Import/Export Data",
        "data_validation": "Data Validation",
        "login": "User Login/Authentication", "logout": "User Logout",
        "navigate": "Page Navigation", "search": "Search/Query",
        "form_fill": "Form Data Entry", "upload": "File Upload",
        "download": "File Download", "submit": "Form Submission",
        "select": "Selection", "modal_dialog": "Dialog Interaction",
        "open_application": "Application Launch",
        "close_application": "Application Close",
        "switch_window": "Window Switch", "email": "Email Operation",
        "file_operation": "File Management",
    }
    return display_map.get(op_name, op_name.replace('_', ' ').title())


def build_operation_context(operations: List[Dict]) -> str:
    """Build a context string from detected operations."""
    if not operations:
        return ""
    high_conf = [op for op in operations
                 if op.get("source") in ("delta", "vision_action")
                 and op.get("confidence", 0) >= 0.7]
    if not high_conf:
        return ""
    lines = ["Detected operations:"]
    for op in high_conf[:3]:
        lines.append(f"- {op['display_name']}")
    return "\n".join(lines)