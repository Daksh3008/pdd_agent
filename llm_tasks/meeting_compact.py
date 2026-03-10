# llm_tasks/meeting_compact.py

"""
Compact meeting transcript → PDD bundle extraction.

TWO consolidated LLM calls:
  Call 1: Document sections (purpose, overview, justification, as-is, to-be)
  Call 2: Process data (steps, detailed steps, inputs, interfaces, exceptions)

This avoids the single-giant-JSON problem where Gemini truncates or
malforms the response, causing silent parse failures.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from core.gemini_client import gemini_client
from core.config import config
from core.utils import safe_sample, timed, enforce_tone, redact_pii_text
from llm_tasks.system_prompts import get_system_prompt, TONE_RULES


# ============================================================
# JSON extraction helpers
# ============================================================

def _repair_json(text: str) -> str:
    """
    Attempt to repair common JSON issues from LLM output.
    Handles: trailing commas, unescaped newlines in strings,
    missing commas between keys, single quotes, unquoted keys,
    truncated responses, unescaped quotes inside values.
    """
    if not text:
        return text

    # Remove any BOM or invisible characters
    text = text.strip('\ufeff\u200b\u200c\u200d')

    # Fix single quotes used as string delimiters
    text = re.sub(r"(?<=[{,\[])\s*'([^']+)'\s*:", r' "\1":', text)
    text = re.sub(r":\s*'([^']*)'(?=\s*[,}\]])", r': "\1"', text)

    # Fix unquoted keys: { key: "value" } -> { "key": "value" }
    text = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', text)

    # Fix trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*\]', ']', text)

    # Fix missing commas between key-value pairs
    text = re.sub(r'"\s*\n\s*"(?=\w+"\s*:)', '",\n"', text)

    # Fix missing commas after closing braces/brackets before new keys
    text = re.sub(r'}\s*\n\s*"(?=\w+"\s*:)', '},\n"', text)
    text = re.sub(r']\s*\n\s*"(?=\w+"\s*:)', '],\n"', text)

    # Fix unescaped newlines inside string values
    text = _escape_newlines_in_strings(text)

    # Fix unescaped quotes inside string values
    text = _fix_inner_quotes(text)

    # Handle truncated JSON (missing closing braces/brackets)
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')

    if open_braces > 0 or open_brackets > 0:
        # Check if we're inside an unterminated string
        in_string = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == '\\' and in_string:
                i += 2
                continue
            if ch == '"':
                in_string = not in_string
            i += 1

        if in_string:
            text = text + '"'

        # Re-clean trailing commas after adding quote
        text = re.sub(r',\s*$', '', text.rstrip())

        for _ in range(max(0, open_brackets)):
            text = text.rstrip().rstrip(',') + ']'
        for _ in range(max(0, open_braces)):
            text = text.rstrip().rstrip(',') + '}'

    return text


def _escape_newlines_in_strings(text: str) -> str:
    """Escape literal newlines inside JSON string values."""
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]

        if ch == '\\' and in_string and i + 1 < len(text):
            result.append(ch)
            result.append(text[i + 1])
            i += 2
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue

        if ch == '\n' and in_string:
            result.append('\\n')
            i += 1
            continue

        if ch == '\r' and in_string:
            i += 1
            continue

        if ch == '\t' and in_string:
            result.append('\\t')
            i += 1
            continue

        result.append(ch)
        i += 1

    return ''.join(result)


def _fix_inner_quotes(text: str) -> str:
    """
    Fix unescaped double quotes inside JSON string values.
    
    Strategy: Walk through character by character, tracking whether we're
    inside a JSON string. When we encounter a quote inside a string that
    doesn't look like a JSON delimiter (not followed by :, ,, }, ]), 
    escape it.
    """
    # First try parsing as-is
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    result = []
    i = 0
    in_string = False
    string_start = -1

    while i < len(text):
        ch = text[i]

        # Handle escape sequences inside strings
        if ch == '\\' and in_string and i + 1 < len(text):
            result.append(ch)
            result.append(text[i + 1])
            i += 2
            continue

        if ch == '"':
            if not in_string:
                # Opening a string
                in_string = True
                string_start = i
                result.append(ch)
                i += 1
                continue
            else:
                # Could be closing the string OR an unescaped inner quote
                # Look ahead to determine
                after = text[i + 1:].lstrip() if i + 1 < len(text) else ''

                is_closing = False
                if not after:
                    is_closing = True
                elif after[0] in (',', '}', ']', ':'):
                    is_closing = True
                elif after.startswith('\\n'):
                    # This is inside the string value, probably an inner quote
                    is_closing = False
                elif re.match(r'^[,}\]:]', after):
                    is_closing = True

                if is_closing:
                    in_string = False
                    result.append(ch)
                else:
                    # Inner quote — escape it
                    result.append('\\"')

                i += 1
                continue

        result.append(ch)
        i += 1

    return ''.join(result)


def _extract_json_object(text: str) -> Optional[str]:
    """Extract the first JSON object from a response, with repair attempts."""
    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Try direct parse first
    if text.startswith("{"):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

    # Extract JSON object using brace matching
    json_text = _extract_balanced_braces(text)

    if json_text:
        # Try parsing as-is
        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass

        # Try repair
        repaired = _repair_json(json_text)
        try:
            json.loads(repaired)
            print("    [JSON] Repaired malformed JSON successfully")
            return repaired
        except json.JSONDecodeError as e:
            print(f"    [JSON] Repair attempt 1 failed: {e}")

            # Try aggressive repair: re-extract values as raw strings
            aggressive = _aggressive_json_repair(json_text)
            if aggressive:
                try:
                    json.loads(aggressive)
                    print("    [JSON] Aggressive repair succeeded")
                    return aggressive
                except json.JSONDecodeError as e2:
                    print(f"    [JSON] Aggressive repair failed: {e2}")

            # Last resort: truncation
            truncated = _truncate_to_valid_json(repaired)
            if truncated:
                print("    [JSON] Recovered partial JSON via truncation")
                return truncated

    # Fallback: regex extraction
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        repaired = _repair_json(candidate)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            aggressive = _aggressive_json_repair(candidate)
            if aggressive:
                try:
                    json.loads(aggressive)
                    return aggressive
                except json.JSONDecodeError:
                    pass

    return None


def _aggressive_json_repair(text: str) -> Optional[str]:
    """
    Aggressive JSON repair: extract key-value pairs using regex patterns
    and reconstruct a clean JSON object. Handles cases where inner quotes,
    newlines, and special characters completely break standard parsing.
    """
    if not text:
        return None

    # Try to extract each top-level key and its value
    # Pattern matches "key": followed by either a string, array, or object value
    
    # First, identify all top-level keys
    key_pattern = re.compile(r'"(\w+)"\s*:\s*', re.DOTALL)
    keys_found = [(m.group(1), m.start(), m.end()) for m in key_pattern.finditer(text)]
    
    if not keys_found:
        return None

    extracted = {}
    
    for idx, (key, key_start, val_start) in enumerate(keys_found):
        # Determine where this value ends (next top-level key or end of object)
        if idx + 1 < len(keys_found):
            # Value ends just before the next key's quote
            # But we need to find the comma before the next key
            next_key_start = keys_found[idx + 1][1]
            raw_value = text[val_start:next_key_start].strip()
            # Remove trailing comma
            raw_value = raw_value.rstrip().rstrip(',').strip()
        else:
            # Last key - value goes to end of object
            raw_value = text[val_start:].strip()
            # Remove trailing braces
            raw_value = raw_value.rstrip().rstrip('}').rstrip(',').strip()

        # Parse the value based on what it starts with
        parsed_value = _parse_raw_value(raw_value, key)
        if parsed_value is not None:
            extracted[key] = parsed_value

    if not extracted:
        return None

    try:
        result = json.dumps(extracted, ensure_ascii=False)
        json.loads(result)  # Verify it's valid
        return result
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _parse_raw_value(raw: str, key: str) -> Any:
    """Parse a raw JSON value string, handling malformed content."""
    if not raw:
        return "" if key not in ('entities',) else {}

    raw = raw.strip()

    # Try direct JSON parse first
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Array value: [...] 
    if raw.startswith('['):
        return _parse_raw_array(raw)

    # Object value: {...}
    if raw.startswith('{'):
        repaired = _repair_json(raw)
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            return {}

    # String value: "..."
    if raw.startswith('"'):
        return _parse_raw_string(raw)

    # Unquoted value - treat as string
    return raw.strip('"').strip()


def _parse_raw_array(raw: str) -> List:
    """Parse a raw array value, handling malformed elements."""
    # Try direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try repair then parse
    repaired = _repair_json(raw)
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass

    # Manual extraction: find string elements
    items = []
    
    # Check if array contains objects (dicts) or strings
    if '{' in raw:
        # Array of objects - extract each {...} block
        obj_pattern = re.compile(r'\{([^{}]*)\}')
        for m in obj_pattern.finditer(raw):
            obj_text = '{' + m.group(1) + '}'
            repaired_obj = _repair_json(obj_text)
            try:
                items.append(json.loads(repaired_obj))
            except (json.JSONDecodeError, ValueError):
                # Try to extract key-value pairs manually
                kv_pattern = re.compile(r'"(\w+)"\s*:\s*"([^"]*)"')
                obj_dict = {}
                for kv in kv_pattern.finditer(m.group(1)):
                    obj_dict[kv.group(1)] = kv.group(2)
                if obj_dict:
                    items.append(obj_dict)
    else:
        # Array of strings - extract quoted strings
        str_pattern = re.compile(r'"((?:[^"\\]|\\.)*)"')
        for m in str_pattern.finditer(raw):
            val = m.group(1).strip()
            if val and len(val) > 3:
                items.append(val)

    return items


def _parse_raw_string(raw: str) -> str:
    """Parse a raw string value, handling unescaped characters."""
    # Try direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip outer quotes and clean up
    if raw.startswith('"') and raw.endswith('"'):
        inner = raw[1:-1]
    elif raw.startswith('"'):
        # Missing closing quote
        inner = raw[1:]
    else:
        inner = raw

    # Escape problematic characters
    inner = inner.replace('\\', '\\\\')  # Escape backslashes first
    inner = inner.replace('\n', '\\n')
    inner = inner.replace('\r', '')
    inner = inner.replace('\t', '\\t')
    
    # Handle unescaped inner quotes - replace with escaped
    # But we already stripped outer quotes, so all remaining quotes are inner
    inner = inner.replace('"', '\\"')

    try:
        return json.loads(f'"{inner}"')
    except (json.JSONDecodeError, ValueError):
        # Last resort: return cleaned raw text
        return inner.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')


def _extract_balanced_braces(text: str) -> Optional[str]:
    """Extract a balanced JSON object by tracking brace depth."""
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    i = start

    while i < len(text):
        ch = text[i]

        if ch == '\\' and in_string:
            i += 2
            continue

        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        i += 1

    # If unbalanced, return what we have plus closing braces
    if depth > 0:
        fragment = text[start:]
        fragment = fragment.rstrip().rstrip(',')
        fragment += '}' * depth
        return fragment

    return None


def _truncate_to_valid_json(text: str) -> Optional[str]:
    """
    Progressively truncate JSON to find the largest valid subset.
    Useful when Gemini truncates the response mid-value.
    """
    if not text:
        return None

    # Strategy 1: Try removing content from the end to find valid JSON
    for end_pattern in [
        r',\s*"[^"]*"\s*:\s*(?:"(?:[^"\\]|\\.)*$)',  # Unterminated string value
        r',\s*"[^"]*"\s*:\s*\[(?:[^\]]*$)',  # Unterminated array
        r',\s*"[^"]*"\s*:\s*\{(?:[^}]*$)',  # Unterminated object
        r',\s*"[^"]*"\s*:\s*(?:"[^"]*"|[\[\{]).*$',  # Last incomplete k-v pair
        r',\s*"[^"]*"\s*:.*$',  # Any trailing incomplete pair
    ]:
        truncated = re.sub(end_pattern, '', text, flags=re.DOTALL)
        if truncated != text and len(truncated) > 10:
            repaired = _repair_json(truncated)
            try:
                json.loads(repaired)
                return repaired
            except json.JSONDecodeError:
                continue

    # Strategy 2: Find the last complete key-value pair by looking for
    # the last valid comma-separated boundary
    # Find positions of all top-level commas (not inside strings/nested)
    comma_positions = _find_top_level_commas(text)
    
    # Try truncating at each comma from the end
    for pos in reversed(comma_positions):
        candidate = text[:pos].rstrip() 
        # Close any open structures
        repaired = _repair_json(candidate)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            continue

    return None


def _find_top_level_commas(text: str) -> List[int]:
    """Find positions of commas at the top level of a JSON object (depth 1)."""
    positions = []
    depth = 0
    in_string = False
    i = 0

    while i < len(text):
        ch = text[i]

        if ch == '\\' and in_string:
            i += 2
            continue

        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch in ('{', '['):
                depth += 1
            elif ch in ('}', ']'):
                depth -= 1
            elif ch == ',' and depth == 1:
                positions.append(i)

        i += 1

    return positions


def _coerce_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        lines = [l.strip("•*- \t") for l in x.splitlines() if l.strip()]
        return [l for l in lines if len(l) > 5]
    return [str(x).strip()] if str(x).strip() else []


def _coerce_list_dict(x: Any, keys: Tuple[str, str]) -> List[Dict[str, str]]:
    k1, k2 = keys
    out: List[Dict[str, str]] = []
    if isinstance(x, list):
        for row in x:
            if isinstance(row, dict):
                v1 = str(row.get(k1, "")).strip()
                v2 = str(row.get(k2, "")).strip()
                if v1:
                    out.append({k1: v1, k2: v2})
            elif isinstance(row, str) and "|" in row:
                a, b = row.split("|", 1)
                a, b = a.strip(), b.strip()
                if a:
                    out.append({k1: a, k2: b})
    elif isinstance(x, str):
        for line in x.splitlines():
            if "|" in line:
                a, b = line.split("|", 1)
                a, b = a.strip(), b.strip()
                if a:
                    out.append({k1: a, k2: b})
    return out


def _apply_tone_and_redaction(text: str) -> str:
    """Apply tone enforcement and PII redaction."""
    if not text:
        return text
    text = enforce_tone(text)
    text = redact_pii_text(text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    return text


def _strip_markdown(text: str) -> str:
    """Remove all markdown formatting from text."""
    if not text:
        return text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*]\s+', '- ', text, flags=re.MULTILINE)
    return text


# ============================================================
# Call 1: Document Sections
# ============================================================

def _generate_document_sections(
    transcript: str,
    project_name_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    LLM Call 1: Extract project name + all narrative document sections.
    Returns dict with project_name, entities, and document sections.
    """
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)
    doc_type = config.document.document_type
    doc_full = config.document.document_type_full

    project_hint = f'Project name hint: "{project_name_hint}"' if project_name_hint else \
        "Derive a short descriptive project name (max 6 words) from the main process discussed."

    prompt = f"""You are a senior Business Analyst creating a {doc_full} ({doc_type}).

{project_hint}

TASK: Produce the following sections as a JSON object.

SECTION REQUIREMENTS:

1. "project_name": Short descriptive name (max 6 words).

2. "entities": Object with arrays: "companies", "applications", "systems", "departments".
   ONLY names explicitly mentioned in the transcript. Use empty array if none found.

3. "purpose": 2-3 paragraphs (150-250 words).
   - What this document defines (objectives, scope, requirements).
   - Intended audience (developers, QA, business stakeholders).
   - Scope of the automation being documented.

4. "overview": 1 paragraph + 5-7 bullet points.
   - Opening paragraph stating the primary business objective.
   - Bullet points listing specific outcomes. Each starts with "The system".

5. "justification": 1 paragraph + 5-7 numbered benefits.
   - Opening paragraph about operational value.
   - Numbered items with a title and description for each benefit.
   DO NOT use markdown bold (no ** markers). Use plain text only.

6. "as_is": 3-4 paragraphs (200-300 words).
   - Current manual process description.
   - Pain points, inefficiencies, risks.
   - End with "Business Challenges:" followed by 4-6 bullet points.

7. "to_be": 3-4 paragraphs (200-300 words).
   - Future automated process, written in present tense as if operational.
   - Cover trigger, data handling, validation, processing, reporting, logging.

CRITICAL JSON RULES:
- Use ONLY names from the transcript. Never invent names.
- NEVER mention the meeting, transcript, recording, or speakers.
- NEVER include personal names, emails, or phone numbers.
- Write in THIRD PERSON, PRESENT TENSE, ACTIVE VOICE.
- DO NOT use markdown formatting (no **, no ##, no __).
- IMPORTANT: All string values MUST be on a SINGLE LINE. Use \\n for paragraph breaks within strings.
- IMPORTANT: Escape all double quotes inside string values with backslash: \\"
- Output STRICT valid JSON only. No markdown fences, no commentary before or after.

JSON FORMAT:
{{
  "project_name": "string",
  "entities": {{"companies": [], "applications": [], "systems": [], "departments": []}},
  "purpose": "paragraph 1\\n\\nparagraph 2\\n\\nparagraph 3",
  "overview": "paragraph\\n\\n- bullet 1\\n- bullet 2",
  "justification": "paragraph\\n\\n1. benefit title - description\\n2. benefit title - description",
  "as_is": "paragraph 1\\n\\nparagraph 2\\n\\nBusiness Challenges:\\n- challenge 1\\n- challenge 2",
  "to_be": "paragraph 1\\n\\nparagraph 2\\n\\nparagraph 3"
}}

TRANSCRIPT:
{sample}
"""

    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        temperature=0.3,
        max_output_tokens=config.llm.max_output_tokens,
        call_name="DocBundle_Sections",
        max_retries=3
    )

    result = {
        "project_name": project_name_hint or "Process Automation Project",
        "entities": {"companies": [], "applications": [], "systems": [], "departments": []},
        "document": {"purpose": "", "overview": "", "justification": "", "as_is": "", "to_be": ""}
    }

    json_text = _extract_json_object(resp or "")
    if json_text:
        try:
            data = json.loads(json_text)

            pn = str(data.get("project_name", "")).strip()
            if pn and len(pn) > 3:
                result["project_name"] = pn

            ent = data.get("entities", {})
            if isinstance(ent, dict):
                for key in ["companies", "applications", "systems", "departments"]:
                    items = ent.get(key, [])
                    if isinstance(items, list):
                        result["entities"][key] = [str(i).strip() for i in items if str(i).strip()]

            for key in ["purpose", "overview", "justification", "as_is", "to_be"]:
                val = str(data.get(key, "")).strip()
                if val:
                    val = _strip_markdown(val)
                    val = _apply_tone_and_redaction(val)
                    result["document"][key] = val

            print(f"    [DocBundle_Sections] Parsed successfully: "
                  f"{sum(1 for v in result['document'].values() if v)}/5 sections")

        except json.JSONDecodeError as e:
            print(f"    [DocBundle_Sections] JSON parse failed after all repair attempts: {e}")
            result["document"] = _fallback_parse_sections(resp)
        except Exception as e:
            print(f"    [DocBundle_Sections] Error: {type(e).__name__}: {e}")
    else:
        print("    [DocBundle_Sections] No JSON found in response, using fallback parser")
        result["document"] = _fallback_parse_sections(resp)

    # Ensure all sections have content
    pn = result["project_name"]
    doc = result["document"]
    doc_full_name = config.document.document_type_full

    if not doc["purpose"] or len(doc["purpose"]) < 50:
        doc["purpose"] = (
            f"This {doc_full_name} defines the objectives, scope, and detailed "
            f"requirements for the {pn} automation initiative. The document serves "
            f"as the primary reference for development teams, quality assurance personnel, "
            f"and business stakeholders involved in the design, implementation, and "
            f"validation of the automated solution.\n\n"
            f"The scope encompasses the complete end-to-end process flow, including "
            f"system interactions, data handling procedures, validation rules, exception "
            f"handling scenarios, and expected outputs. The document captures both the "
            f"current manual state and the target automated state of the {pn} process.\n\n"
            f"This document applies to all phases of the automation lifecycle, from "
            f"initial development through testing, deployment, and ongoing maintenance."
        )

    if not doc["overview"] or len(doc["overview"]) < 50:
        doc["overview"] = (
            f"The primary objective of the {pn} automation is to streamline "
            f"and standardize the existing manual process, ensuring consistency, "
            f"accuracy, and compliance with established business rules.\n\n"
            f"- The system automates repetitive manual tasks to reduce processing time.\n"
            f"- The system enforces standardized validation rules across all records.\n"
            f"- The system generates comprehensive audit trails for compliance tracking.\n"
            f"- The system reduces human error through automated data handling.\n"
            f"- The system provides real-time status reporting and exception notifications."
        )

    if not doc["justification"] or len(doc["justification"]) < 50:
        doc["justification"] = (
            f"Automating the {pn} process delivers measurable operational value "
            f"by reducing manual effort, improving accuracy, and ensuring consistent "
            f"execution across all processing cycles.\n\n"
            f"1. Reduced Processing Time: The system completes the end-to-end process "
            f"in minutes compared to hours of manual effort.\n"
            f"2. Improved Accuracy: The system eliminates manual data entry errors "
            f"through automated validation and processing.\n"
            f"3. Enhanced Compliance: The system maintains detailed audit logs for "
            f"regulatory and governance requirements.\n"
            f"4. Consistent Execution: The system applies identical business rules "
            f"to every record, eliminating subjective variation.\n"
            f"5. Scalability: The system handles increased volume without additional "
            f"manual resources."
        )

    if not doc["as_is"] or len(doc["as_is"]) < 50:
        doc["as_is"] = (
            f"The current {pn} process relies on manual execution by trained operators. "
            f"The operator logs into the required applications, navigates to the relevant "
            f"modules, and performs data extraction, validation, and processing tasks "
            f"by hand.\n\n"
            f"The manual process requires significant time investment for each processing "
            f"cycle. The operator must review individual records, cross-reference data "
            f"across multiple sources, and manually update system records based on the "
            f"validation results.\n\n"
            f"Business Challenges:\n"
            f"- High processing time due to manual record-by-record handling.\n"
            f"- Risk of human error in data entry and validation.\n"
            f"- Inconsistent application of business rules across operators.\n"
            f"- Limited audit trail for compliance verification.\n"
            f"- Difficulty scaling with increased processing volume."
        )

    if not doc["to_be"] or len(doc["to_be"]) < 50:
        doc["to_be"] = (
            f"The {pn} automation executes the end-to-end process through a structured "
            f"sequence of system actions. The automation initiates upon a configured "
            f"trigger, establishes connections to required applications, and processes "
            f"data according to defined business rules.\n\n"
            f"The system validates each record against the established criteria, performs "
            f"the required actions for compliant records, and logs exceptions for "
            f"non-compliant entries. The automation handles error scenarios through "
            f"configured retry logic and exception routing.\n\n"
            f"Upon completion, the system generates a comprehensive execution report "
            f"detailing processed records, exceptions encountered, and overall "
            f"processing statistics. The system updates the process status and "
            f"archives execution logs for audit purposes."
        )

    return result


def _fallback_parse_sections(raw_text: str) -> Dict[str, str]:
    """Parse sections from non-JSON LLM response as fallback."""
    sections = {"purpose": "", "overview": "", "justification": "", "as_is": "", "to_be": ""}
    if not raw_text:
        return sections

    raw_text = _strip_markdown(raw_text)

    section_patterns = {
        "purpose": [
            r'(?:"?purpose"?\s*[:=]\s*"?)(.*?)(?="?\s*,?\s*"?(?:overview|justification|as.is|to.be|entities)"?\s*[:=]|\Z)',
            r'(?:purpose|document purpose)[:\s]*(.*?)(?=overview|justification|as.is|to.be|\Z)',
        ],
        "overview": [
            r'(?:"?overview"?\s*[:=]\s*"?)(.*?)(?="?\s*,?\s*"?(?:justification|as.is|to.be)"?\s*[:=]|\Z)',
            r'(?:overview|objective)[:\s]*(.*?)(?=justification|as.is|to.be|\Z)',
        ],
        "justification": [
            r'(?:"?justification"?\s*[:=]\s*"?)(.*?)(?="?\s*,?\s*"?(?:as.is|to.be)"?\s*[:=]|\Z)',
            r'(?:justification|business justification)[:\s]*(.*?)(?=as.is|to.be|\Z)',
        ],
        "as_is": [
            r'(?:"?as.is"?\s*[:=]\s*"?)(.*?)(?="?\s*,?\s*"?(?:to.be)"?\s*[:=]|\Z)',
            r'(?:as.is|current state|current process)[:\s]*(.*?)(?=to.be|future|\Z)',
        ],
        "to_be": [
            r'(?:"?to.be"?\s*[:=]\s*"?)(.*?)(?="?\s*}|\Z)',
            r'(?:to.be|future state|automated process)[:\s]*(.*?)$',
        ],
    }

    for key, patterns in section_patterns.items():
        for pattern in patterns:
            m = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
            if m and len(m.group(1).strip()) > 30:
                val = m.group(1).strip().strip('"').strip()
                val = val.replace('\\n', '\n')
                sections[key] = _apply_tone_and_redaction(val)
                break

    found = sum(1 for v in sections.values() if v)
    if found > 0:
        print(f"    [DocBundle_Sections] Fallback parser recovered {found}/5 sections")

    return sections


# ============================================================
# Call 2: Process Data (steps, requirements)
# ============================================================

def _generate_process_data(
    transcript: str,
    project_name: str,
    entities: Dict
) -> Dict[str, Any]:
    """
    LLM Call 2: Extract process steps, detailed steps, and all requirements tables.
    """
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)

    apps_hint = ""
    if entities.get("applications"):
        apps_hint = f"Applications mentioned: {', '.join(entities['applications'])}"

    prompt = f"""You are a senior Business Analyst extracting automation process data.

Project: "{project_name}"
{apps_hint}

TASK: Extract ALL process steps and requirements from this transcript.

OUTPUT STRICT JSON with this exact structure:

{{
  "process_steps": [
    "Step description starting with a verb (8-15 items)"
  ],
  "detailed_steps": [
    "Detailed screen-level action description (15-25 items)"
  ],
  "input_requirements": [
    {{"parameter": "Parameter Name", "description": "What it is and why needed"}}
  ],
  "interface_requirements": [
    {{"application": "App Name", "purpose": "Why the automation uses it"}}
  ],
  "exception_handling": [
    {{"exception": "Error scenario", "handling": "What the system does"}}
  ]
}}

DETAILED INSTRUCTIONS:

process_steps (8-15 items):
- High-level automation steps, each starting with an action verb.
- Examples: "Connect to the application using credentials",
  "Navigate to the data module", "Extract records from source",
  "Validate records against criteria", "Generate execution report"

detailed_steps (15-25 items):
- Screen-level actions for PDD Section 2.4.
- Each describes ONE specific action: login, navigate, click, enter data,
  download, process, validate, export, report, logout.
- Each starts with "The system" or "The automation".
- Examples: "The system opens the target application login page",
  "The system enters the configured credentials and clicks Sign In",
  "The system navigates to the Reports tab in the main menu"

input_requirements (5-10 items):
- Parameters needed: credentials, URLs, file paths, config values.
- NEVER include actual passwords or personal data.

interface_requirements (3-8 items):
- Applications and systems the automation interacts with.
- ONLY names from the transcript.

exception_handling (6-10 items):
- Error scenarios with system response.
- Include: login failure, timeout, missing data, validation error, system crash.
- Handling must be in third person: "The system retries...", "The system logs..."

CRITICAL JSON RULES:
- Use ONLY application names from the transcript.
- NEVER mention the meeting, transcript, or speakers.
- NEVER include personal names, emails, or phone numbers.
- Third person, present tense, active voice.
- DO NOT use markdown formatting.
- IMPORTANT: Keep all string values on a SINGLE LINE. No literal line breaks inside strings.
- IMPORTANT: Escape any double quotes inside strings with backslash: \\"
- Output STRICT valid JSON only. No text before or after the JSON object.

TRANSCRIPT:
{sample}
"""

    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        temperature=0.3,
        max_output_tokens=config.llm.max_output_tokens,
        call_name="DocBundle_ProcessData",
        max_retries=3
    )

    result = {
        "process_steps": [],
        "detailed_steps": [],
        "input_requirements": [],
        "interface_requirements": [],
        "exception_handling": [],
    }

    json_text = _extract_json_object(resp or "")
    if json_text:
        try:
            data = json.loads(json_text)

            result["process_steps"] = _coerce_list_str(data.get("process_steps"))
            result["detailed_steps"] = _coerce_list_str(data.get("detailed_steps"))
            result["input_requirements"] = _coerce_list_dict(
                data.get("input_requirements"), ("parameter", "description")
            )
            result["interface_requirements"] = _coerce_list_dict(
                data.get("interface_requirements"), ("application", "purpose")
            )
            result["exception_handling"] = _coerce_list_dict(
                data.get("exception_handling"), ("exception", "handling")
            )

            print(f"    [DocBundle_ProcessData] Parsed: "
                  f"{len(result['process_steps'])} steps, "
                  f"{len(result['detailed_steps'])} detailed, "
                  f"{len(result['input_requirements'])} inputs, "
                  f"{len(result['interface_requirements'])} interfaces, "
                  f"{len(result['exception_handling'])} exceptions")

        except json.JSONDecodeError as e:
            print(f"    [DocBundle_ProcessData] JSON parse failed after all repairs: {e}")
            result = _fallback_parse_process_data(resp)
        except Exception as e:
            print(f"    [DocBundle_ProcessData] Error: {type(e).__name__}: {e}")
    else:
        print("    [DocBundle_ProcessData] No JSON found, using fallback parser")
        result = _fallback_parse_process_data(resp)

    # Redact PII from all text
    result["process_steps"] = [redact_pii_text(s) for s in result["process_steps"]]
    result["detailed_steps"] = [redact_pii_text(s) for s in result["detailed_steps"]]

    # Apply fallbacks for empty sections
    if not result["process_steps"]:
        apps = ', '.join(entities.get('applications', [])) or 'the target application'
        result["process_steps"] = [
            f"The system connects to {apps} using authorized credentials.",
            f"The system navigates to the relevant processing module within {apps}.",
            "The system extracts the required data from the configured source.",
            "The system filters records based on the defined business criteria.",
            "The system validates each record against the established rules.",
            f"The system performs the required processing actions in {apps}.",
            "The system captures the updated status for each processed item.",
            "The system generates a comprehensive execution report.",
            "The system updates the process status and logs execution details.",
        ]
        print("    [DocBundle_ProcessData] Using fallback process steps")

    if not result["detailed_steps"]:
        result["detailed_steps"] = [
            "The system opens the target application and navigates to the login page.",
            "The system enters the configured credentials and authenticates.",
            "The system navigates to the main processing module.",
            "The system selects the appropriate data source or report.",
            "The system applies the configured filters and criteria.",
            "The system extracts the matching records from the source.",
            "The system validates each record against the defined business rules.",
            "The system flags records that fail validation checks.",
            "The system processes each validated record according to the workflow.",
            "The system updates the record status after processing.",
            "The system captures processing results and timestamps.",
            "The system generates a summary report of all processed records.",
            "The system exports the report to the configured output location.",
            "The system logs all execution details for audit purposes.",
            "The system closes the application and terminates the session.",
        ]
        print("    [DocBundle_ProcessData] Using fallback detailed steps")

    if not result["input_requirements"]:
        result["input_requirements"] = [
            {"parameter": "Application URL", "description": "Web address of the target application portal."},
            {"parameter": "User Credentials", "description": "Username and password for secure application access."},
            {"parameter": "Input Data Source", "description": "File path or database connection for source data."},
            {"parameter": "Processing Criteria", "description": "Business rules and filters for record selection."},
            {"parameter": "Output Location", "description": "File path or destination for generated reports."},
        ]

    if not result["interface_requirements"]:
        result["interface_requirements"] = [
            {"application": "Target Application", "purpose": "Primary application for process execution."},
        ]

    if not result["exception_handling"]:
        result["exception_handling"] = [
            {"exception": "Application Login Failure",
             "handling": "The system retries login up to 3 times. If authentication fails, the system stops and sends a notification."},
            {"exception": "Element Not Found",
             "handling": "The system waits up to 30 seconds. If not found, the system captures a screenshot and logs the error."},
            {"exception": "Data Validation Error",
             "handling": "The system flags the invalid record, logs details, and continues processing remaining items."},
            {"exception": "Application Timeout",
             "handling": "The system retries the operation after a configured wait period and logs the timeout event."},
            {"exception": "System Exception",
             "handling": "The system captures error details, saves a diagnostic screenshot, and terminates gracefully."},
        ]

    return result


def _fallback_parse_process_data(raw_text: str) -> Dict[str, Any]:
    """Parse process data from non-JSON response."""
    result = {
        "process_steps": [],
        "detailed_steps": [],
        "input_requirements": [],
        "interface_requirements": [],
        "exception_handling": [],
    }
    if not raw_text:
        return result

    raw_text = _strip_markdown(raw_text)

    lines = raw_text.split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        lower = line.lower()

        # Detect section headers
        if 'process_step' in lower or 'process step' in lower:
            current_section = 'process_steps'
            continue
        elif 'detailed_step' in lower or 'detailed step' in lower:
            current_section = 'detailed_steps'
            continue
        elif 'input_req' in lower or 'input req' in lower or 'input_param' in lower:
            current_section = 'input_requirements'
            continue
        elif 'interface_req' in lower or 'interface req' in lower:
            current_section = 'interface_requirements'
            continue
        elif 'exception' in lower and ('handl' in lower or ':' in line):
            current_section = 'exception_handling'
            continue

        if not line or not current_section:
            continue

        # Parse numbered/bulleted items
        cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line).strip()
        cleaned = re.sub(r'^[-•*]\s*', '', cleaned).strip()
        cleaned = cleaned.strip('"')

        if len(cleaned) < 10:
            continue

        if current_section in ('process_steps', 'detailed_steps'):
            result[current_section].append(cleaned)
        elif current_section == 'input_requirements':
            if '|' in cleaned:
                parts = cleaned.split('|', 1)
                result[current_section].append({
                    "parameter": parts[0].strip(),
                    "description": parts[1].strip() if len(parts) > 1 else ""
                })
            elif ':' in cleaned:
                parts = cleaned.split(':', 1)
                result[current_section].append({
                    "parameter": parts[0].strip(),
                    "description": parts[1].strip() if len(parts) > 1 else ""
                })
        elif current_section == 'interface_requirements':
            if '|' in cleaned:
                parts = cleaned.split('|', 1)
                result[current_section].append({
                    "application": parts[0].strip(),
                    "purpose": parts[1].strip() if len(parts) > 1 else ""
                })
        elif current_section == 'exception_handling':
            if '|' in cleaned:
                parts = cleaned.split('|', 1)
                result[current_section].append({
                    "exception": parts[0].strip(),
                    "handling": parts[1].strip() if len(parts) > 1 else ""
                })

    found_sections = sum(1 for v in result.values() if v)
    if found_sections > 0:
        print(f"    [DocBundle_ProcessData] Fallback parser recovered {found_sections}/5 sections")

    return result


# ============================================================
# Public API
# ============================================================

def generate_doc_bundle_from_transcript(
    transcript: str,
    project_name_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Two consolidated LLM calls to extract ALL PDD content.
    Call 1: Document narrative sections
    Call 2: Process steps and requirements tables
    """
    start = time.time()

    # Call 1: Sections
    print("    [DocBundle] Call 1/2: Document sections...")
    sections_result = _generate_document_sections(transcript, project_name_hint)

    project_name = sections_result["project_name"]
    entities = sections_result["entities"]

    # Call 2: Process data
    print("    [DocBundle] Call 2/2: Process steps & requirements...")
    process_result = _generate_process_data(transcript, project_name, entities)

    # Combine
    result = {
        "project_name": project_name,
        "entities": entities,
        "document": sections_result["document"],
        "process": {
            "process_steps": process_result["process_steps"],
            "detailed_steps": process_result["detailed_steps"],
        },
        "requirements": {
            "input_requirements": process_result["input_requirements"],
            "interface_requirements": process_result["interface_requirements"],
            "exception_handling": process_result["exception_handling"],
        }
    }

    timed("DocBundle_Combined", start)
    return result


def generate_dot_from_transcript(
    transcript: str,
    project_name: str,
    process_steps: Optional[List[str]] = None
) -> str:
    """
    Single LLM call to generate DOT flowchart code.
    Enforces short labels (max 5 words per node).
    """
    start = time.time()
    sample = safe_sample(transcript, max_len=config.llm.max_sample_text)

    steps_block = ""
    if process_steps:
        steps_block = "PROCESS STEPS (use these as nodes):\n" + "\n".join(
            f"{i+1}. {s[:80]}" for i, s in enumerate(process_steps[:18])
        )

    max_words = config.flowchart.max_label_words

    prompt = f"""Generate a Graphviz DOT flowchart for this automation process.

Process: "{project_name}"

{steps_block}

STRICT FORMATTING RULES:
1. Output ONLY valid DOT code. No markdown, no explanation, no code fences.
2. Must start with: digraph ProcessFlow {{
3. Must include Start (oval, green) and End (oval, red) nodes.
4. Use rankdir=TB.
5. Every node label MUST be MAX {max_words} WORDS. Short verb+object phrases only.
   GOOD: "Login to Portal", "Validate Records", "Export Report"
   BAD: "The system validates each record against the defined criteria"
6. Use diamond shape for decisions, box for steps.
7. Colors: process=lightblue, decision=gold, start=lightgreen, end=lightcoral.
8. All nodes must have style=filled.

TRANSCRIPT (context only):
{sample[:3000]}
"""

    resp = gemini_client.generate(
        prompt=prompt,
        system_prompt=get_system_prompt(),
        temperature=0.2,
        call_name="DOT_FromTranscript",
        max_retries=3
    )

    dot = (resp or "").strip()
    dot = re.sub(r'^```(?:dot|graphviz)?\s*', '', dot)
    dot = re.sub(r'\s*```$', '', dot)
    dot = dot.strip()

    m = re.search(r"(digraph\s+\w*\s*\{.*\})", dot, re.DOTALL)
    if m:
        dot = m.group(1).strip()

    dot = _enforce_short_labels(dot, max_words)

    timed("DOT_FromTranscript", start)
    return dot


def _enforce_short_labels(dot_code: str, max_words: int = 5) -> str:
    """Post-process DOT code to enforce short labels."""
    if not dot_code:
        return dot_code

    def _shorten(match):
        prefix = match.group(1)
        label = match.group(2)
        suffix = match.group(3)

        label = re.sub(
            r'\b(the|a|an|to|of|for|in|on|at|by|with|and|is|are|was|were|been|being)\b',
            ' ', label, flags=re.IGNORECASE
        )
        label = re.sub(r'\s+', ' ', label).strip()

        words = label.split()
        if len(words) > max_words:
            label = ' '.join(words[:max_words])

        if label:
            label = label[0].upper() + label[1:]

        return f'{prefix}"{label}"{suffix}'

    result = re.sub(
        r'(label\s*=\s*)"([^"]+)"(\s*[,\]])',
        _shorten, dot_code
    )
    return result