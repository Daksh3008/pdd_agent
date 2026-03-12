# pdd_no_audio/frame_analysis/vision_describer.py

"""
Vision model (llama3.2-vision:11b) frame description with ROI cropping and smart prompts.
Now combines before/after images side-by-side to work around single-image limitation.
Enhanced with auth/login screen detection forces vision calls for auth transitions.
FIXED: Strict output formatting to prevent prompt leakage into document.
"""

import re
import cv2
import os
import numpy as np
from typing import List, Dict, Optional

from pdd_no_audio.clients.vision_llm import vision_client
from pdd_no_audio.utils import (
    clean_vision_response, build_operation_context, detect_auth_screen
)
from pdd_no_audio.config import llm_params


VISION_SYSTEM_PROMPT = """You are a senior business analyst analyzing screenshots from a business process demonstration.

CRITICAL RULES:
1. Output ONLY in the exact format specified in the prompt.
2. Do NOT include any headers like "Part 1", "Part 2", "Instructions", etc.
3. Do NOT explain your reasoning or ask questions.
4. Do NOT mention screenshots, recordings, frames, or demonstrations.
5. If you cannot determine something, make your best inference and state it factually.
6. Write in third person: "The user...", "The system...", "The screen shows..."
7. Be SPECIFIC: mention actual text labels, column headers, button names visible on screen.
8. For spreadsheets: identify functions (VLOOKUP, FILTER, SORT), column names, cell ranges.
9. For web apps: identify page names, buttons, fields, menu paths.
10. For login/auth screens: identify username fields, password fields, sign-in buttons, SSO options."""


def _crop_to_region(image_path: str, region: Dict, padding: int = 20) -> Optional[str]:
    """Crop image to region (with padding) and save to a temporary file."""
    if not region or not os.path.exists(image_path):
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    x = max(0, region.get("x", 0) - padding)
    y = max(0, region.get("y", 0) - padding)
    x2 = min(w, x + region.get("w", 100) + 2 * padding)
    y2 = min(h, y + region.get("h", 50) + 2 * padding)

    if x2 - x < 50 or y2 - y < 30:
        return None

    cropped = img[y:y2, x:x2]
    base, ext = os.path.splitext(image_path)
    cropped_path = f"{base}_crop{ext}"
    cv2.imwrite(cropped_path, cropped)
    return cropped_path


def _combine_images_side_by_side(img1_path: str, img2_path: str, max_height: int = 800) -> Optional[str]:
    """Combine two images side-by-side into a single image."""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            return None

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        target_height = min(max_height, max(h1, h2))
        if h1 != target_height:
            scale = target_height / h1
            img1 = cv2.resize(img1, (int(w1 * scale), target_height))
        if h2 != target_height:
            scale = target_height / h2
            img2 = cv2.resize(img2, (int(w2 * scale), target_height))

        combined = np.hstack((img1, img2))
        base, ext = os.path.splitext(img1_path)
        combined_path = f"{base}_combined{ext}"
        cv2.imwrite(combined_path, combined)
        return combined_path

    except Exception as e:
        print(f"    [Vision] Error combining images: {e}")
        return None


def _sanitize_vision_response(text: str) -> str:
    """Remove prompt instructions that leak into output."""
    if not text:
        return ""
    
    # Patterns that indicate prompt leakage - remove these completely
    patterns_to_remove = [
        # Part headers
        r'PART\s*\d+\s*[—\-:.]?\s*[A-Z\s]*:?\s*',
        r'Part\s*\d+\s*[—\-:.]?\s*[A-Za-z\s]*:?\s*',
        
        # Instruction headers
        r'INSTRUCTIONS?:.*?(?=\n[A-Z]|\n\n|$)',
        r'RULES?:.*?(?=\n[A-Z]|\n\n|$)',
        r'CRITICAL RULES?:.*?(?=\n[A-Z]|\n\n|$)',
        r'OUTPUT FORMAT:.*?(?=\n[A-Z]|\n\n|$)',
        r'STRICT OUTPUT FORMAT.*?(?=SCREEN:|ACTION:|$)',
        
        # Context headers from prompt
        r'BEFORE screen state:.*?(?=AFTER|ACTION|$)',
        r'AFTER screen state:.*?(?=ACTION|SCREEN|$)',
        r'Action observed:.*?(?=SCREEN|$)',
        r'Text changes detected.*?(?=SCREEN|ACTION|$)',
        r'Detected operations on this screen:.*?(?=SCREEN|ACTION|$)',
        r'Change type:.*?(?=SCREEN|ACTION|$)',
        
        # Prompt instruction fragments
        r'You are analyzing.*?(?=SCREEN|ACTION|The |$)',
        r'These two screenshots.*?(?=SCREEN|ACTION|The |$)',
        r'LEFT\s*=\s*BEFORE.*?(?=SCREEN|ACTION|$)',
        r'RIGHT\s*=\s*AFTER.*?(?=SCREEN|ACTION|$)',
        r'Return EXACTLY.*?(?=SCREEN|ACTION|$)',
        r'Do NOT include.*?(?=SCREEN|ACTION|The |$)',
        r'Do NOT explain.*?(?=SCREEN|ACTION|The |$)',
        r'Do NOT mention.*?(?=SCREEN|ACTION|The |$)',
        r'If you cannot.*?(?=SCREEN|ACTION|The |$)',
        r'Write in third person.*?(?=SCREEN|ACTION|The |$)',
        r'Be SPECIFIC.*?(?=SCREEN|ACTION|The |$)',
        
        # Login-specific prompt fragments
        r'LOGIN SCREEN DETAILS:.*?(?=SCREEN|ACTION|The |$)',
        r'LOGOUT DETAILS:.*?(?=SCREEN|ACTION|The |$)',
        r'MFA SCREEN:.*?(?=SCREEN|ACTION|The |$)',
        r'PASSWORD SCREEN:.*?(?=SCREEN|ACTION|The |$)',
        r'ACTION PERFORMED:.*?(?=SCREEN|ACTION|The |$)',
        
        # Question fragments (model asking for info)
        r'[Pp]lease provide.*',
        r'I need more information.*',
        r'[Cc]ould you (please )?provide.*',
        r'I cannot determine.*',
        r'[Bb]ased on the provided information,?\s*(it seems|I|the).*',
        r'I would need.*',
        r'To accurately.*',
        r'Without more context.*',
        
        # Bullet points from instructions
        r'^[\s]*[•\-\*]\s+(?:What|Is there|Are there|Note|Describe|Was|Did|How).*$',
        
        # Section dividers
        r'^[\s]*[-=]{3,}[\s]*$',
    ]
    
    cleaned = text
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
    
    # Remove multiple newlines and clean whitespace
    cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    # Remove empty lines at start/end
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    cleaned = '\n'.join(lines)
    
    return cleaned.strip()


def _extract_screen_action(response: str) -> Dict[str, str]:
    """
    Extract SCREEN and ACTION sections from vision response.
    Uses multiple parsing strategies with strict fallbacks.
    """
    result = {"screen_description": "", "action_description": ""}
    
    if not response:
        return result
    
    # First, sanitize the response to remove prompt leakage
    cleaned = _sanitize_vision_response(response)
    
    # Strategy 1: Look for explicit SCREEN: and ACTION: markers
    screen_match = re.search(
        r'SCREEN:\s*(.*?)(?=ACTION:|$)', 
        cleaned, 
        re.DOTALL | re.IGNORECASE
    )
    action_match = re.search(
        r'ACTION:\s*(.*?)$', 
        cleaned, 
        re.DOTALL | re.IGNORECASE
    )
    
    if screen_match and action_match:
        screen_text = screen_match.group(1).strip()
        action_text = action_match.group(1).strip()
        
        # Validate that these aren't just more instructions
        if len(screen_text) > 10 and not _is_instruction_text(screen_text):
            result["screen_description"] = screen_text
        if len(action_text) > 10 and not _is_instruction_text(action_text):
            result["action_description"] = action_text
    
    # Strategy 2: If we got ACTION but not SCREEN, use the first part as screen
    if result["action_description"] and not result["screen_description"]:
        # Take text before ACTION: as screen description
        before_action = re.split(r'ACTION:', cleaned, flags=re.IGNORECASE)[0].strip()
        if len(before_action) > 20 and not _is_instruction_text(before_action):
            result["screen_description"] = before_action
    
    # Strategy 3: If still nothing, try to extract any useful content
    if not result["screen_description"] and not result["action_description"]:
        # Look for sentences that start with "The user", "The system", "The screen"
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        action_sentences = []
        screen_sentences = []
        
        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 15:
                continue
            if _is_instruction_text(sent):
                continue
            
            lower = sent.lower()
            if any(lower.startswith(p) for p in ['the user ', 'user ', 'clicked ', 'selected ', 'entered ']):
                action_sentences.append(sent)
            elif any(lower.startswith(p) for p in ['the screen ', 'the page ', 'the application ', 'this is ', 'showing ']):
                screen_sentences.append(sent)
            else:
                # Could be either - add to screen if no action yet
                if not action_sentences:
                    screen_sentences.append(sent)
                else:
                    action_sentences.append(sent)
        
        if screen_sentences:
            result["screen_description"] = ' '.join(screen_sentences[:3])
        if action_sentences:
            result["action_description"] = ' '.join(action_sentences[:2])
    
    # Strategy 4: Ultimate fallback - use sanitized text as action
    if not result["action_description"]:
        if cleaned and len(cleaned) > 20 and not _is_instruction_text(cleaned):
            # Take first 200 chars as a generic action description
            result["action_description"] = cleaned[:200].strip()
            if not result["action_description"].endswith('.'):
                result["action_description"] += "."
    
    # Final validation and cleanup
    for key in result:
        if result[key]:
            result[key] = _final_cleanup(result[key])
    
    return result


def _is_instruction_text(text: str) -> bool:
    """Check if text appears to be instruction/prompt content rather than actual description."""
    if not text:
        return True
    
    lower = text.lower()
    
    instruction_indicators = [
        'instructions:', 'rules:', 'do not', "don't", 'must be', 'should be',
        'please provide', 'i need', 'could you', 'would you',
        'return exactly', 'output format', 'strict format',
        'part 1', 'part 2', 'part 3',
        'before screen', 'after screen', 'left =', 'right =',
        'you are analyzing', 'these two screenshots',
        'what application', 'is there a', 'are there any',
        'describe in detail', 'note any', 'identify the',
        'for excel', 'for web apps', 'for login',
        'if this is', 'if unclear',
    ]
    
    return any(indicator in lower for indicator in instruction_indicators)


def _final_cleanup(text: str) -> str:
    """Final cleanup of extracted text."""
    if not text:
        return ""
    
    # Remove any remaining SCREEN:/ACTION: prefixes
    text = re.sub(r'^(SCREEN|ACTION):\s*', '', text, flags=re.IGNORECASE)
    
    # Remove numbered list prefixes that look like instructions
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
    
    # Remove leading/trailing whitespace and normalize spaces
    text = ' '.join(text.split())
    
    # Ensure it doesn't start with lowercase (except for known words)
    if text and text[0].islower() and not text.startswith(('the ', 'a ', 'an ')):
        text = text[0].upper() + text[1:]
    
    return text.strip()


def _select_prompt(change_type: str, operation_category: str = None,
                   auth_info: Dict = None) -> str:
    """
    Select appropriate prompt template based on change type, operation, and auth detection.
    FIXED: Uses strict SCREEN:/ACTION: format to enable reliable parsing.
    """
    
    # Base format instruction - ALWAYS required
    format_instruction = """
OUTPUT EXACTLY IN THIS FORMAT (no other text):

SCREEN:
[2-4 sentences describing the RIGHT screenshot - application name, current page/tab, visible data, fields, buttons]

ACTION:
[1-2 sentences describing exactly what the user did to go from LEFT to RIGHT screenshot]
"""

    # AUTH-SPECIFIC PROMPTS
    if auth_info and auth_info.get("is_auth"):
        auth_type = auth_info.get("auth_type", "login")

        if auth_type == "login":
            return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER (login/sign-in screen).
{format_instruction}

Focus on: login page details, username/email field, password field, sign-in button text, SSO options, any error messages."""

        elif auth_type == "logout":
            return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER (logout/sign-out action).
{format_instruction}

Focus on: where logout option was located, confirmation dialogs, resulting page after logout."""

        elif auth_type == "mfa_verification":
            return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER (MFA/verification step).
{format_instruction}

Focus on: type of verification (OTP, push, email code), input fields, verification outcome."""

        elif auth_type == "password_change":
            return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER (password change screen).
{format_instruction}

Focus on: password fields visible, requirements shown, submit button, success/error messages."""

    # OPERATION-SPECIFIC PROMPTS
    if operation_category == "Excel":
        return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER (Excel/spreadsheet operation).
{format_instruction}

Focus on: Excel function used (VLOOKUP, FILTER, SORT, etc.), specific columns/cells involved, formula bar content, any dialogs or ribbon selections."""

    elif operation_category == "Web":
        return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER (web application).
{format_instruction}

Focus on: page/section name, button clicked, field filled, menu selection, form submission."""

    # CHANGE-TYPE SPECIFIC PROMPTS
    if change_type == "page_transition":
        return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER (page navigation).
{format_instruction}

Focus on: what page/screen the user navigated from and to, navigation method used."""

    elif change_type == "modal_popup":
        return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER (dialog/popup appeared).
{format_instruction}

Focus on: dialog title, options presented, what triggered the dialog, what action user took."""

    elif change_type == "form_input":
        return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER (form/data entry).
{format_instruction}

Focus on: which field was filled, what data was entered, any validation messages."""

    # DEFAULT PROMPT
    return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER.
{format_instruction}

Describe what application is shown and what specific action the user performed."""


def describe_transition(
    frame_before_path: str,
    frame_after_path: str,
    ocr_diff_summary: str = "",
    operation_context: str = "",
    change_region: Optional[Dict] = None,
    change_type: str = "",
    operation_category: str = None,
    call_index: int = 0,
    auth_info: Dict = None
) -> Dict[str, str]:
    """
    Describe transition with optional ROI cropping and smart prompts.
    Auth-aware: uses specialized prompts for login/logout screens.
    FIXED: Uses strict parsing to prevent prompt leakage.
    """
    before_img = frame_before_path
    after_img = frame_after_path
    used_crop = False

    # Skip cropping for auth screens — we need full context
    if change_region and not (auth_info and auth_info.get("is_auth")):
        cropped_before = _crop_to_region(frame_before_path, change_region)
        cropped_after = _crop_to_region(frame_after_path, change_region)
        if cropped_before and cropped_after:
            before_img = cropped_before
            after_img = cropped_after
            used_crop = True

    combined_path = _combine_images_side_by_side(before_img, after_img)
    if combined_path is None:
        combined_path = before_img

    # Get the appropriate prompt
    prompt = _select_prompt(change_type, operation_category, auth_info)

    # Add context hints (but keep them minimal to avoid leakage)
    if ocr_diff_summary:
        # Truncate and simplify OCR context
        ocr_hint = ocr_diff_summary[:200] if len(ocr_diff_summary) > 200 else ocr_diff_summary
        prompt += f"\n\nContext hint: {ocr_hint}"

    response = vision_client.generate(
        prompt=prompt,
        image_paths=[combined_path],
        system_prompt=VISION_SYSTEM_PROMPT,
        call_name=f"Transition_{call_index}",
        max_retries=2
    )

    # Clean up temporary files
    if used_crop:
        for p in [before_img, after_img]:
            if p not in [frame_before_path, frame_after_path] and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass
    if combined_path not in [frame_before_path, before_img] and os.path.exists(combined_path):
        try:
            os.remove(combined_path)
        except:
            pass

    # Parse response with strict extraction
    result = _extract_screen_action(response)
    
    # Log parsing result for debugging
    if not result["action_description"]:
        print(f"    [Vision] Warning: No action extracted for transition {call_index}")
        # Ultimate fallback
        result["action_description"] = "The user performed an action on the screen."
    
    if not result["screen_description"]:
        result["screen_description"] = "Screen state changed."
    
    return result


def analyze_transitions_smart(
    key_frames: List[Dict],
    ocr_diffs: List[Dict] = None,
    detected_operations: List[List[Dict]] = None,
    change_data: List[Dict] = None,
    auth_flags: List[Dict] = None
) -> List[Dict]:
    """
    Smart transition analysis with auth-awareness.
    Auth transitions always get vision calls (high priority).
    """
    if len(key_frames) < 2:
        return []

    total_pairs = len(key_frames) - 1
    max_vision = llm_params.max_vision_calls
    min_vision = min(llm_params.min_vision_calls, total_pairs)
    ocr_threshold = llm_params.ocr_sufficient_threshold

    # ── Score each transition for vision priority ──
    scored = []
    for i in range(total_pairs):
        score = 0.0
        reasons = []

        # RULE 0: Auth transitions ALWAYS get vision (highest priority)
        before_auth = auth_flags[i] if auth_flags and i < len(auth_flags) else {}
        after_auth = auth_flags[i + 1] if auth_flags and i + 1 < len(auth_flags) else {}

        if (before_auth.get("is_auth") or after_auth.get("is_auth")):
            auth_type = (before_auth.get("auth_type") or
                        after_auth.get("auth_type") or "auth")
            score += 120.0  # Higher than first/last transition
            reasons.append(f"auth_screen:{auth_type}")

        # RULE 1: First and last always get vision
        if i == 0:
            score += 100.0
            reasons.append("first_transition")
        if i == total_pairs - 1:
            score += 90.0
            reasons.append("last_transition")

        # RULE 2: Page transitions need vision
        if change_data and i < len(change_data):
            magnitude = change_data[i].get("pixel_change_magnitude", 0)
            change_type = change_data[i].get("change_type", "")

            if change_type == "page_transition":
                score += 60.0
                reasons.append("page_transition")
            elif change_type == "modal_popup":
                score += 50.0
                reasons.append("modal_popup")
            elif magnitude > 0.3:
                score += 40.0
                reasons.append(f"large_change_{magnitude:.2f}")

        # RULE 3: Operations detected
        if detected_operations and i < len(detected_operations):
            ops = detected_operations[i]
            if ops:
                excel_ops = [op for op in ops if op["category"] == "Excel"]
                auth_ops = [op for op in ops
                           if op["operation"] in ("login", "logout",
                                                  "session_management",
                                                  "password_management")]
                if auth_ops:
                    score += 80.0
                    reasons.append(f"auth_ops:{','.join(op['operation'] for op in auth_ops)}")
                elif excel_ops:
                    score += 45.0
                    reasons.append(f"excel_ops:{','.join(op['operation'] for op in excel_ops)}")
                else:
                    score += 20.0
                    reasons.append("operations_detected")

        # RULE 4: Low OCR change but significant pixel change
        if ocr_diffs and i < len(ocr_diffs):
            ocr_ratio = ocr_diffs[i].get("change_ratio", 0)
            pixel_mag = (change_data[i].get("pixel_change_magnitude", 0)
                        if change_data and i < len(change_data) else 0)

            if ocr_ratio < 0.1 and pixel_mag > 0.05:
                score += 35.0
                reasons.append("visual_change_no_text")
            elif ocr_ratio > ocr_threshold:
                score -= 15.0
                reasons.append("ocr_sufficient")

        # RULE 5: Periodic check
        if i > 0 and i < total_pairs - 1 and i % 5 == 0:
            score += 10.0
            reasons.append("periodic_check")

        scored.append((i, score, reasons))

    # ── Select transitions for vision ──
    scored.sort(key=lambda x: x[1], reverse=True)

    vision_indices = set()
    for idx, score, reasons in scored:
        if len(vision_indices) >= max_vision:
            # But always include auth transitions even over budget
            is_auth = any("auth" in r for r in reasons)
            if not is_auth:
                break
        if score > 0 or len(vision_indices) < min_vision:
            vision_indices.add(idx)

    print(
        f"    [Vision] Smart selection: {len(vision_indices)} vision calls "
        f"for {total_pairs} transitions"
    )
    for idx, score, reasons in scored[:len(vision_indices) + 5]:
        if idx in vision_indices:
            print(f"      Transition {idx+1}: score={score:.0f} [{', '.join(reasons)}]")
    skipped = total_pairs - len(vision_indices)
    if skipped > 0:
        print(f"      ... {skipped} transitions using OCR-only")

    # ── Process all transitions ──
    transitions = []
    vision_count = 0

    for i in range(total_pairs):
        before = key_frames[i]
        after = key_frames[i + 1]

        ocr_summary = ""
        ocr_before_text = before.get("ocr_text", "")
        ocr_after_text = after.get("ocr_text", "")

        if ocr_diffs and i < len(ocr_diffs):
            diff = ocr_diffs[i]
            added = diff.get("added_words", [])[:15]
            removed = diff.get("removed_words", [])[:15]
            if added:
                ocr_summary += f"New text: {', '.join(added)}\n"
            if removed:
                ocr_summary += f"Removed text: {', '.join(removed)}\n"

        op_context = ""
        operation_category = None
        if detected_operations and i < len(detected_operations):
            ops = detected_operations[i]
            if ops:
                operation_category = ops[0]["category"]
            op_context = build_operation_context(ops)

        change_region = None
        change_type = ""
        if change_data and i < len(change_data):
            change_region = change_data[i].get("primary_region")
            change_type = change_data[i].get("change_type", "")

        # Get auth info for this transition
        trans_auth = {}
        if auth_flags:
            before_auth = auth_flags[i] if i < len(auth_flags) else {}
            after_auth = auth_flags[i + 1] if i + 1 < len(auth_flags) else {}
            if before_auth.get("is_auth") or after_auth.get("is_auth"):
                trans_auth = before_auth if before_auth.get("is_auth") else after_auth

        if i in vision_indices:
            # ── VISION PATH ──
            vision_result = describe_transition(
                before["path"], after["path"],
                ocr_diff_summary=ocr_summary,
                operation_context=op_context,
                change_region=change_region,
                change_type=change_type,
                operation_category=operation_category,
                call_index=vision_count,
                auth_info=trans_auth
            )
            vision_count += 1

            after["vision_description"] = vision_result["screen_description"]

            transitions.append({
                "frame_before": before,
                "frame_after": after,
                "change_description": vision_result["action_description"],
                "timestamp_before": before.get("timestamp", 0),
                "timestamp_after": after.get("timestamp", 0),
                "order": i,
                "used_vision": True,
                "ocr_context": ocr_summary,
                "operation_context": op_context,
                "change_region": change_region,
                "change_type": change_type,
                "auth_info": trans_auth
            })

            if vision_count % 3 == 0:
                print(
                    f"    [Vision] {vision_count}/{len(vision_indices)} "
                    f"vision calls complete"
                )

        else:
            # ── OCR-ONLY PATH ──
            ocr_desc = _build_rich_ocr_description(
                ocr_before_text, ocr_after_text,
                ocr_diffs[i] if ocr_diffs and i < len(ocr_diffs) else {},
                change_data[i] if change_data and i < len(change_data) else {},
                detected_operations[i] if detected_operations and i < len(detected_operations) else [],
                trans_auth
            )

            after["vision_description"] = ocr_after_text

            transitions.append({
                "frame_before": before,
                "frame_after": after,
                "change_description": ocr_desc,
                "timestamp_before": before.get("timestamp", 0),
                "timestamp_after": after.get("timestamp", 0),
                "order": i,
                "used_vision": False,
                "ocr_context": ocr_summary,
                "operation_context": op_context,
                "change_region": change_region,
                "change_type": change_type,
                "auth_info": trans_auth
            })

    print(
        f"    [Vision] Complete: {vision_count} vision, "
        f"{total_pairs - vision_count} OCR-only"
    )
    return transitions


def _build_rich_ocr_description(
    ocr_before: str,
    ocr_after: str,
    ocr_diff: Dict,
    change_info: Dict,
    operations: List[Dict],
    auth_info: Dict = None
) -> str:
    """Build a detailed change description from OCR data alone. Auth-aware."""
    parts = []

    # Auth detection — generate detailed auth description from OCR alone
    if auth_info and auth_info.get("is_auth"):
        from pdd_no_audio.utils import get_auth_step_description
        auth_desc = get_auth_step_description(
            auth_info["auth_type"],
            auth_info.get("indicators", [])
        )
        parts.append(auth_desc)
        # Still add other context below

    # Change type
    change_type = change_info.get("change_type", "")
    if change_type == "page_transition":
        parts.append("The user navigated to a different page or screen.")
    elif change_type == "modal_popup":
        parts.append("A dialog or popup appeared on screen.")
    elif change_type == "form_input":
        parts.append("The user entered data or interacted with a form element.")
    elif change_type == "menu_interaction":
        parts.append("The user interacted with a menu or navigation element.")

    # Text changes
    added = ocr_diff.get("added_words", [])
    removed = ocr_diff.get("removed_words", [])

    if added:
        parts.append(f"New text visible: {', '.join(added[:15])}")
    if removed:
        parts.append(f"Previous text removed: {', '.join(removed[:10])}")

    # Operations (only if detected from delta, not static presence)
    if operations:
        # Filter to only delta-detected operations
        delta_ops = [op for op in operations if op.get("source") == "delta"]
        if delta_ops:
            op_names = [op["display_name"] for op in delta_ops]
            parts.append(f"Operations detected: {', '.join(op_names)}")

    # Pixel change magnitude
    magnitude = change_info.get("pixel_change_magnitude", 0)
    if magnitude > 0.5:
        parts.append("Major visual change occurred.")
    elif magnitude > 0.2:
        parts.append("Significant screen update occurred.")

    if parts:
        return " ".join(parts)

    return "The user performed an action on the screen."


def identify_application(frame_path: str) -> str:
    """Identify the application shown in a frame."""
    prompt = """What application, website, or software tool is shown in this screenshot?

OUTPUT EXACTLY IN THIS FORMAT:
APPLICATION: [application name only, e.g., "Microsoft Excel", "Google Chrome - Salesforce", "SAP GUI"]

Do not include any other text."""

    response = vision_client.generate(
        prompt=prompt,
        image_paths=[frame_path],
        call_name="IdentifyApp"
    )

    if response:
        # Extract application name
        match = re.search(r'APPLICATION:\s*(.+)', response, re.IGNORECASE)
        if match:
            name = match.group(1).strip().strip('"\'')
            if len(name) > 2 and len(name) < 50:
                return name
        
        # Fallback: try to get first line if no marker
        name = response.strip().split('\n')[0].strip().strip('"\'')
        # Remove any "APPLICATION:" prefix if present
        name = re.sub(r'^APPLICATION:\s*', '', name, flags=re.IGNORECASE)
        if len(name) > 2 and len(name) < 50:
            return name
    
    return ""


# Legacy compatibility
def describe_frame_batch(key_frames: List[Dict]) -> List[Dict]:
    return key_frames


def describe_changes_batch(
    key_frames: List[Dict],
    ocr_diffs: List[Dict] = None,
    detected_operations: List[List[Dict]] = None
) -> List[Dict]:
    return analyze_transitions_smart(key_frames, ocr_diffs, detected_operations)