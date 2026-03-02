# pdd_no_audio/frame_analysis/vision_describer.py

"""
Vision model (llama3.2-vision:11b) frame description with ROI cropping and smart prompts.
Now combines before/after images side-by-side to work around single-image limitation.
Enhanced: auth/login screen detection forces vision calls for auth transitions.
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

RULES:
1. Describe EXACTLY what you see — every detail matters.
2. Identify the APPLICATION (Excel, SAP, browser, etc.).
3. For LOGIN/AUTHENTICATION screens:
   - Identify username/email fields, password fields, sign-in buttons.
   - Note SSO options, MFA prompts, "Remember me" checkboxes.
   - Describe the authentication provider (e.g., Microsoft, Okta, Google).
   - Note any error messages or validation prompts.
4. For LOGOUT/SIGN-OUT actions:
   - Identify the logout button/link location (user menu, sidebar, etc.).
   - Note confirmation dialogs or "session ended" messages.
5. For spreadsheets: identify VLOOKUP, FILTER, SORT, PIVOT, formulas, column names, cell ranges.
6. For web apps: identify page names, buttons, fields, menu paths.
7. Note dialogs, popups, ribbon tabs, toolbar selections, status messages.
8. Be SPECIFIC: mention actual text labels, column headers, button names.
9. Do NOT mention screenshots or recordings."""


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


def _select_prompt(change_type: str, operation_category: str = None,
                   auth_info: Dict = None) -> str:
    """
    Select appropriate prompt template based on change type, operation, and auth detection.
    """
    # AUTH-SPECIFIC PROMPTS
    if auth_info and auth_info.get("is_auth"):
        auth_type = auth_info.get("auth_type", "login")

        if auth_type == "login":
            return """These two screenshots are shown side-by-side: LEFT = BEFORE, RIGHT = AFTER.

This appears to be a LOGIN / SIGN-IN screen. Describe in detail:

PART 1 — LOGIN SCREEN DETAILS:
- What application login page is shown?
- Is there a username/email field? What placeholder text or labels are visible?
- Is there a password field?
- What is the sign-in / login button text?
- Are there SSO options (e.g., "Sign in with Microsoft", "Sign in with Google")?
- Is there a "Remember me" or "Keep me signed in" checkbox?
- Is there a "Forgot password?" link?
- Any MFA/2FA prompts or verification steps?
- Any error messages or validation messages visible?

PART 2 — ACTION PERFORMED:
- Did the user enter credentials and click Sign In?
- Did the user use SSO?
- Was authentication successful (did the screen change to a dashboard/home page)?
- Describe the exact authentication flow observed."""

        elif auth_type == "logout":
            return """These two screenshots are shown side-by-side: LEFT = BEFORE, RIGHT = AFTER.

This appears to be a LOGOUT / SIGN-OUT action. Describe in detail:

PART 1 — LOGOUT DETAILS:
- Where was the logout option located (user menu, sidebar, header)?
- What was the exact text of the logout button/link?
- Was there a confirmation dialog?
- What page appeared after logout (login page, goodbye message)?

PART 2 — ACTION PERFORMED:
- How did the user initiate the logout?
- Was the session ended successfully?
- Describe the exact logout flow observed."""

        elif auth_type == "mfa_verification":
            return """These two screenshots are shown side-by-side: LEFT = BEFORE, RIGHT = AFTER.

This appears to be a MULTI-FACTOR AUTHENTICATION (MFA) step. Describe in detail:

PART 1 — MFA SCREEN:
- What type of verification is required (OTP, push notification, email code, authenticator app)?
- What input field is shown?
- What instructions are displayed?

PART 2 — ACTION PERFORMED:
- How did the user complete the verification?
- Was verification successful?"""

        elif auth_type == "password_change":
            return """These two screenshots are shown side-by-side: LEFT = BEFORE, RIGHT = AFTER.

This appears to be a PASSWORD CHANGE/RESET screen. Describe in detail:

PART 1 — PASSWORD SCREEN:
- What fields are visible (current password, new password, confirm password)?
- What password requirements/policy is shown?
- What is the submit button text?

PART 2 — ACTION PERFORMED:
- Did the user change their password?
- Was the change successful?"""

    # STANDARD PROMPTS (unchanged logic)
    base_prompt = """These two screenshots are shown side-by-side: LEFT = BEFORE, RIGHT = AFTER. The user performed an action between them.

PART 1 — AFTER SCREEN: Describe the RIGHT screenshot (the state after the action) in detail:
- What application is shown?
- What page/tab/sheet is active?
- What data, columns, fields, buttons are visible?
- Any formula bar content, dialogs, menus, ribbon selections?

PART 2 — ACTION PERFORMED: What EXACT action did the user perform?"""

    if operation_category == "Excel":
        return base_prompt + """
- For Excel: Was it VLOOKUP, FILTER, SORT, DUPLICATE REMOVAL, a formula, copy/paste?
  Which columns, cells, or ranges were involved?
- What menu item or toolbar option was used?
- Provide the exact formula if visible."""
    elif operation_category == "Web":
        return base_prompt + """
- For web apps: What button was clicked? What field was filled? What was selected?
- Was this a login, logout, or authentication action?
- What menu item or toolbar option was used?
- Provide the exact navigation path."""
    elif change_type == "modal_popup":
        return base_prompt + """
- What dialog or popup appeared?
- What options were presented?
- What action did the user take?"""
    else:
        return base_prompt + """
- What exactly changed between the two screens?
- Was this a login, logout, navigation, or data entry action?
- Describe the user's action in one detailed sentence starting with "The user..."."""


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

    prompt = _select_prompt(change_type, operation_category, auth_info)

    if ocr_diff_summary:
        prompt += f"\nText changes detected between screens:\n{ocr_diff_summary}"
    if operation_context:
        prompt += f"\n{operation_context}"

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

    result = {"screen_description": "", "action_description": ""}
    if response:
        cleaned = clean_vision_response(response)
        screen_match = re.search(r'SCREEN:\s*(.*?)(?=ACTION:|$)', cleaned, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r'ACTION:\s*(.*?)$', cleaned, re.DOTALL | re.IGNORECASE)
        if screen_match:
            result["screen_description"] = screen_match.group(1).strip()
        if action_match:
            result["action_description"] = action_match.group(1).strip()
        if not result["screen_description"] and not result["action_description"]:
            result["screen_description"] = cleaned
            result["action_description"] = cleaned
    else:
        print(f"    [Vision] No response for transition {call_index} after retries")
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
                ocr_summary += f"New text appeared: {', '.join(added)}\n"
            if removed:
                ocr_summary += f"Text disappeared: {', '.join(removed)}\n"

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
        parts.append(f"New text visible on screen: {', '.join(added[:15])}")
    if removed:
        parts.append(f"Previous text no longer visible: {', '.join(removed[:10])}")

    # Operations
    if operations:
        op_names = [op["display_name"] for op in operations]
        parts.append(f"Detected operations: {', '.join(op_names)}")

    # Pixel change magnitude
    magnitude = change_info.get("pixel_change_magnitude", 0)
    if magnitude > 0.5:
        parts.append("Major visual change — likely a page or application switch.")
    elif magnitude > 0.2:
        parts.append("Significant screen change — new section or large update.")
    elif magnitude > 0.05:
        parts.append("Moderate screen change — specific area updated.")

    if ocr_after and len(ocr_after) > 20:
        parts.append(f"Current screen text includes: {ocr_after[:300]}")

    if parts:
        return " ".join(parts)

    return "The user performed an action on the screen."


def identify_application(frame_path: str) -> str:
    """Identify the application shown in a frame."""
    prompt = """What application, website, or software tool is shown?
Reply with ONLY the application name. Nothing else.
Examples: "Microsoft Excel", "Google Chrome - Salesforce", "SAP GUI"."""

    response = vision_client.generate(
        prompt=prompt,
        image_paths=[frame_path],
        call_name="IdentifyApp"
    )

    if response:
        name = response.strip().strip('"\'').split('\n')[0].strip()
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