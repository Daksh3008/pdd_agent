"""Use Gemini to select the best frame for a query and locate bounding box coordinates."""

import json
import logging
import os
import re

from google import genai
from google.genai import types
from PIL import Image

logger = logging.getLogger(__name__)

_SELECT_PROMPT = (
    "You are an expert UI analyst specialising in screen-recording videos. "
    "You will receive a user query and a set of screenshot frames.\n\n"
    "IMPORTANT: The user's query may be vague, imprecise, or use informal language. "
    "Do NOT take the query literally. Instead:\n"
    "1. Infer the user's TRUE INTENT — what action or screen are they really looking for?\n"
    "2. Carefully examine EVERY frame. Look at:\n"
    "   - Application/website names, page titles, URLs\n"
    "   - UI elements visible (buttons, input fields, menus, dialogs)\n"
    "   - Any text, labels, or data shown on screen\n"
    "   - The state of the application (is a form filled? is a dropdown open?)\n"
    "3. Consider the workflow context — the frames are sequential steps in a process. "
    "Think about which step best matches what the user is describing.\n"
    "4. Pick the single frame that BEST matches the user's intent, even if the "
    "query wording doesn't exactly match what's shown.\n\n"
    "Return ONLY a JSON object:\n"
    '  {"selected_frame_index": <0-based index of the best frame>,'
    ' "reason": "<explain what you see in the frame and why it matches the query>"}\n'
)

_BBOX_PROMPT = (
    "You are an expert UI element locator. You will receive:\n"
    "1. A user query describing an action on a UI\n"
    "2. A screenshot frame\n"
    "3. OmniParser output listing detected UI elements with their bounding boxes\n\n"
    "IMPORTANT: The user's query may be vague or imprecise. Do NOT take it literally.\n"
    "1. First, understand the user's TRUE INTENT — what element do they want to interact with?\n"
    "2. Carefully examine the screenshot to understand the full UI context.\n"
    "3. Review ALL OmniParser elements and their positions.\n"
    "4. Match the user's intent to the most relevant UI element, even if the wording "
    "doesn't exactly match the element's text.\n"
    "5. Consider element types — if the user says 'click', look for clickable elements "
    "(buttons, links, input fields, cells). If they say 'type', look for input fields.\n\n"
    "The OmniParser elements have bounding boxes in normalised [x1, y1, x2, y2] "
    "format (0.0–1.0 relative to image dimensions).\n\n"
    "Return ONLY a JSON object:\n"
    '  {"element_text": "<text/content of the matched element>",'
    ' "bbox": [x1, y1, x2, y2],'
    ' "reason": "<explain what the element is and why it matches the user intent>"}\n'
    "The bbox values must be normalised floats between 0.0 and 1.0."
)


def _load_image(filepath: str) -> Image.Image | None:
    if not os.path.isfile(filepath):
        return None
    try:
        img = Image.open(filepath)
        img.thumbnail((1024, 1024), Image.LANCZOS)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img
    except Exception:
        logger.exception("Failed to load image %s", filepath)
        return None


def _parse_json(raw: str) -> dict:
    """Best-effort JSON extraction from Gemini response."""
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: -len("```")].rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to fix truncated JSON by closing open strings/brackets
    fixed = text.rstrip()
    # Close any open string
    if fixed.count('"') % 2 == 1:
        fixed += '"'
    # Close any open array
    open_brackets = fixed.count("[") - fixed.count("]")
    fixed += "]" * max(open_brackets, 0)
    # Close any open object
    open_braces = fixed.count("{") - fixed.count("}")
    fixed += "}" * max(open_braces, 0)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Try to extract a JSON object with regex
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    logger.warning("Could not parse Gemini JSON response: %s", text[:200])
    return {}


def select_best_frame(
    api_key: str,
    query: str,
    frame_paths: list[str],
    model: str = "gemini-2.5-flash",
) -> dict:
    """Send candidate frames to Gemini and return the best match index."""
    client = genai.Client(api_key=api_key)

    contents: list = []
    valid_indices: list[int] = []

    for idx, path in enumerate(frame_paths):
        img = _load_image(path)
        if img is None:
            continue
        contents.append(img)
        contents.append(f"Frame {idx}: {os.path.basename(path)}")
        valid_indices.append(idx)

    if not valid_indices:
        return {"selected_frame_index": 0, "reason": "no valid frames"}

    contents.append(
        f'User query: "{query}"\n\n'
        f"Pick the best frame (0-{len(frame_paths) - 1}) that matches this query."
    )

    config = types.GenerateContentConfig(
        system_instruction=_SELECT_PROMPT,
        temperature=0.1,
        max_output_tokens=4096,
        response_mime_type="application/json",
    )

    response = client.models.generate_content(
        model=model, contents=contents, config=config
    )

    raw = (response.text or "").strip()
    result = _parse_json(raw)
    logger.info("Gemini selected frame %d: %s", result.get("selected_frame_index", 0), result.get("reason", ""))
    return result


def locate_bounding_box(
    api_key: str,
    query: str,
    frame_path: str,
    omniparser_elements: list[dict],
    model: str = "gemini-2.5-flash",
) -> dict:
    """Send the chosen frame + OmniParser elements to Gemini to locate the target element bbox."""
    client = genai.Client(api_key=api_key)

    img = _load_image(frame_path)
    if img is None:
        return {"element_text": "", "bbox": [0, 0, 0, 0], "reason": "image load failed"}

    elements_str = json.dumps(omniparser_elements, indent=2)

    contents = [
        img,
        f'User query: "{query}"\n\n'
        f"OmniParser detected elements:\n{elements_str}\n\n"
        "Identify the UI element the user is referring to and return its bounding box.",
    ]

    config = types.GenerateContentConfig(
        system_instruction=_BBOX_PROMPT,
        temperature=0.1,
        max_output_tokens=4096,
        response_mime_type="application/json",
    )

    response = client.models.generate_content(
        model=model, contents=contents, config=config
    )

    raw = (response.text or "").strip()
    result = _parse_json(raw)
    logger.info("Gemini located element '%s' at %s", result.get("element_text", ""), result.get("bbox", []))
    return result
