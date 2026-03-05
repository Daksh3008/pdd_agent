# core/config.py

"""
Unified configuration for the merged PDD Agent.
Supports both audio and silent-video pipelines.
Uses Google Gemini API for all LLM calls (text + vision).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


# ============================================================
# Gemini API Configuration
# ============================================================

@dataclass
class GeminiConfig:
    """Google Gemini API configuration."""
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    # Try the exact recommended strings for the new SDK
    text_model: str = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash") # If you enabled billing
    # If no billing yet, try: "gemini-1.5-flash-latest" or "gemini-1.5-flash-8b"
    
    vision_model: str = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")

    # Rate limiting
    requests_per_minute: int = int(os.getenv("GEMINI_RPM", "15"))
    tokens_per_minute: int = int(os.getenv("GEMINI_TPM", "1000000"))

# ============================================================
# Whisper Configuration (audio pipeline only)
# ============================================================

@dataclass
class WhisperConfig:
    """Whisper transcription configuration."""
    model_name: str = "base"
    language: str = None       # None = auto-detect
    task: str = "transcribe"   # "transcribe" or "translate"


# ============================================================
# Document Configuration
# ============================================================

@dataclass
class DocumentConfig:
    """Document generation configuration."""
    document_type: str = "PDD"
    document_type_full: str = "Process Definition Document"
    classification: str = "Internal"
    status: str = "Initial Draft"
    version: str = "1.0"
    confidential: bool = True
    margin_inches: float = 0.5
    process_steps_header: str = "Automation Process Steps"
    detailed_steps_header: str = "High Level To Be Detailed Process"


# ============================================================
# Frame Extraction (silent video pipeline)
# ============================================================

@dataclass
class FrameExtractionConfig:
    """Frame extraction settings for silent video pipeline."""
    ssim_threshold: float = 0.85
    min_frame_gap_seconds: float = 1.5
    max_key_frames: int = 40
    sample_interval_seconds: float = 0.5
    histogram_threshold: float = 0.7
    frames_per_minute: int = 4
    absolute_max_frames: int = 300


# ============================================================
# Annotation Configuration
# ============================================================

@dataclass
class AnnotationConfig:
    """Screenshot annotation settings."""
    enabled: bool = True
    box_color: tuple = (0, 0, 255)
    box_thickness: int = 3
    arrow_color: tuple = (255, 0, 0)
    font_scale: float = 0.8
    font_thickness: int = 2
    label_bg_color: tuple = (0, 0, 255)
    label_text_color: tuple = (255, 255, 255)


# ============================================================
# Image Preprocessing
# ============================================================

@dataclass
class ImageConfig:
    """Image preprocessing before sending to Gemini vision."""
    max_width: int = 1024
    max_height: int = 768
    jpeg_quality: int = 80


# ============================================================
# Flowchart Configuration
# ============================================================

@dataclass
class FlowchartConfig:
    """Flowchart rendering settings."""
    dpi: int = 300
    max_width_inches: float = 7.0
    font_name: str = "Arial"
    font_size: int = 11


# ============================================================
# LLM Parameters
# ============================================================

@dataclass
class LLMParams:
    """Parameters for Gemini generation calls."""
    temperature: float = 0.4
    top_p: float = 0.85
    max_output_tokens: int = 4096

    # Timeouts
    request_timeout: int = 120

    # Prompt sizing
    max_sample_text: int = 8000
    max_sample_small: int = 3000
    max_sample_entity: int = 4000
    chunk_size: int = 5000
    overlap_size: int = 200
    max_chunks: int = 5

    # Vision optimization (silent video pipeline)
    max_vision_calls: int = 15
    min_vision_calls: int = 5
    ocr_sufficient_threshold: float = 0.3
    vision_calls_per_10_frames: int = 3
    absolute_max_vision_calls: int = 50

    # Parallel workers
    max_workers: int = 3


# ============================================================
# Path Configuration
# ============================================================

@dataclass
class PathConfig:
    """Default paths."""
    output_dir: str = "./outputs"
    ffmpeg_path: str = "ffmpeg"


# ============================================================
# Action Keywords (audio pipeline — timestamp extraction)
# ============================================================

ACTION_KEYWORDS: Dict[str, List[str]] = {
    "click": ["click", "press", "tap", "select", "choose", "hit", "push"],
    "submit": ["submit", "confirm", "send", "apply", "proceed", "finalize", "approve"],
    "open": ["open", "launch", "access", "start", "initiate", "run"],
    "navigate": ["navigate", "go to", "move to", "visit", "switch to"],
    "type": ["type", "enter", "input", "write", "fill", "insert"],
    "scroll": ["scroll", "move down", "move up", "swipe"],
    "copy": ["copy", "duplicate", "clone"],
    "paste": ["paste", "insert", "add", "attach"],
    "delete": ["delete", "remove", "erase", "clear", "discard"],
    "upload": ["upload", "attach", "add file"],
    "download": ["download", "save", "fetch", "retrieve"],
    "save": ["save", "store", "keep", "preserve", "record"],
    "refresh": ["refresh", "reload", "update"],
    "close": ["close", "exit", "shut down", "terminate", "end"],
    "approve": ["approve", "authorize", "sign off", "validate"],
    "reject": ["reject", "deny", "decline", "disapprove"],
    "review": ["review", "analyze", "check", "evaluate", "inspect"],
    "create": ["create", "build", "generate", "produce"],
    "update": ["update", "modify", "change", "edit", "revise"],
}


# ============================================================
# Operation Dictionaries (silent video pipeline)
# ============================================================

EXCEL_OPERATIONS = {
    "vlookup": ["vlookup", "v-lookup", "vertical lookup"],
    "hlookup": ["hlookup", "h-lookup"],
    "filter": ["filter", "auto filter", "autofilter", "data filter"],
    "sort": ["sort", "sorting", "sort ascending", "sort descending"],
    "pivot_table": ["pivot", "pivot table", "pivottable"],
    "duplicate_removal": ["remove duplicate", "duplicate", "dedup"],
    "conditional_formatting": ["conditional format", "highlight cells"],
    "formula": ["formula", "sum", "average", "count", "countif", "sumif", "if(", "index", "match"],
    "copy_paste": ["copy", "paste", "paste special", "paste values"],
    "find_replace": ["find and replace", "find & replace", "ctrl+h"],
    "chart": ["chart", "graph", "bar chart", "line chart", "pie chart"],
    "macro": ["macro", "vba", "run macro"],
    "import_export": ["import", "export", "csv", "save as"],
    "data_validation": ["data validation", "dropdown list"],
}

WEB_OPERATIONS = {
    "login": [
        "login", "log in", "sign in", "signin", "authenticate",
        "credentials", "username", "password", "forgot password",
        "remember me", "sso", "mfa", "otp", "verification code",
    ],
    "logout": [
        "logout", "log out", "sign out", "signout", "end session",
        "session expired", "signed out",
    ],
    "navigate": ["navigate", "go to", "click on", "menu", "tab", "sidebar"],
    "search": ["search", "search bar", "query", "find"],
    "form_fill": ["fill", "enter", "type", "input", "text field", "dropdown"],
    "upload": ["upload", "attach", "browse", "choose file"],
    "download": ["download", "export", "save"],
    "submit": ["submit", "save", "confirm", "apply", "send", "ok", "next"],
    "select": ["select", "choose", "checkbox", "radio button", "toggle"],
    "modal_dialog": ["popup", "modal", "dialog", "alert", "confirmation"],
}

GENERAL_OPERATIONS = {
    "open_application": ["open", "launch", "start", "run application"],
    "close_application": ["close", "exit", "quit", "terminate"],
    "switch_window": ["switch", "alt+tab", "window"],
    "copy_data": ["copy", "clipboard", "ctrl+c"],
    "paste_data": ["paste", "ctrl+v"],
    "email": ["email", "outlook", "mail", "compose"],
    "file_operation": ["rename", "move", "delete file", "create folder"],
}

AUTH_VISUAL_INDICATORS = [
    "username", "user name", "user id", "userid", "email",
    "password", "passcode", "pin",
    "sign in", "log in", "login", "signin",
    "sign out", "log out", "logout", "signout",
    "forgot", "remember me", "keep me",
    "submit", "continue", "next",
    "welcome", "hello",
    "sso", "single sign", "okta", "azure ad",
]


# ============================================================
# Unified Config Instance
# ============================================================

@dataclass
class AppConfig:
    """Master configuration object."""
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    frame: FrameExtractionConfig = field(default_factory=FrameExtractionConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    flowchart: FlowchartConfig = field(default_factory=FlowchartConfig)
    llm: LLMParams = field(default_factory=LLMParams)
    paths: PathConfig = field(default_factory=PathConfig)


# Single global config instance
config = AppConfig()