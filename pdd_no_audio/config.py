# pdd_no_audio/config.py

"""
Configuration settings for PDD Agent (Silent Screen Recording).
Optimized for CPU-only inference with llama3.2-vision:11b.
Keeps full frame extraction for context, reduces vision calls smartly.
"""

import os
from dataclasses import dataclass


@dataclass
class OllamaTextConfig:
    """Text LLM configuration (qwen2.5:14b for step synthesis)."""
    host: str = os.getenv("OLLAMA_HOST", "192.168.31.30")
    port: int = int(os.getenv("OLLAMA_PORT", "11434"))
    model: str = os.getenv("OLLAMA_TEXT_MODEL", "qwen2.5:14b")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class OllamaVisionConfig:
    """Vision LLM configuration (llama3.2-vision:11b)."""
    host: str = os.getenv("OLLAMA_HOST", "192.168.31.30")
    port: int = int(os.getenv("OLLAMA_PORT", "11434"))
    model: str = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision:11b")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class DocumentConfig:
    """Document configuration — PDD format."""
    document_type: str = "PDD"
    document_type_full: str = "Process Definition Document"
    classification: str = "Internal"
    status: str = "Initial Draft"
    version: str = "1.0"
    confidential: bool = True
    margin_inches: float = 0.5
    process_steps_header: str = "Automation Process Steps"
    detailed_steps_header: str = "High Level To Be Detailed Process"


@dataclass
class FrameExtractionConfig:
    """
    Frame extraction settings.
    max_key_frames is the DEFAULT for short videos.
    For longer videos, frames scale automatically:
      frames = max(max_key_frames, duration_minutes * frames_per_minute)
    """
    ssim_threshold: float = 0.85
    min_frame_gap_seconds: float = 1.5
    max_key_frames: int = 40
    sample_interval_seconds: float = 0.5
    histogram_threshold: float = 0.7

    # Scaling for longer videos
    frames_per_minute: int = 4
    absolute_max_frames: int = 300


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


@dataclass
class ImageConfig:
    """Image preprocessing before sending to vision model."""
    max_width: int = 1024
    max_height: int = 768
    jpeg_quality: int = 75


@dataclass
class FlowchartConfig:
    """Flowchart rendering settings."""
    dpi: int = 300
    max_width_inches: float = 7.0
    font_name: str = "Arial"
    font_size: int = 11
    save_to_outputs: bool = True


@dataclass
class LLMParams:
    """Parameters for LLM generation calls."""
    num_ctx: int = 8192
    temperature: float = 0.4
    top_p: float = 0.85
    repeat_penalty: float = 1.15

    # Timeouts
    connect_timeout: int = 60
    stream_chunk_timeout: int = 1800
    total_timeout: int = 1800
    vision_timeout: int = 2400

    # Prompt sizing
    max_sample_text: int = 6000
    max_sample_small: int = 1500

    # Vision optimization
    max_vision_calls: int = 12
    min_vision_calls: int = 5
    ocr_sufficient_threshold: float = 0.3

    # Scale vision calls for longer videos
    vision_calls_per_10_frames: int = 3
    absolute_max_vision_calls: int = 40


@dataclass
class PathConfig:
    """Default paths."""
    output_dir: str = "./outputs"


# ============================================================
# Known Operations
# ============================================================

EXCEL_OPERATIONS = {
    "vlookup": ["vlookup", "v-lookup", "vertical lookup", "lookup formula"],
    "hlookup": ["hlookup", "h-lookup", "horizontal lookup"],
    "filter": ["filter", "auto filter", "autofilter", "data filter", "filtering"],
    "sort": ["sort", "sorting", "sort ascending", "sort descending", "order by"],
    "pivot_table": ["pivot", "pivot table", "pivottable", "summarize"],
    "duplicate_removal": ["remove duplicate", "duplicate", "dedup", "deduplicate", "remove duplicates"],
    "conditional_formatting": ["conditional format", "highlight cells", "color scale", "data bars"],
    "formula": ["formula", "sum", "average", "count", "countif", "sumif", "if(", "index", "match"],
    "copy_paste": ["copy", "paste", "paste special", "paste values"],
    "find_replace": ["find and replace", "find & replace", "ctrl+h", "search and replace"],
    "freeze_panes": ["freeze", "freeze panes", "split panes"],
    "merge_cells": ["merge", "merge cells", "unmerge"],
    "insert_delete": ["insert row", "insert column", "delete row", "delete column", "insert sheet"],
    "chart": ["chart", "graph", "bar chart", "line chart", "pie chart"],
    "macro": ["macro", "vba", "run macro", "enable macro"],
    "import_export": ["import", "export", "csv", "save as", "export to"],
    "text_to_columns": ["text to columns", "delimiter", "split text"],
    "data_validation": ["data validation", "dropdown list", "validation rule"],
    "subtotal": ["subtotal", "group", "outline"],
    "concatenate": ["concatenate", "concat", "textjoin", "combine text"],
}

WEB_OPERATIONS = {
    "login": ["login", "sign in", "log in", "authenticate", "credentials", "username", "password"],
    "navigate": ["navigate", "go to", "click on", "menu", "tab", "sidebar", "breadcrumb"],
    "search": ["search", "search bar", "query", "find", "look up"],
    "form_fill": ["fill", "enter", "type", "input", "text field", "text box", "dropdown"],
    "upload": ["upload", "attach", "browse", "choose file", "drag and drop"],
    "download": ["download", "export", "save", "retrieve"],
    "submit": ["submit", "save", "confirm", "apply", "send", "ok", "next"],
    "select": ["select", "choose", "pick", "checkbox", "radio button", "toggle"],
    "scroll": ["scroll", "scroll down", "scroll up", "page down"],
    "expand_collapse": ["expand", "collapse", "accordion", "toggle", "show more"],
    "table_interaction": ["sort column", "filter table", "select row", "pagination", "next page"],
    "modal_dialog": ["popup", "modal", "dialog", "alert", "confirmation"],
    "refresh": ["refresh", "reload", "update"],
    "logout": ["logout", "sign out", "log out"],
}

GENERAL_OPERATIONS = {
    "open_application": ["open", "launch", "start", "run application"],
    "close_application": ["close", "exit", "quit", "terminate"],
    "switch_window": ["switch", "alt+tab", "window", "task bar"],
    "copy_data": ["copy", "clipboard", "ctrl+c"],
    "paste_data": ["paste", "ctrl+v"],
    "screenshot": ["screenshot", "print screen", "snip"],
    "email": ["email", "outlook", "mail", "compose", "send email"],
    "file_operation": ["rename", "move", "delete file", "create folder", "new folder"],
}


# Initialize
text_config = OllamaTextConfig()
vision_config = OllamaVisionConfig()
doc_config = DocumentConfig()
frame_config = FrameExtractionConfig()
annotation_config = AnnotationConfig()
image_config = ImageConfig()
flowchart_config = FlowchartConfig()
llm_params = LLMParams()
path_config = PathConfig()