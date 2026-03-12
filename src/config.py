# src/config.py

"""
Configuration settings for the PDD Agent.
All environment variables and constants are managed here.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""
    host: str = os.getenv("OLLAMA_HOST", "192.168.31.30")
    port: int = int(os.getenv("OLLAMA_PORT", "11434"))
    model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class WhisperConfig:
    """
    Whisper transcription configuration.
    
    language: Set to None for auto-detection.
              Whisper will detect the spoken language automatically.
    task:     "transcribe" keeps original language.
              "translate" converts to English.
              Auto-selected based on detected language when language=None.
    """
    model_name: str = "base"
    language: str = None       # None = auto-detect (was hardcoded "hi")
    task: str = "transcribe"   # "transcribe" = keep original (was hardcoded "translate")


@dataclass
class PathConfig:
    """Default paths configuration."""
    output_dir: str = "./outputs"
    ffmpeg_path: str = "ffmpeg"


@dataclass
class DocumentConfig:
    """
    Document generation configuration.
    Controls document type, labels, and structure.
    All values can be overridden per-run.
    """
    # Document type — determines title page and section headers
    document_type: str = "PDD"  # "PDD", "BRD", "SDD", "SOP"
    document_type_full: str = "Process Definition Document"

    # Classification
    classification: str = "Internal"
    status: str = "Initial Draft"
    version: str = "1.0"
    confidential: bool = True

    # Section labels — technology-neutral
    process_steps_header: str = "Automation Process Steps"
    detailed_steps_header: str = "High Level To Be Detailed Process"

    # Page layout
    margin_inches: float = 0.5  # Narrow margins


@dataclass
class LLMParams:
    """
    Parameters for LLM generation.

    CRITICAL: The Ollama server returns HTTP 500 when prompts are too large.
    Total prompt = template text + transcript sample.
    Template text is ~800-1200 chars, so sample must stay well under limit.
    """
    num_ctx: int = 8192
    temperature: float = 0.4
    top_p: float = 0.85
    repeat_penalty: float = 1.15

    # Timeouts
    connect_timeout: int = 30
    stream_chunk_timeout: int = 600
    total_timeout: int = 900

    # Prompt sizing
    # Template overhead is ~800-1200 chars per prompt
    # Total prompt must stay under ~4000 chars for reliable generation
    max_prompt_text: int = 6000       # Total max prompt (template + sample)
    max_sample_text: int = 2500       # Max transcript sample per prompt
    max_sample_small: int = 1500      # Smaller sample for simpler tasks
    max_sample_entity: int = 2000     # Sample for entity extraction
    chunk_size: int = 3000
    overlap_size: int = 150
    max_chunks: int = 3


# Action keywords for timestamp extraction (fallback)
ACTION_KEYWORDS: Dict[str, List[str]] = {
    "click": ["click", "press", "tap", "select", "choose", "hit", "push"],
    "submit": ["submit", "confirm", "send", "apply", "proceed", "finalize", "approve"],
    "open": ["open", "launch", "access", "start", "initiate", "run"],
    "navigate": ["navigate", "go to", "move to", "visit", "switch to", "redirect"],
    "type": ["type", "enter", "input", "write", "fill", "insert", "key in"],
    "scroll": ["scroll", "move down", "move up", "swipe", "browse"],
    "copy": ["copy", "duplicate", "clone", "replicate"],
    "paste": ["paste", "insert", "add", "attach", "drop", "transfer"],
    "delete": ["delete", "remove", "erase", "clear", "discard", "wipe"],
    "upload": ["upload", "attach", "add file"],
    "download": ["download", "save", "fetch", "retrieve", "get file"],
    "save": ["save", "store", "keep", "preserve", "record"],
    "refresh": ["refresh", "reload", "update", "restart"],
    "close": ["close", "exit", "shut down", "terminate", "end"],
    "approve": ["approve", "authorize", "sign off", "validate"],
    "reject": ["reject", "deny", "decline", "disapprove", "refuse"],
    "review": ["review", "analyze", "check", "evaluate", "inspect"],
    "create": ["create", "build", "generate", "produce", "construct"],
    "update": ["update", "modify", "change", "edit", "revise"],
}

# Initialize
ollama_config = OllamaConfig()
whisper_config = WhisperConfig()
path_config = PathConfig()
doc_config = DocumentConfig()
llm_params = LLMParams()