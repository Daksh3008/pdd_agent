# src/config.py

"""
Configuration settings for the PDD Agent.
All environment variables and constants are managed here.
"""

import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""
    host: str = os.getenv("OLLAMA_HOST", "192.168.31.30")
    port: int = int(os.getenv("OLLAMA_PORT", "11434"))
    model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class WhisperConfig:
    """Whisper transcription configuration."""
    model_name: str = "base"  # Options: tiny, base, small, medium, large, large-v2
    language: str = "hi"  # Source language for translation
    task: str = "translate"  # 'translate' or 'transcribe'


@dataclass
class PathConfig:
    """Default paths configuration."""
    output_dir: str = "./outputs"
    ffmpeg_path: str = "ffmpeg"  # Use system ffmpeg or specify full path


# Action keywords for timestamp extraction
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


# Initialize default configs
ollama_config = OllamaConfig()
whisper_config = WhisperConfig()
path_config = PathConfig()