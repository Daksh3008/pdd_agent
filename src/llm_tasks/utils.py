# src/llm_tasks/utils.py

"""
Shared utility functions for LLM tasks.
Sampling, chunking, entity verification, conversation filtering.
"""

import re
import time
from typing import Dict, List, Set

from config import llm_params


MAX_SAMPLE = llm_params.max_sample_text
MAX_SAMPLE_SMALL = llm_params.max_sample_small
MAX_SAMPLE_ENTITY = llm_params.max_sample_entity
CHUNK_SIZE = llm_params.chunk_size
MAX_CHUNKS = llm_params.max_chunks
OVERLAP_SIZE = llm_params.overlap_size


# ============================================================
# Timing
# ============================================================

def _timed(name: str, start: float):
    """Log elapsed time for a task."""
    print(f"    [{name}] done in {time.time() - start:.1f}s")


# ============================================================
# Text Sampling
# ============================================================

def _safe_sample(transcript: str, max_len: int = None) -> str:
    """Take a safe-sized sample. Beginning + End."""
    max_len = max_len or MAX_SAMPLE
    if len(transcript) <= max_len:
        return transcript
    first = int(max_len * 0.6)
    last = max_len - first - 30
    return transcript[:first] + "\n[...]\n" + transcript[-last:]


def split_into_chunks(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            bp = text.rfind('. ', start + CHUNK_SIZE - 300, end)
            if bp != -1:
                end = bp + 1
        chunks.append(text[start:end].strip())
        start = end - OVERLAP_SIZE
        if start <= 0 and chunks:
            break
    return chunks


# ============================================================
# Entity Helpers
# ============================================================

def _build_entity_hint(entities: Dict) -> str:
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


def _verify_entities_against_transcript(entities: Dict, transcript: str) -> Dict:
    """
    Remove entity names that don't appear in the transcript.
    Catches LLM hallucinations.
    """
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
# Conversation Filtering
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


def _filter_conversation_steps(steps: List[str]) -> List[str]:
    """Remove steps that describe conversations/meetings, not process actions."""
    filtered = []
    for s in steps:
        s_lower = s.lower()
        if not any(phrase in s_lower for phrase in CONVERSATION_PHRASES):
            filtered.append(s)
        else:
            print(f"    [Filter] Removed conversation step: '{s[:60]}...'")
    return filtered if filtered else steps[:8]


# ============================================================
# Step Parsing
# ============================================================

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