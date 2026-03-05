# llm/timestamps.py

"""
Key timestamp identification and description paraphrasing.
Used by the audio pipeline to identify action moments in transcript.
"""

import re
import time
from typing import Dict, List

from core.gemini_client import gemini_client
from core.config import ACTION_KEYWORDS
from core.utils import timed, parse_numbered_steps
from llm_tasks.system_prompts import get_system_prompt


def identify_key_timestamps(
    transcript: str,
    transcript_path: str
) -> List[Dict]:
    """
    Identify key action timestamps from transcript.
    Uses LLM to classify which transcript lines show process actions.
    """
    start = time.time()
    lines = []

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(
                    r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(.*)',
                    line.strip()
                )
                if m:
                    lines.append({
                        "timestamp": float(m.group(1)),
                        "text": m.group(3).strip()
                    })
    except Exception as e:
        print(f"    [Timestamps] Error reading transcript: {e}")
        return []

    if not lines:
        return []

    # Sample lines for LLM analysis
    step = max(1, len(lines) // 20)
    sampled = lines[::step][:20]
    text = "\n".join(
        [f"[{l['timestamp']:.1f}] {l['text']}" for l in sampled]
    )

    prompt = f"""Which lines show a PROCESS ACTION (opening app, clicking, typing, navigating)?
NOT: talking, explaining, greeting.

Per line respond: [time] YES action  OR  [time] NO

{text}

Answers:"""

    response = gemini_client.generate(
        prompt=prompt,
        call_name="KeyTimestamps"
    )

    moments = []
    if response:
        for line in response.split('\n'):
            m = re.search(
                r'\[?(\d+\.?\d*)\]?\s*YES\s*[-:.]?\s*(.*)',
                line, re.IGNORECASE
            )
            if m:
                ts = float(m.group(1))
                desc = m.group(2).strip()
                if not desc:
                    for sl in sampled:
                        if abs(sl["timestamp"] - ts) < 1.0:
                            desc = sl["text"]
                            break
                moments.append({
                    "timestamp": ts,
                    "description": desc or "Process action"
                })

    # Fallback: keyword matching
    if len(moments) < 3:
        all_kw = set()
        for kl in ACTION_KEYWORDS.values():
            for kw in kl:
                all_kw.add(kw.lower())
        for tl in lines:
            if any(kw in tl["text"].lower() for kw in all_kw):
                moments.append({
                    "timestamp": tl["timestamp"],
                    "description": tl["text"]
                })

    # Deduplicate close timestamps
    if moments:
        deduped = [moments[0]]
        for km in moments[1:]:
            if abs(km["timestamp"] - deduped[-1]["timestamp"]) > 5.0:
                deduped.append(km)
        moments = deduped

    # Limit to 15
    if len(moments) > 15:
        s = len(moments) // 15
        moments = moments[::s][:15]

    timed(f"Timestamps ({len(moments)})", start)
    return moments


def paraphrase_batch(
    texts: List[str],
    batch_size: int = 5
) -> List[str]:
    """
    Paraphrase frame descriptions into professional process steps.
    Used to improve raw OCR/transcript descriptions.
    """
    if not texts:
        return []

    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        numbered = "\n".join(
            [f"{j+1}. {t[:100]}" for j, t in enumerate(batch)]
        )

        prompt = f"""Rewrite each as a professional process step description.
What the SYSTEM does at this point. Third person, 1 sentence each.
Use ONLY names from the original text.

{numbered}

Rewritten:
1."""

        response = gemini_client.generate(
            prompt=prompt,
            system_prompt=get_system_prompt(),
            call_name=f"Paraphrase_batch{i//batch_size + 1}"
        )

        batch_results = []
        if response:
            if not response.strip().startswith("1"):
                response = "1. " + response
            batch_results = parse_numbered_steps(response)

        for j in range(len(batch)):
            if j < len(batch_results):
                results.append(batch_results[j])
            else:
                results.append(batch[j][:120])

    return results