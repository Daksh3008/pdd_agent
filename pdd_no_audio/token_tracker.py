# pdd_no_audio/token_tracker.py

"""
Token tracking for SOP Agent LLM calls.
Tracks both vision and text model usage separately.
Includes per-model summary in report and CSV.
"""

import os
import csv
import time
from typing import List, Dict
from dataclasses import dataclass

from pdd_no_audio.config import path_config


@dataclass
class CallRecord:
    """Record of a single LLM call."""
    call_name: str
    model: str
    prompt_chars: int
    prompt_tokens_est: int
    system_chars: int
    system_tokens_est: int
    response_chars: int
    response_tokens_est: int
    total_tokens_est: int
    duration_seconds: float
    prompt_tokens_actual: int = 0
    response_tokens_actual: int = 0
    total_tokens_actual: int = 0
    has_image: bool = False


class TokenTracker:
    """Tracks token usage across all LLM calls."""

    def __init__(self):
        self.calls: List[CallRecord] = []
        self.start_time: float = time.time()

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def record(
        self,
        call_name: str,
        model: str,
        prompt: str,
        response: str,
        duration: float,
        system_prompt: str = "",
        actual_prompt_tokens: int = 0,
        actual_response_tokens: int = 0,
        has_image: bool = False
    ):
        prompt = prompt or ""
        response = response or ""
        system_prompt = system_prompt or ""

        prompt_tokens = self.estimate_tokens(prompt)
        system_tokens = self.estimate_tokens(system_prompt)
        response_tokens = self.estimate_tokens(response)
        total_est = prompt_tokens + system_tokens + response_tokens

        record = CallRecord(
            call_name=call_name,
            model=model,
            prompt_chars=len(prompt),
            prompt_tokens_est=prompt_tokens,
            system_chars=len(system_prompt),
            system_tokens_est=system_tokens,
            response_chars=len(response),
            response_tokens_est=response_tokens,
            total_tokens_est=total_est,
            duration_seconds=duration,
            prompt_tokens_actual=actual_prompt_tokens,
            response_tokens_actual=actual_response_tokens,
            total_tokens_actual=actual_prompt_tokens + actual_response_tokens,
            has_image=has_image
        )
        self.calls.append(record)

    def get_model_summary(self) -> Dict[str, Dict]:
        """Get per-model token and timing summary."""
        models = {}
        for c in self.calls:
            if c.model not in models:
                models[c.model] = {
                    "calls": 0,
                    "prompt_tokens_est": 0,
                    "system_tokens_est": 0,
                    "response_tokens_est": 0,
                    "total_tokens_est": 0,
                    "prompt_tokens_actual": 0,
                    "response_tokens_actual": 0,
                    "total_tokens_actual": 0,
                    "duration_seconds": 0.0,
                    "image_calls": 0
                }
            m = models[c.model]
            m["calls"] += 1
            m["prompt_tokens_est"] += c.prompt_tokens_est
            m["system_tokens_est"] += c.system_tokens_est
            m["response_tokens_est"] += c.response_tokens_est
            m["total_tokens_est"] += c.total_tokens_est
            m["prompt_tokens_actual"] += c.prompt_tokens_actual
            m["response_tokens_actual"] += c.response_tokens_actual
            m["total_tokens_actual"] += c.total_tokens_actual
            m["duration_seconds"] += c.duration_seconds
            if c.has_image:
                m["image_calls"] += 1
        return models

    def get_grand_totals(self) -> Dict:
        """Get combined totals across all models."""
        totals = {
            "calls": len(self.calls),
            "prompt_tokens_est": 0,
            "system_tokens_est": 0,
            "response_tokens_est": 0,
            "total_tokens_est": 0,
            "prompt_tokens_actual": 0,
            "response_tokens_actual": 0,
            "total_tokens_actual": 0,
            "duration_seconds": 0.0
        }
        for c in self.calls:
            totals["prompt_tokens_est"] += c.prompt_tokens_est
            totals["system_tokens_est"] += c.system_tokens_est
            totals["response_tokens_est"] += c.response_tokens_est
            totals["total_tokens_est"] += c.total_tokens_est
            totals["prompt_tokens_actual"] += c.prompt_tokens_actual
            totals["response_tokens_actual"] += c.response_tokens_actual
            totals["total_tokens_actual"] += c.total_tokens_actual
            totals["duration_seconds"] += c.duration_seconds
        return totals

    def print_report(self):
        """Print formatted token usage report with per-model breakdown."""
        model_summary = self.get_model_summary()
        grand_totals = self.get_grand_totals()
        wall_time = time.time() - self.start_time

        print("\n" + "=" * 90)
        print("TOKEN USAGE REPORT — SOP Agent")
        print("=" * 90)

        # Per-call details
        print(
            f"{'Call':<30} {'Model':<20} {'Prompt':>8} "
            f"{'Response':>8} {'Total':>8} {'Time':>7} {'Img':>4}"
        )
        print("-" * 90)

        for c in self.calls:
            model_short = c.model.split(':')[0][-12:]
            img_flag = "📷" if c.has_image else ""
            actual_str = ""
            if c.total_tokens_actual > 0:
                actual_str = f" (actual:{c.total_tokens_actual})"
            print(
                f"{c.call_name:<30} {model_short:<20} "
                f"{c.prompt_tokens_est:>8} "
                f"{c.response_tokens_est:>8} "
                f"{c.total_tokens_est:>8} "
                f"{c.duration_seconds:>6.1f}s "
                f"{img_flag:>4}{actual_str}"
            )

        # Per-model summary
        print("\n" + "-" * 90)
        print("PER-MODEL SUMMARY")
        print("-" * 90)
        print(
            f"{'Model':<35} {'Calls':>6} {'Prompt':>10} "
            f"{'Response':>10} {'Total(Est)':>12} {'Total(Act)':>12} {'Time':>8}"
        )
        print("-" * 90)

        for model_name, m in model_summary.items():
            img_note = f" ({m['image_calls']} img)" if m["image_calls"] > 0 else ""
            print(
                f"{model_name:<35} {m['calls']:>6}{img_note:>0} "
                f"{m['prompt_tokens_est']:>10} "
                f"{m['response_tokens_est']:>10} "
                f"{m['total_tokens_est']:>12,} "
                f"{m['total_tokens_actual']:>12,} "
                f"{m['duration_seconds']:>7.1f}s"
            )

        # Grand total
        print("-" * 90)
        print(
            f"{'GRAND TOTAL':<35} {grand_totals['calls']:>6} "
            f"{grand_totals['prompt_tokens_est']:>10} "
            f"{grand_totals['response_tokens_est']:>10} "
            f"{grand_totals['total_tokens_est']:>12,} "
            f"{grand_totals['total_tokens_actual']:>12,} "
            f"{grand_totals['duration_seconds']:>7.1f}s"
        )

        print(f"\nPipeline wall time: {wall_time:.1f}s ({wall_time / 60:.1f}min)")
        print("=" * 90)

    def save_csv(self, project_name: str = ""):
        """Save token report as CSV with per-model and grand total summaries."""
        output_dir = path_config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        safe_name = "".join(
            c for c in project_name if c.isalnum() or c in (' ', '-', '_')
        ).strip()[:50]
        filename = f"{safe_name}_sop_token_usage.csv" if safe_name else "sop_token_usage.csv"
        csv_path = os.path.join(output_dir, filename)

        counter = 1
        base_path = csv_path
        while os.path.exists(csv_path):
            csv_path = f"{base_path.rsplit('.', 1)[0]}_{counter}.csv"
            counter += 1

        model_summary = self.get_model_summary()
        grand_totals = self.get_grand_totals()
        wall_time = time.time() - self.start_time

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # ── Section 1: Per-call details ──
            writer.writerow(['=== PER-CALL DETAILS ==='])
            writer.writerow([
                'Call Name', 'Model', 'Has Image',
                'Prompt Chars', 'Prompt Tokens (Est)',
                'System Chars', 'System Tokens (Est)',
                'Response Chars', 'Response Tokens (Est)',
                'Total Tokens (Est)', 'Duration (s)',
                'Prompt Tokens (Actual)', 'Response Tokens (Actual)',
                'Total Tokens (Actual)'
            ])

            for c in self.calls:
                writer.writerow([
                    c.call_name, c.model, c.has_image,
                    c.prompt_chars, c.prompt_tokens_est,
                    c.system_chars, c.system_tokens_est,
                    c.response_chars, c.response_tokens_est,
                    c.total_tokens_est, round(c.duration_seconds, 1),
                    c.prompt_tokens_actual, c.response_tokens_actual,
                    c.total_tokens_actual
                ])

            writer.writerow([])

            # ── Section 2: Per-model summary ──
            writer.writerow(['=== PER-MODEL SUMMARY ==='])
            writer.writerow([
                'Model', 'Total Calls', 'Image Calls',
                'Prompt Tokens (Est)', 'System Tokens (Est)',
                'Response Tokens (Est)', 'Total Tokens (Est)',
                'Prompt Tokens (Actual)', 'Response Tokens (Actual)',
                'Total Tokens (Actual)', 'Total Duration (s)'
            ])

            for model_name, m in model_summary.items():
                writer.writerow([
                    model_name, m['calls'], m['image_calls'],
                    m['prompt_tokens_est'], m['system_tokens_est'],
                    m['response_tokens_est'], m['total_tokens_est'],
                    m['prompt_tokens_actual'], m['response_tokens_actual'],
                    m['total_tokens_actual'], round(m['duration_seconds'], 1)
                ])

            writer.writerow([])

            # ── Section 3: Grand totals ──
            writer.writerow(['=== GRAND TOTALS ==='])
            writer.writerow([
                'Metric', 'Value'
            ])
            writer.writerow(['Total LLM Calls', grand_totals['calls']])
            writer.writerow(['Total Prompt Tokens (Est)', grand_totals['prompt_tokens_est']])
            writer.writerow(['Total System Tokens (Est)', grand_totals['system_tokens_est']])
            writer.writerow(['Total Response Tokens (Est)', grand_totals['response_tokens_est']])
            writer.writerow(['Total Tokens (Est)', grand_totals['total_tokens_est']])
            writer.writerow(['Total Prompt Tokens (Actual)', grand_totals['prompt_tokens_actual']])
            writer.writerow(['Total Response Tokens (Actual)', grand_totals['response_tokens_actual']])
            writer.writerow(['Total Tokens (Actual)', grand_totals['total_tokens_actual']])
            writer.writerow(['Total LLM Duration (s)', round(grand_totals['duration_seconds'], 1)])
            writer.writerow(['Total LLM Duration (min)', round(grand_totals['duration_seconds'] / 60, 1)])
            writer.writerow(['Pipeline Wall Time (s)', round(wall_time, 1)])
            writer.writerow(['Pipeline Wall Time (min)', round(wall_time / 60, 1)])

        print(f"  📁 Token CSV: {csv_path}")
        return csv_path


tracker = TokenTracker()


def reset_tracker():
    """Reset the global tracker for a new pipeline run."""
    global tracker
    tracker = TokenTracker()
    return tracker