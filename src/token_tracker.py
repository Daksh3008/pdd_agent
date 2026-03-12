# src/token_tracker.py

"""
Token tracking for LLM calls.
Records input tokens, output tokens, system tokens, and timing for every call.

Token estimation uses the ~4 characters per token approximation.
Ollama also returns actual token counts in streaming responses when available.

Saves report as CSV to outputs/ directory.
"""

import os
import csv
import time
from typing import Dict, List
from dataclasses import dataclass

from config import path_config


@dataclass
class CallRecord:
    """Record of a single LLM call."""
    call_name: str
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


class TokenTracker:
    """
    Tracks token usage across all LLM calls in a pipeline run.

    Usage:
        tracker = TokenTracker()
        tracker.record("EntityExtraction", prompt, system, response, duration)
        ...
        tracker.print_report()
        tracker.save_csv()
    """

    def __init__(self):
        self.calls: List[CallRecord] = []
        self.start_time: float = time.time()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count. ~4 chars per token."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def record(
        self,
        call_name: str,
        prompt: str,
        response: str,
        duration: float,
        system_prompt: str = "",
        actual_prompt_tokens: int = 0,
        actual_response_tokens: int = 0
    ):
        """Record a single LLM call."""
        prompt = prompt or ""
        response = response or ""
        system_prompt = system_prompt or ""

        prompt_tokens = self.estimate_tokens(prompt)
        system_tokens = self.estimate_tokens(system_prompt)
        response_tokens = self.estimate_tokens(response)
        total_est = prompt_tokens + system_tokens + response_tokens
        actual_total = actual_prompt_tokens + actual_response_tokens

        record = CallRecord(
            call_name=call_name,
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
            total_tokens_actual=actual_total
        )
        self.calls.append(record)

    def get_summary(self) -> Dict:
        """Get summary of all calls."""
        total_prompt_est = 0
        total_system_est = 0
        total_response_est = 0
        total_prompt_actual = 0
        total_response_actual = 0
        total_duration = 0.0

        call_details = []

        for c in self.calls:
            total_prompt_est += c.prompt_tokens_est
            total_system_est += c.system_tokens_est
            total_response_est += c.response_tokens_est
            total_prompt_actual += c.prompt_tokens_actual
            total_response_actual += c.response_tokens_actual
            total_duration += c.duration_seconds

            call_details.append({
                "name": c.call_name,
                "prompt_chars": c.prompt_chars,
                "prompt_tokens_est": c.prompt_tokens_est,
                "system_chars": c.system_chars,
                "system_tokens_est": c.system_tokens_est,
                "response_chars": c.response_chars,
                "response_tokens_est": c.response_tokens_est,
                "total_tokens_est": c.total_tokens_est,
                "duration_seconds": round(c.duration_seconds, 1),
                "prompt_tokens_actual": c.prompt_tokens_actual,
                "response_tokens_actual": c.response_tokens_actual,
                "total_tokens_actual": c.total_tokens_actual
            })

        total_est = total_prompt_est + total_system_est + total_response_est
        total_actual = total_prompt_actual + total_response_actual

        return {
            "calls": call_details,
            "num_calls": len(self.calls),
            "totals": {
                "estimated": {
                    "prompt_tokens": total_prompt_est,
                    "system_tokens": total_system_est,
                    "response_tokens": total_response_est,
                    "total_tokens": total_est
                },
                "actual": {
                    "prompt_tokens": total_prompt_actual,
                    "response_tokens": total_response_actual,
                    "total_tokens": total_actual
                },
                "duration_seconds": round(total_duration, 1),
                "pipeline_wall_time": round(
                    time.time() - self.start_time, 1
                )
            }
        }

    def print_report(self):
        """Print formatted token usage report to console."""
        summary = self.get_summary()
        totals = summary["totals"]
        est = totals["estimated"]
        actual = totals["actual"]

        print("\n" + "=" * 75)
        print("TOKEN USAGE REPORT")
        print("=" * 75)
        print(
            f"{'Call':<30} {'Prompt':>8} {'System':>8} "
            f"{'Response':>8} {'Total':>8} {'Time':>7}"
        )
        print("-" * 75)

        for c in summary["calls"]:
            actual_str = ""
            if c["total_tokens_actual"] > 0:
                actual_str = f" (actual: {c['total_tokens_actual']})"
            print(
                f"{c['name']:<30} {c['prompt_tokens_est']:>8} "
                f"{c['system_tokens_est']:>8} "
                f"{c['response_tokens_est']:>8} "
                f"{c['total_tokens_est']:>8} "
                f"{c['duration_seconds']:>6.1f}s"
                f"{actual_str}"
            )

        print("-" * 75)
        print(
            f"{'TOTAL (estimated)':<30} {est['prompt_tokens']:>8} "
            f"{est['system_tokens']:>8} {est['response_tokens']:>8} "
            f"{est['total_tokens']:>8} "
            f"{totals['duration_seconds']:>6.1f}s"
        )

        if actual["total_tokens"] > 0:
            print(
                f"{'TOTAL (actual from Ollama)':<30} "
                f"{actual['prompt_tokens']:>8} {'':>8} "
                f"{actual['response_tokens']:>8} "
                f"{actual['total_tokens']:>8}"
            )

        print(
            f"\nPipeline wall time: "
            f"{totals['pipeline_wall_time']:.1f}s "
            f"({totals['pipeline_wall_time']/60:.1f}min)"
        )
        print(
            f"LLM calls: {summary['num_calls']} | "
            f"Total estimated tokens: {est['total_tokens']:,}"
        )

        if actual["total_tokens"] > 0:
            print(f"Total actual tokens: {actual['total_tokens']:,}")

        print("=" * 75)

    def save_csv(self, project_name: str = ""):
        """
        Save token report as CSV to the outputs/ directory.

        Args:
            project_name: Used for filename. If empty, uses generic name.
        """
        output_dir = path_config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Build filename
        safe_name = "".join(
            c for c in project_name
            if c.isalnum() or c in (' ', '-', '_')
        ).strip()[:50]

        if safe_name:
            filename = f"{safe_name}_token_usage.csv"
        else:
            filename = "token_usage.csv"

        csv_path = os.path.join(output_dir, filename)

        # Avoid overwriting
        counter = 1
        base_path = csv_path
        while os.path.exists(csv_path):
            csv_path = f"{base_path.rsplit('.', 1)[0]}_{counter}.csv"
            counter += 1

        summary = self.get_summary()
        est = summary["totals"]["estimated"]
        actual = summary["totals"]["actual"]

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Call Name',
                'Prompt Chars',
                'Prompt Tokens (Est)',
                'System Chars',
                'System Tokens (Est)',
                'Response Chars',
                'Response Tokens (Est)',
                'Total Tokens (Est)',
                'Duration (seconds)',
                'Prompt Tokens (Actual)',
                'Response Tokens (Actual)',
                'Total Tokens (Actual)'
            ])

            # Per-call rows
            for c in summary["calls"]:
                writer.writerow([
                    c["name"],
                    c["prompt_chars"],
                    c["prompt_tokens_est"],
                    c["system_chars"],
                    c["system_tokens_est"],
                    c["response_chars"],
                    c["response_tokens_est"],
                    c["total_tokens_est"],
                    c["duration_seconds"],
                    c["prompt_tokens_actual"],
                    c["response_tokens_actual"],
                    c["total_tokens_actual"]
                ])

            # Blank row
            writer.writerow([])

            # Totals row
            writer.writerow([
                'TOTAL (Estimated)',
                '',
                est["prompt_tokens"],
                '',
                est["system_tokens"],
                '',
                est["response_tokens"],
                est["total_tokens"],
                summary["totals"]["duration_seconds"],
                actual["prompt_tokens"],
                actual["response_tokens"],
                actual["total_tokens"]
            ])

            # Blank row
            writer.writerow([])

            # Metadata
            writer.writerow([
                'Pipeline Wall Time (seconds)',
                summary["totals"]["pipeline_wall_time"]
            ])
            writer.writerow([
                'Pipeline Wall Time (minutes)',
                round(summary["totals"]["pipeline_wall_time"] / 60, 1)
            ])
            writer.writerow([
                'Total LLM Calls',
                summary["num_calls"]
            ])

        print(f"  📁 Token usage CSV: {csv_path}")
        return csv_path


# Global tracker instance
tracker = TokenTracker()


def reset_tracker():
    """Reset the global tracker for a new pipeline run."""
    global tracker
    tracker = TokenTracker()
    return tracker