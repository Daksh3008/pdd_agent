# src/llm_client.py

"""
LLM Client for Ollama API.
ALL calls use streaming. Detects HTTP 500 errors and reports them clearly.
Integrated with TokenTracker for per-call token counting.
"""

import requests
import json
import time
from typing import Optional
from config import ollama_config, llm_params


class OllamaClient:
    """Client for Ollama LLM. Always streams to prevent timeout."""

    def __init__(self, config=None, params=None):
        self.config = config or ollama_config
        self.params = params or llm_params
        self.api_url = f"{self.config.base_url}/api/generate"
        self._tracker = None

    def set_tracker(self, tracker):
        """Attach a TokenTracker to record all calls."""
        self._tracker = tracker

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        num_ctx: int = None,
        timeout: int = None,
        call_name: str = None
    ) -> Optional[str]:
        """
        Generate response from LLM using streaming.
        Records token usage if tracker is attached.

        Args:
            prompt: The prompt text.
            system_prompt: Optional system prompt.
            temperature: Override temperature.
            num_ctx: Override context window.
            timeout: Override total timeout.
            call_name: Name for token tracking (e.g., "EntityExtraction").
        """
        total_timeout = timeout or self.params.total_timeout
        ctx = num_ctx or self.params.num_ctx

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_ctx": ctx,
                "temperature": (
                    temperature if temperature is not None
                    else self.params.temperature
                ),
                "top_p": self.params.top_p,
                "repeat_penalty": self.params.repeat_penalty,
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        prompt_len = len(prompt)
        prompt_preview = prompt[:80].replace('\n', ' ')

        response_parts = []
        actual_prompt_tokens = 0
        actual_response_tokens = 0

        try:
            start = time.time()
            print(
                f"    [LLM] Sending ({prompt_len} chars, ctx={ctx}): "
                f"\"{prompt_preview}...\""
            )

            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True,
                timeout=(
                    self.params.connect_timeout,
                    self.params.stream_chunk_timeout
                )
            )

            # CHECK HTTP STATUS
            if response.status_code == 500:
                elapsed = time.time() - start
                print(f"    [LLM] ✗ HTTP 500 from Ollama ({elapsed:.1f}s)")
                print(f"    [LLM]   Model crashed or ran out of memory.")
                print(
                    f"    [LLM]   Prompt was {prompt_len} chars. "
                    f"Try reducing prompt size."
                )
                response.close()
                self._record_call(
                    call_name, prompt, system_prompt, None,
                    time.time() - start, 0, 0
                )
                return None

            if response.status_code != 200:
                elapsed = time.time() - start
                print(
                    f"    [LLM] ✗ HTTP {response.status_code} "
                    f"({elapsed:.1f}s)"
                )
                try:
                    error_body = response.text[:200]
                    print(f"    [LLM]   Response: {error_body}")
                except Exception:
                    pass
                response.close()
                self._record_call(
                    call_name, prompt, system_prompt, None,
                    time.time() - start, 0, 0
                )
                return None

            # Read streaming response
            last_chunk_time = time.time()

            for line in response.iter_lines():
                now = time.time()

                if line:
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in chunk:
                        print(
                            f"    [LLM] ✗ Model error: {chunk['error']}"
                        )
                        break

                    if "response" in chunk:
                        response_parts.append(chunk["response"])
                        last_chunk_time = now

                    # Capture actual token counts from final chunk
                    if chunk.get("done", False):
                        actual_prompt_tokens = chunk.get(
                            "prompt_eval_count", 0
                        )
                        actual_response_tokens = chunk.get(
                            "eval_count", 0
                        )
                        break

                # Stall detection
                if now - last_chunk_time > self.params.stream_chunk_timeout:
                    print(
                        f"    [LLM] ⚠ Stream stalled for "
                        f"{self.params.stream_chunk_timeout}s"
                    )
                    break

                # Total timeout
                if now - start > total_timeout:
                    print(
                        f"    [LLM] ⚠ Total timeout {total_timeout}s "
                        f"— returning partial"
                    )
                    break

            response.close()

            result = "".join(response_parts).strip()
            elapsed = time.time() - start

            if result:
                token_info = ""
                if actual_prompt_tokens > 0:
                    token_info = (
                        f" [tokens: {actual_prompt_tokens}→"
                        f"{actual_response_tokens}]"
                    )
                print(
                    f"    [LLM] ✓ {len(result)} chars in "
                    f"{elapsed:.1f}s{token_info}"
                )
            else:
                print(f"    [LLM] ⚠ Empty response after {elapsed:.1f}s")

            # Record for tracking
            self._record_call(
                call_name, prompt, system_prompt, result,
                elapsed, actual_prompt_tokens, actual_response_tokens
            )

            return result if result else None

        except requests.exceptions.ConnectionError:
            print(
                f"    [LLM] ✗ Cannot connect to {self.config.base_url}"
            )
            self._record_call(
                call_name, prompt, system_prompt, None,
                time.time() - start if 'start' in dir() else 0, 0, 0
            )
            return None
        except requests.exceptions.ReadTimeout:
            if response_parts:
                result = "".join(response_parts).strip()
                if result:
                    print(
                        f"    [LLM] ⚠ Read timeout but got "
                        f"{len(result)} chars — using partial"
                    )
                    self._record_call(
                        call_name, prompt, system_prompt, result,
                        time.time() - start, 0, 0
                    )
                    return result
            print(f"    [LLM] ✗ Read timeout, no data received")
            self._record_call(
                call_name, prompt, system_prompt, None,
                time.time() - start, 0, 0
            )
            return None
        except requests.exceptions.ConnectTimeout:
            print(
                f"    [LLM] ✗ Connection timeout "
                f"({self.params.connect_timeout}s)"
            )
            return None
        except Exception as e:
            if response_parts:
                result = "".join(response_parts).strip()
                if result:
                    print(
                        f"    [LLM] ⚠ Error but got {len(result)} "
                        f"chars — using partial"
                    )
                    self._record_call(
                        call_name, prompt, system_prompt, result,
                        time.time() - start, 0, 0
                    )
                    return result
            print(f"    [LLM] ✗ Error: {type(e).__name__}: {e}")
            return None

    def _record_call(
        self, call_name, prompt, system_prompt, response,
        duration, actual_prompt, actual_response
    ):
        """Record call to tracker if attached."""
        if self._tracker and call_name:
            self._tracker.record(
                call_name=call_name,
                prompt=prompt or "",
                response=response or "",
                duration=duration,
                system_prompt=system_prompt or "",
                actual_prompt_tokens=actual_prompt,
                actual_response_tokens=actual_response
            )

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> Optional[dict]:
        """Get model information."""
        try:
            response = requests.post(
                f"{self.config.base_url}/api/show",
                json={"name": self.config.model},
                timeout=10
            )
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def test_generation(self) -> bool:
        """Quick test to verify model can generate."""
        print("    [LLM] Running generation test...")
        result = self.generate(
            prompt="Reply with only the word: OK",
            timeout=60,
            num_ctx=512,
            call_name="Test"
        )
        if result:
            print(f"    [LLM] Test passed: '{result[:20]}'")
            return True
        else:
            print(f"    [LLM] Test FAILED — model cannot generate")
            return False


# Default client
llm_client = OllamaClient()