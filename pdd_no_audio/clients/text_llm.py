# pdd_no_audio/clients/text_llm.py

"""
Text LLM Client for Ollama API (qwen2.5:14b).
Used for SOP step synthesis and section generation.
ALL calls use streaming.
"""

import requests
import json
import time
from typing import Optional

from pdd_no_audio.config import text_config, llm_params


class TextLLMClient:
    """Client for text-only LLM calls. Always streams."""

    def __init__(self, config=None, params=None):
        self.config = config or text_config
        self.params = params or llm_params
        self.api_url = f"{self.config.base_url}/api/generate"
        self._tracker = None

    def set_tracker(self, tracker):
        """Attach a TokenTracker."""
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
        """Generate response from text LLM using streaming."""
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
                f"    [TextLLM] Sending ({prompt_len} chars, ctx={ctx}): "
                f"\"{prompt_preview}...\""
            )

            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True,
                timeout=(self.params.connect_timeout, self.params.stream_chunk_timeout)
            )

            if response.status_code == 500:
                elapsed = time.time() - start
                print(f"    [TextLLM] ✗ HTTP 500 ({elapsed:.1f}s)")
                response.close()
                self._record(call_name, prompt, system_prompt, None, time.time() - start, 0, 0)
                return None

            if response.status_code != 200:
                elapsed = time.time() - start
                print(f"    [TextLLM] ✗ HTTP {response.status_code} ({elapsed:.1f}s)")
                response.close()
                self._record(call_name, prompt, system_prompt, None, time.time() - start, 0, 0)
                return None

            last_chunk_time = time.time()

            for line in response.iter_lines():
                now = time.time()
                if line:
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in chunk:
                        print(f"    [TextLLM] ✗ Error: {chunk['error']}")
                        break

                    if "response" in chunk:
                        response_parts.append(chunk["response"])
                        last_chunk_time = now

                    if chunk.get("done", False):
                        actual_prompt_tokens = chunk.get("prompt_eval_count", 0)
                        actual_response_tokens = chunk.get("eval_count", 0)
                        break

                if now - last_chunk_time > self.params.stream_chunk_timeout:
                    print(f"    [TextLLM] ⚠ Stream stalled")
                    break

                if now - start > total_timeout:
                    print(f"    [TextLLM] ⚠ Total timeout — returning partial")
                    break

            response.close()

            result = "".join(response_parts).strip()
            elapsed = time.time() - start

            if result:
                token_info = ""
                if actual_prompt_tokens > 0:
                    token_info = f" [tokens: {actual_prompt_tokens}→{actual_response_tokens}]"
                print(f"    [TextLLM] ✓ {len(result)} chars in {elapsed:.1f}s{token_info}")
            else:
                print(f"    [TextLLM] ⚠ Empty response after {elapsed:.1f}s")

            self._record(
                call_name, prompt, system_prompt, result,
                elapsed, actual_prompt_tokens, actual_response_tokens
            )
            return result if result else None

        except requests.exceptions.ConnectionError:
            print(f"    [TextLLM] ✗ Cannot connect to {self.config.base_url}")
            return None
        except requests.exceptions.ReadTimeout:
            if response_parts:
                result = "".join(response_parts).strip()
                if result:
                    print(f"    [TextLLM] ⚠ Timeout but got {len(result)} chars")
                    self._record(call_name, prompt, system_prompt, result, time.time() - start, 0, 0)
                    return result
            print(f"    [TextLLM] ✗ Read timeout")
            return None
        except requests.exceptions.ConnectTimeout:
            print(f"    [TextLLM] ✗ Connection timeout")
            return None
        except Exception as e:
            if response_parts:
                result = "".join(response_parts).strip()
                if result:
                    self._record(call_name, prompt, system_prompt, result, time.time() - start, 0, 0)
                    return result
            print(f"    [TextLLM] ✗ {type(e).__name__}: {e}")
            return None

    def _record(self, call_name, prompt, system_prompt, response, duration, actual_prompt, actual_response):
        """Record call to tracker."""
        if self._tracker and call_name:
            self._tracker.record(
                call_name=call_name,
                model=self.config.model,
                prompt=prompt or "",
                response=response or "",
                duration=duration,
                system_prompt=system_prompt or "",
                actual_prompt_tokens=actual_prompt,
                actual_response_tokens=actual_response,
                has_image=False
            )

    def is_available(self) -> bool:
        """Check if Ollama server is available with this model."""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for m in models:
                    if self.config.model in m.get("name", ""):
                        return True
                return True
            return False
        except Exception:
            return False


text_client = TextLLMClient()