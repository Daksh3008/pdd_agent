# src/llm_client.py

"""
LLM Client for Ollama API.
ALL calls use streaming. Detects HTTP 500 errors and reports them clearly.
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

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        num_ctx: int = None,
        timeout: int = None
    ) -> Optional[str]:
        """
        Generate response from LLM using streaming.
        Detects HTTP 500 errors (model crash/OOM) and reports clearly.
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

            # CHECK HTTP STATUS BEFORE READING STREAM
            if response.status_code == 500:
                elapsed = time.time() - start
                print(f"    [LLM] ✗ HTTP 500 from Ollama ({elapsed:.1f}s)")
                print(f"    [LLM]   Model crashed or ran out of memory.")
                print(
                    f"    [LLM]   Prompt was {prompt_len} chars. "
                    f"Try reducing prompt size."
                )
                print(
                    f"    [LLM]   Current num_ctx={ctx}. "
                    f"Try reducing if this persists."
                )
                response.close()
                return None

            if response.status_code != 200:
                elapsed = time.time() - start
                print(
                    f"    [LLM] ✗ HTTP {response.status_code} ({elapsed:.1f}s)"
                )
                try:
                    error_body = response.text[:200]
                    print(f"    [LLM]   Response: {error_body}")
                except Exception:
                    pass
                response.close()
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
                        print(f"    [LLM] ✗ Model error: {chunk['error']}")
                        break

                    if "response" in chunk:
                        response_parts.append(chunk["response"])
                        last_chunk_time = now

                    if chunk.get("done", False):
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
                print(f"    [LLM] ✓ {len(result)} chars in {elapsed:.1f}s")
            else:
                print(f"    [LLM] ⚠ Empty response after {elapsed:.1f}s")

            return result if result else None

        except requests.exceptions.ConnectionError:
            print(f"    [LLM] ✗ Cannot connect to {self.config.base_url}")
            return None
        except requests.exceptions.ReadTimeout:
            if response_parts:
                result = "".join(response_parts).strip()
                if result:
                    print(
                        f"    [LLM] ⚠ Read timeout but got {len(result)} "
                        f"chars — using partial"
                    )
                    return result
            print(f"    [LLM] ✗ Read timeout, no data received")
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
                    return result
            print(f"    [LLM] ✗ Error: {type(e).__name__}: {e}")
            return None

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
        """
        Quick test to verify model can actually generate.
        Useful for diagnosing 500 errors vs connection issues.
        """
        print("    [LLM] Running generation test...")
        result = self.generate(
            prompt="Reply with only the word: OK",
            timeout=60,
            num_ctx=512
        )
        if result:
            print(f"    [LLM] Test passed: '{result[:20]}'")
            return True
        else:
            print(f"    [LLM] Test FAILED — model cannot generate")
            return False


# Default client
llm_client = OllamaClient()