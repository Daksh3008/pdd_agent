# pdd_no_audio/clients/vision_llm.py

"""
Vision LLM Client for Ollama API (llama3.2-vision:11b).
Optimized: resizes images before encoding to reduce processing time.
Enhanced with robust error handling and response body logging.
"""

import base64
import requests
import json
import time
import os
import io
from typing import Optional, List

from pdd_no_audio.config import vision_config, llm_params, image_config


class VisionLLMClient:
    """Client for vision LLM calls. Resizes and compresses images."""

    def __init__(self, config=None, params=None):
        self.config = config or vision_config
        self.params = params or llm_params
        self.api_url = f"{self.config.base_url}/api/generate"
        self._tracker = None

    def set_tracker(self, tracker):
        self._tracker = tracker

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64 with resizing for faster inference."""
        if not image_path or not os.path.exists(image_path):
            print(f"    [VisionLLM] Image not found: {image_path}")
            return None
        try:
            from PIL import Image

            img = Image.open(image_path)
            original_size = img.size

            # Resize to max dimensions while maintaining aspect ratio
            max_w = image_config.max_width
            max_h = image_config.max_height

            if img.width > max_w or img.height > max_h:
                img.thumbnail((max_w, max_h), Image.LANCZOS)

            # Convert to JPEG bytes with compression
            buffer = io.BytesIO()
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode not in ('RGB', 'L'):
                # Convert any other mode (e.g., P, CMYK) to RGB
                img = img.convert('RGB')
            img.save(buffer, format='JPEG', quality=image_config.jpeg_quality)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

            if img.size != original_size:
                orig_kb = os.path.getsize(image_path) / 1024
                new_kb = len(buffer.getvalue()) / 1024
                print(
                    f"    [VisionLLM] Image resized: "
                    f"{original_size[0]}x{original_size[1]} → "
                    f"{img.size[0]}x{img.size[1]} "
                    f"({orig_kb:.0f}KB → {new_kb:.0f}KB)"
                )

            return encoded

        except ImportError:
            # Fallback: raw encoding without resize
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"    [VisionLLM] Image encode error: {e}")
            return None

    def generate(
        self,
        prompt: str,
        image_paths: List[str] = None,
        system_prompt: str = None,
        temperature: float = None,
        num_ctx: int = None,
        timeout: int = None,
        call_name: str = None,
        max_retries: int = 2
    ) -> Optional[str]:
        """
        Generate response from vision LLM with images. Retries on failure.
        """
        total_timeout = timeout or self.params.vision_timeout
        ctx = num_ctx or self.params.num_ctx

        for attempt in range(max_retries + 1):
            result = self._generate_once(
                prompt, image_paths, system_prompt, temperature,
                ctx, total_timeout, call_name, attempt
            )
            if result is not None:
                return result
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"    [VisionLLM] Retry {attempt+1}/{max_retries} after {wait}s")
                time.sleep(wait)
        return None

    def _generate_once(
        self,
        prompt: str,
        image_paths: List[str],
        system_prompt: str,
        temperature: float,
        num_ctx: int,
        total_timeout: int,
        call_name: str,
        attempt: int
    ) -> Optional[str]:
        """Single attempt at generation."""
        encoded_images = []
        if image_paths:
            for img_path in image_paths:
                encoded = self._encode_image(img_path)
                if encoded:
                    encoded_images.append(encoded)

        if image_paths and not encoded_images:
            print(f"    [VisionLLM] ⚠ No images could be encoded")
            return None

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_ctx": num_ctx,
                "temperature": (
                    temperature if temperature is not None
                    else self.params.temperature
                ),
                "top_p": self.params.top_p,
                "repeat_penalty": self.params.repeat_penalty,
            }
        }

        if encoded_images:
            payload["images"] = encoded_images
        if system_prompt:
            payload["system"] = system_prompt

        prompt_len = len(prompt)
        num_images = len(encoded_images)
        prompt_preview = prompt[:80].replace('\n', ' ')
        response_parts = []
        actual_prompt_tokens = 0
        actual_response_tokens = 0

        try:
            start = time.time()
            print(
                f"    [VisionLLM] Sending ({prompt_len} chars, "
                f"{num_images} img, timeout={total_timeout}s): "
                f"\"{prompt_preview}...\" (attempt {attempt+1})"
            )

            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True,
                timeout=(self.params.connect_timeout, self.params.stream_chunk_timeout)
            )

            # Log response body for non-200 to debug
            if response.status_code != 200:
                elapsed = time.time() - start
                body = ""
                try:
                    body = response.text[:500]
                except:
                    pass
                print(f"    [VisionLLM] ✗ HTTP {response.status_code} ({elapsed:.1f}s) - Response: {body}")
                response.close()
                self._record(call_name, prompt, system_prompt, None, time.time() - start, 0, 0, True)
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
                        print(f"    [VisionLLM] ✗ Error: {chunk['error']}")
                        break
                    if "response" in chunk:
                        response_parts.append(chunk["response"])
                        last_chunk_time = now
                    if chunk.get("done", False):
                        actual_prompt_tokens = chunk.get("prompt_eval_count", 0)
                        actual_response_tokens = chunk.get("eval_count", 0)
                        break

                if now - last_chunk_time > self.params.stream_chunk_timeout:
                    print(f"    [VisionLLM] ⚠ Stream stalled")
                    break
                if now - start > total_timeout:
                    print(f"    [VisionLLM] ⚠ Timeout — returning partial")
                    break

            response.close()
            result = "".join(response_parts).strip()
            elapsed = time.time() - start

            if result:
                token_info = ""
                if actual_prompt_tokens > 0:
                    token_info = f" [tokens: {actual_prompt_tokens}→{actual_response_tokens}]"
                print(f"    [VisionLLM] ✓ {len(result)} chars in {elapsed:.1f}s{token_info}")
            else:
                print(f"    [VisionLLM] ⚠ Empty response after {elapsed:.1f}s")

            self._record(call_name, prompt, system_prompt, result, elapsed, actual_prompt_tokens, actual_response_tokens, True)
            return result if result else None

        except requests.exceptions.ConnectionError:
            print(f"    [VisionLLM] ✗ Cannot connect to {self.config.base_url}")
            return None
        except requests.exceptions.ReadTimeout:
            if response_parts:
                result = "".join(response_parts).strip()
                if result:
                    print(f"    [VisionLLM] ⚠ Timeout but got {len(result)} chars")
                    self._record(call_name, prompt, system_prompt, result, time.time() - start, 0, 0, True)
                    return result
            print(f"    [VisionLLM] ✗ Read timeout")
            return None
        except requests.exceptions.ConnectTimeout:
            print(f"    [VisionLLM] ✗ Connection timeout")
            return None
        except Exception as e:
            if response_parts:
                result = "".join(response_parts).strip()
                if result:
                    self._record(call_name, prompt, system_prompt, result, time.time() - start, 0, 0, True)
                    return result
            print(f"    [VisionLLM] ✗ {type(e).__name__}: {e}")
            return None

    def _record(self, call_name, prompt, system_prompt, response, duration, actual_prompt, actual_response, has_image):
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
                has_image=has_image
            )

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for m in models:
                    if self.config.model in m.get("name", ""):
                        return True
            return False
        except Exception:
            return False


vision_client = VisionLLMClient()