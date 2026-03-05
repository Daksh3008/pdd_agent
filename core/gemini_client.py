# core/gemini_client.py

"""
Unified Gemini API client for text and vision calls.
Includes Thread Locking to safely handle parallel workers without hitting rate limits.
"""

import os
import time
import threading
from typing import Optional, List
from PIL import Image

from google import genai
from google.genai import types
from google.genai.errors import APIError

from core.config import config


class GeminiClient:
    """
    Unified client for Google Gemini API.
    Supports text-only and multimodal (text + image) generation.
    """

    def __init__(self):
        self._client = None
        self._tracker = None
        
        # Rate Limiting & Thread Safety
        self._lock = threading.Lock()
        self._last_request_time = 0.0
        # Calculate minimum seconds between requests (e.g., 60 / 15 RPM = 4 seconds)
        self._min_request_interval = 60.0 / max(config.gemini.requests_per_minute, 1)
        
        self._configure()

    def _configure(self):
        """Configure Gemini API client."""
        api_key = config.gemini.api_key
        if not api_key:
            print("    [Gemini] ⚠ No API key set. Set GEMINI_API_KEY environment variable.")
            return
        try:
            self._client = genai.Client(api_key=api_key)
            print(f"    [Gemini] ✓ Configured (text: {config.gemini.text_model}, "
                  f"vision: {config.gemini.vision_model})")
        except Exception as e:
            print(f"    [Gemini] ✗ Configuration failed: {e}")

    def set_tracker(self, tracker):
        """Attach a TokenTracker to record all calls."""
        self._tracker = tracker

    def _rate_limit(self):
        """Thread-safe rate limiter based on requests per minute."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_request_interval:
                wait = self._min_request_interval - elapsed
                time.sleep(wait)
            # Update time *after* sleeping to ensure the next thread waits full duration
            self._last_request_time = time.time()

    def _prepare_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and resize image for Gemini vision."""
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            img = Image.open(image_path)
            max_w = config.image.max_width
            max_h = config.image.max_height

            if img.width > max_w or img.height > max_h:
                img.thumbnail((max_w, max_h), Image.LANCZOS)

            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')

            return img
        except Exception as e:
            print(f"    [Gemini] Image load error: {e}")
            return None

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        image_paths: List[str] = None,
        temperature: float = None,
        max_output_tokens: int = None,
        call_name: str = None,
        max_retries: int = 3
    ) -> Optional[str]:
        """Generate response from Gemini."""
        if not self._client:
            print("    [Gemini] ✗ Not configured. Set GEMINI_API_KEY.")
            return None

        has_images = bool(image_paths)
        model_name = config.gemini.vision_model if has_images else config.gemini.text_model
        temp = temperature if temperature is not None else config.llm.temperature
        max_tokens = max_output_tokens or config.llm.max_output_tokens

        # Prepare contents list
        contents = []
        pil_images = []
        if has_images:
            for img_path in image_paths:
                img = self._prepare_image(img_path)
                if img:
                    pil_images.append(img)
                    contents.append(img)
            if not pil_images and image_paths:
                print(f"    [Gemini] ⚠ No images could be loaded")
                return None
                
        contents.append(prompt)

        gen_config_kwargs = {
            "temperature": temp,
            "top_p": config.llm.top_p,
            "max_output_tokens": max_tokens,
        }
        if system_prompt:
            gen_config_kwargs["system_instruction"] = system_prompt
            
        gen_config = types.GenerateContentConfig(**gen_config_kwargs)

        # Retry loop
        for attempt in range(max_retries + 1):
            try:
                # This will safely queue up parallel threads!
                self._rate_limit()
                
                start = time.time()
                prompt_preview = prompt[:80].replace('\n', ' ')
                img_str = f", {len(pil_images)} img" if pil_images else ""
                print(
                    f"    [Gemini] Sending ({len(prompt)} chars{img_str}, "
                    f"model={model_name}): \"{prompt_preview}...\" "
                    f"(attempt {attempt + 1})"
                )

                response = self._client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config
                )

                elapsed = time.time() - start
                result = response.text.strip() if response.text else None

                actual_prompt_tokens = 0
                actual_response_tokens = 0
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    actual_prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    actual_response_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

                if result:
                    token_info = f" [tokens: {actual_prompt_tokens}→{actual_response_tokens}]" if actual_prompt_tokens > 0 else ""
                    print(f"    [Gemini] ✓ {len(result)} chars in {elapsed:.1f}s{token_info}")
                else:
                    print(f"    [Gemini] ⚠ Empty response after {elapsed:.1f}s")

                self._record(call_name, model_name, prompt, system_prompt, result, elapsed, actual_prompt_tokens, actual_response_tokens, has_images)
                return result

            except APIError as e:
                elapsed = time.time() - start if 'start' in dir() else 0
                
                if e.code == 429:
                    wait = 10 * (attempt + 1) # Force longer wait on 429
                    print(f"    [Gemini] ⚠ Quota exceeded. Thread pausing for {wait}s...")
                    time.sleep(wait)
                elif e.code in [500, 503]:
                    wait = 5 * (attempt + 1)
                    print(f"    [Gemini] ⚠ Server error ({e.code}). Retry in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    [Gemini] ✗ API Error {e.code}: {e.message}")
                    break
                    
                if attempt == max_retries:
                    self._record(call_name, model_name, prompt, system_prompt, None, elapsed, 0, 0, has_images)
                    
            except Exception as e:
                print(f"    [Gemini] ✗ Unexpected error: {type(e).__name__}: {str(e)}")
                if attempt == max_retries:
                    break

        return None

    def _record(self, call_name, model_name, prompt, system_prompt, response, duration, actual_prompt, actual_response, has_image):
        if self._tracker and call_name:
            self._tracker.record(call_name=call_name, model=model_name, prompt=prompt or "", response=response or "", duration=duration, system_prompt=system_prompt or "", actual_prompt_tokens=actual_prompt, actual_response_tokens=actual_response, has_image=has_image)

    def is_available(self) -> bool:
        if not self._client:
            return False
        try:
            response = self._client.models.generate_content(
                model=config.gemini.text_model,
                contents="Reply with OK",
                config=types.GenerateContentConfig(max_output_tokens=5, temperature=0.0)
            )
            return bool(response.text)
        except Exception:
            return False

# Global client instance
gemini_client = GeminiClient()