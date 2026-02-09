# src/llm_client.py

"""
LLM Client for Ollama API.
Handles all communication with the Ollama server.
"""

import requests
import json
from typing import Optional, Generator
from config import ollama_config


class OllamaClient:
    """Client for interacting with Ollama LLM."""
    
    def __init__(self, config=None):
        self.config = config or ollama_config
        self.api_url = f"{self.config.base_url}/api/generate"
        self.timeout = 600  # 10 minutes timeout for large prompts
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        timeout: int = None
    ) -> Optional[str]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt to send to the LLM.
            system_prompt: Optional system prompt for context.
            timeout: Custom timeout in seconds.
            
        Returns:
            Generated text response or None if failed.
        """
        timeout = timeout or self.timeout
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 8192,  # Context window
                "temperature": 0.7,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            print(f"    [LLM] Sending request (timeout: {timeout}s)...")
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to Ollama at {self.config.base_url}")
            print("Please ensure Ollama is running and accessible.")
            return None
        except requests.exceptions.Timeout:
            print(f"Error: Request to Ollama timed out after {timeout}s.")
            print("Try using a smaller transcript or increase timeout.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return None
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response from the LLM.
        Keeps connection alive for long responses.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            
        Yields:
            Response text chunks.
        """
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_ctx": 8192,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            with requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True,
                timeout=600
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                        if chunk.get("done", False):
                            break
        except Exception as e:
            print(f"Streaming error: {e}")
            yield ""
    
    def generate_with_stream(
        self,
        prompt: str,
        system_prompt: str = None
    ) -> Optional[str]:
        """
        Generate response using streaming (better for long prompts).
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            
        Returns:
            Complete response text.
        """
        try:
            print("    [LLM] Streaming response...")
            response_parts = []
            for chunk in self.generate_stream(prompt, system_prompt):
                response_parts.append(chunk)
            return "".join(response_parts).strip()
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=10
            )
            return response.status_code == 200
        except:
            return False


# Default client instance
llm_client = OllamaClient()