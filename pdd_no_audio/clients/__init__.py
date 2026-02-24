# pdd_no_audio/clients/__init__.py

"""
LLM clients for SOP Agent.
Text client (qwen2.5:14b) and Vision client (llama3.2-vision:11b).
"""

from pdd_no_audio.clients.text_llm import text_client
from pdd_no_audio.clients.vision_llm import vision_client