# 🤖 PDD Agent - Meeting to Process Definition Document Converter

An AI-powered agent that automatically converts meeting video recordings into professional **Process Definition Documents (PDD)** used in RPA (Robotic Process Automation) projects.


Meeting Video/Transcript
        │
        ▼
┌─────────────────────┐
│  1. Audio Extraction │  (FFmpeg: video → audio)
│  2. Transcription    │  (Whisper: audio → timestamped text)
│  3. Entity Extraction│  (LLM: find companies, apps, systems, project name)
│  4. Section Generation│  (LLM: purpose, overview, as-is, to-be, steps...)
│  5. Flowchart        │  (LLM generates DOT → Graphviz renders PNG)
│  6. Frame Extraction │  (OpenCV: pull screenshots from video)
│  7. Frame Matching   │  (OCR + text similarity: match frames to steps)
│  8. Document Assembly│  (python-docx: build formatted .docx)
└─────────────────────┘
        │
        ▼
   PDD/BRD .docx Document
   (with flowchart, screenshots, tables, sections)


Module-by-Module Breakdown
1. Input Layer 
- FrontEnd/app.py	Streamlit UI — upload video/transcript, configure options, download output
- main.py	CLI interface — python main.py video meeting.mp4 or python main.py transcript text.txt

2. Audio/Transcription Layer
- src/video_to_audio.py	FFmpeg wrapper — extracts audio from video
- src/transcribe_audio.py	Whisper wrapper — generates timestamped transcript ([0.00 - 5.23] text...). Auto-detects language; translates non-English to English

3. LLM Extraction Layer
This is the brain of the system. It makes 10+ sequential LLM calls to a local Ollama server (default: qwen2.5:14b) to extract structured content from the transcript

File	                LLM Calls	    Output
entity_extraction.py	Call 1	        Companies, applications, systems, departments, project name
document_sections.py	Calls 2–5	    Document purpose, overview/justification, as-is process, to-be process
process_steps.py	    Calls 6, 8	    8–15 high-level automation steps + 10–25 detailed screen-level steps
requirements.py	        Calls 7, 9, 10	Input requirements, interface requirements, exception handling
flowchart_dot.py	    Call 11	        Graphviz DOT code (with deterministic step classification + LLM structure generation)
timestamps.py	        Call 12+	    Key action timestamps + description paraphrasing
system_prompt.py	    —	            PDD vs BRD system prompts (anti-hallucination rules)
utils.py	            —	            Text sampling, chunking, entity verification, conversation filtering, step parsing
compat.py	            —	            Backward compatibility wrappers


4. LLM client

- src/llm_client.py	    Ollama API client — always streams responses, detects HTTP 500, handles stalls/timeouts, integrates with token tracker
- src/config.py	        All configuration: Ollama host/model, Whisper settings, LLM parameters (context window, temperature, prompt size limits), action keywords


5. Visual Layer
File	                    Role
src/frame_extractor.py	    OpenCV — extracts video frames at specific timestamps (LLM-identified, keyword-based, or evenly-spaced fallback). Builds a candidate pool of 3× the needed frames
src/frame_matcher.py	    OCR + text similarity matching — runs Tesseract OCR on each frame, then scores every frame against every detailed step using enhanced word similarity   (with synonyms, substring matching, importance weighting). Greedy assignment with chronological fallback fill
src/flowchart_generator.py	Renders LLM-generated DOT code via graphviz.Source (preserves subgraphs). Multi-level fallback: styled → raw → fallback parser

6. Document Assembly
File	                Role
src/pdd_document.py	    python-docx generator — builds a complete professional document with: title page, version info, review/approval table, table of contents, all numbered sections (1.1–3.0), flowchart image, process step tables, input/exception tables, and Section 2.4 with sub-numbered detailed steps + matched screenshots

7. Orchaestration & Tracking
File	                    Role
src/pdd_agent.py	        Main orchestrator (PDDAgent class) — coordinates the entire pipeline: transcription → LLM calls → frame extraction → matching → document generation. Handles both video and transcript-only modes
src/token_tracker.py	    Records every LLM call's token usage (estimated + actual from Ollama), timing, and saves a CSV report



## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AI_PDD-main.git
cd AI_PDD-main/PDD-Using-AI

python -m venv venv
venv\scripts\activate

pip install -r requirements.txt

choco install ffmpeg
or 
winget install FFmpeg.FFmpeg
or 
download from: https://ffmpeg.org/download.html (and add path after intstallation)

choco install graphviz
# Or download from: https://graphviz.org/download/

on server machine:
# Download from: https://ollama.com/download/windows
# Run the installer

# On the Ollama machine
ollama pull llama3.1:8b

CREATE a .env file in project root or set environment variable
# Ollama Configuration (LLM Server)
OLLAMA_HOST=192.168.31.30
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.1:8b

# Whisper Configuration
WHISPER_MODEL=base

# Paths (Optional)
OUTPUT_DIR=./outputs
FFMPEG_PATH=ffmpeg

open powershell
# Set environment variables permanently
[System.Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "Machine")
[System.Environment]::SetEnvironmentVariable("OLLAMA_NUM_CTX", "16384", "Machine")
# Restart computer or Ollama service

OR temporary (current session only)
$env:OLLAMA_HOST="0.0.0.0"
$env:OLLAMA_NUM_CTX=16384
ollama serve

allow for firewall
# Run PowerShell as Administrator
New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound -Port 11434 -Protocol TCP -Action Allow

edit configuration file 
in class OllamaConfig
in host: str = os.getenv("OLLAMA_HOST", "192.168.31.30")  # Your Ollama server IP

verify connection
# Or using Python
python -c "from src.llm_client import llm_client; print('Connected!' if llm_client.is_available() else 'Failed!')


# --------------------------------------------------------------------

METHOD 1: Command Line Interface 
(put .mp4 file in inputs/ if running via CLI method)

# Basic usage
python main.py path/to/meeting.mp4

# With project name
python main.py path/to/meeting.mp4 --name "HR Onboarding Process"

# With custom output directory
python main.py path/to/meeting.mp4 --name "My Project" --output ./my_outputs

# Check Ollama connection
python main.py --check

METHOD 2: Streamlit web interface
streamlit run FrontEnd/app.py

if browser doesn't open automatically go to http://localhost:8501

## ------------------------------------------------------------------------------------

---

## 📋 Quick Commands Summary

Save this as a quick reference:

```bash
# ============ SETUP ============
# Install dependencies
pip install -r requirements.txt

# ============ OLLAMA (Remote Machine) ============
# Windows PowerShell
$env:OLLAMA_HOST="0.0.0.0"; $env:OLLAMA_NUM_CTX=16384; ollama serve

# Linux/Mac
OLLAMA_HOST=0.0.0.0 OLLAMA_NUM_CTX=16384 ollama serve

# ============ RUN PDD AGENT ============
# CLI
python main.py video.mp4 --name "Project Name" --output ./outputs

# Web UI
streamlit run FrontEnd/app.py

# Check connection
python main.py --check
