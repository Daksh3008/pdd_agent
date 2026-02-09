# 🤖 PDD Agent - Meeting to Process Definition Document Converter

An AI-powered agent that automatically converts meeting video recordings into professional **Process Definition Documents (PDD)** used in RPA (Robotic Process Automation) projects.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green.svg)
![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Video Processing** | Extract audio from video files (MP4, AVI, MOV, MKV) |
| 🎤 **Speech-to-Text** | Transcribe audio using OpenAI's Whisper model |
| 🤖 **AI Analysis** | Extract process information using LLM (Ollama) |
| 📊 **Flowchart Generation** | Auto-generate process flowcharts from transcripts |
| 🖼️ **Frame Extraction** | Capture key screenshots at action timestamps |
| 📄 **Document Generation** | Create professional PDD Word documents |
| 🌐 **Web Interface** | User-friendly Streamlit UI for easy uploads |

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────────┐
│ PDD Agent Pipeline │
├─────────────────────────────────────────────────────────────────┤
│ │
│ 📹 Video Input │
│ │ │
│ ▼ │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ FFmpeg │───▶│ Whisper │───▶│ Transcript │ │
│ │ (Audio Ext) │ │(Transcribe) │ │ .txt │ │
│ └─────────────┘ └─────────────┘ └──────┬──────┘ │
│ │ │
│ ┌────────────────────────────┼───────┐ │
│ │ LLM (Ollama) │ │ │
│ ▼ ▼ ▼ ▼ │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│ │ Project │ │ Process │ │ I/O │ │ DOT │ │
│ │ Name │ │ Summary │ │ Extract │ │ Code │ │
│ └──────────┘ └──────────┘ └──────────┘ └───┬────┘ │
│ │ │
│ ┌─────────────┐ ┌────────▼─────┐ │
│ │ OpenCV │ │ Graphviz │ │
│ │ (Frames) │ │ (Flowchart) │ │
│ └──────┬──────┘ └──────┬───────┘ │
│ │ │ │
│ └──────────────────┬─────────────────────────┘ │
│ ▼ │
│ ┌─────────────────┐ │
│ │ python-docx │ │
│ │ (PDD Document) │ │
│ └────────┬────────┘ │
│ ▼ │
│ 📄 PDD_Document.docx │
│ │
└─────────────────────────────────────────────────────────────────┘


---

## 📦 Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.10+ | Runtime environment |
| **FFmpeg** | Latest | Audio extraction from video |
| **Ollama** | Latest | LLM for text analysis |
| **Graphviz** | Latest | Flowchart rendering |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **Storage** | 10 GB | 20 GB |
| **GPU** | Not required | NVIDIA GPU (for faster Whisper) |

---

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
