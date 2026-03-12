PHASE 1: Frame Extraction (No LLM needed — pure OpenCV)
├── Load video
├── SSIM-based scene detection
├── Smart frame selection (15-40 key frames)
├── Cursor position detection (best-effort, OpenCV only)
└── Output: ordered list of key frames + metadata

         ↓

PHASE 2: OCR Pass (No LLM needed — Tesseract only)
├── Run Tesseract on ALL key frames
├── Extract all visible text per frame
├── Compute text diff between consecutive frames
└── Output: per-frame OCR text + diffs

         ↓

PHASE 3: Vision Model Pass (numarkdown-8b — ONE batch)
├── Ollama loads numarkdown-8b
├── For each key frame:
│   ├── "Describe this screen"
│   ├── "What application is this?"
│   ├── "What UI elements are visible?"
│   └── "What action is being performed?"
├── Also: pairwise "What changed between frame N and N+1?"
└── Ollama unloads numarkdown-8b
         
         MODEL SWAP (~30-60 seconds)

         ↓

PHASE 4: Step Synthesis (qwen2.5:14b — ONE batch)
├── Ollama loads qwen2.5:14b
├── Input per transition:
│   │   Frame N description (from vision model)
│   │   Frame N+1 description (from vision model)
│   │   OCR text diff
│   │   Cursor position change
│   └── → Output: SOP step text
├── Generate SOP sections:
│   ├── Purpose & scope
│   ├── Prerequisites
│   ├── Troubleshooting
│   └── Flowchart DOT code
└── Ollama keeps qwen2.5:14b loaded (or unloads naturally)

         ↓

PHASE 5: Document Assembly (No LLM — python-docx)
├── Annotate frames (OpenCV — red boxes, arrows)
├── Build SOP .docx
│   ├── Cover page
│   ├── Prerequisites table
│   ├── Step-by-step with annotated screenshots
│   ├── Flowchart
│   └── Troubleshooting
└── Save to outputs/





Prompt 1: Single Frame Description
You are analyzing a screenshot from a screen recording 
of someone demonstrating a software procedure.

Describe this screen:
1. What application or website is shown?
2. What page/section is the user on?
3. What UI elements are visible (buttons, menus, 
   fields, tables, tabs)?
4. What data or text is displayed?
5. Is there any cursor, selection, or highlighted 
   element visible?

Be specific and factual. Only describe what you 
actually see.



Prompt 2: Pairwise Change Detection
These are two consecutive screenshots from a screen 
recording. The user performed an action between them.

BEFORE: [image 1]
AFTER: [image 2]

What action did the user perform?
- What was clicked, typed, or selected?
- What changed on screen?
- What appeared or disappeared?

Describe the single action in one sentence.
Format: "The user [action] [target] [result]"


Prompt 3: Application Identification
What application, website, or software tool is shown 
in this screenshot?
Reply with ONLY the application name. Nothing else.


# ----------------------------------------------------------------------------

Based on the provided codebase, the PDD Generation Agent is a highly optimized, dual-pipeline system. It leverages **Google Gemini (Text & Vision)** as its core LLM, **OpenAI Whisper** for transcription, and **OpenCV/Tesseract** for visual processing. 

Here is the detailed, step-by-step technical breakdown of how the pipeline generates a Process Definition Document (PDD) for the three different input scenarios you requested.

---

### Scenario 1: Video + Audio (Standard Meeting Recording)
**Pipeline Used:** `AudioPipeline`
In this scenario, a user uploads a recording of a meeting (e.g., a Zoom/Teams call) where someone is narrating a process while demonstrating it on screen.

1. **Audio Extraction & Transcription (Whisper):**
   * **FFmpeg** extracts the audio track from the video and saves it as an MP3.
   * **OpenAI Whisper** processes the audio. It automatically detects the language and generates a timestamped transcript. If the language isn't English, it translates it to English (`task="translate"`).
2. **Consolidated LLM Extraction (Gemini Text):**
   * To avoid rate limits and JSON truncation, the transcript is sent to Gemini in exactly **two consolidated calls** (`meeting_compact.py`):
     * *Call 1 (Narrative):* Extracts the Project Name, Entities (Apps, Companies), Purpose, Overview, Business Justification, As-Is process, and To-Be process.
     * *Call 2 (Process Data):* Extracts high-level steps, detailed screen-level steps (for section 2.4), Input Requirements, Interface Requirements, and Exception Handling scenarios.
   * *Tone Enforcement & Redaction:* The text is post-processed to enforce a formal 3rd-person, present-tense tone (e.g., converting "I clicked" to "The system clicks") and PII (emails, phone numbers, names) is automatically redacted.
3. **Flowchart Generation:**
   * A 3rd LLM call asks Gemini to generate a Graphviz `DOT` language representation of the process steps.
   * The `flowchart_renderer.py` validates the DOT code, injects styling, and uses the `graphviz` library to render a PNG flowchart.
4. **Keyword-Based Frame Extraction (OpenCV):**
   * The system parses the transcript for action keywords (e.g., "click", "submit", "open") and identifies their timestamps.
   * OpenCV jumps to those exact timestamps in the video and captures screenshots.
   * The `frame_matcher.py` assigns these screenshots to the detailed process steps generated in Step 2, ensuring chronological alignment.
5. **Document Assembly (python-docx):**
   * The `PDDGenerator` compiles all the extracted text, the generated flowchart, and the extracted screenshots into a styled, professional DOCX file.

---

### Scenario 2: Video + Audio + Transcript (Provided Transcript)
**Pipeline Used:** `AudioPipeline`
In this scenario, the user uploads the video *and* provides their own text transcript (e.g., an `.srt` or `.vtt` file from Zoom/Teams).

* **Step 1 is Bypassed:** The pipeline skips the FFmpeg audio extraction and Whisper transcription entirely, saving significant processing time.
* **Steps 2 through 5 are Identical:** The pipeline immediately feeds the provided transcript into the Gemini LLM for section generation, builds the flowchart, uses the timestamps in the provided transcript to pull frames via OpenCV, and compiles the DOCX.

---

### Scenario 3: Video Only (Silent Screen Recording)
**Pipeline Used:** `VideoPipeline`
This is the most complex pipeline. It is used when a user uploads a silent video of an automation running, with absolutely no audio or text context. It relies heavily on Computer Vision and Gemini's Vision API.

1. **Scene Detection & Micro-Frame Extraction (OpenCV):**
   * Instead of pulling frames blindly, `scene_detector.py` uses **SSIM (Structural Similarity Index)** to compare consecutive frames. When SSIM drops below a threshold (meaning the screen changed significantly), it marks a "Scene Change".
   * *Micro-frames:* It grabs extra frames milliseconds before and after the change to ensure animations/popups aren't missed.
   * *Smart Sampler:* It scores frames based on edge density (UI complexity) and filters them down to a target budget (e.g., 40 keyframes).
2. **OCR & Auth Detection (Tesseract):**
   * `ocr_engine.py` runs Tesseract OCR in parallel over all extracted keyframes.
   * A heuristic engine (`detect_auth_screen`) scans the OCR text for keywords like "username", "password", "sign in", or "SSO" to automatically flag login or logout screens without needing an LLM call.
3. **Delta Analysis & Vision AI Budgeting:**
   * `change_detector.py` calculates the exact pixel difference and OCR text difference between Frame A and Frame B to classify the transition (e.g., "page_transition", "modal_popup", "form_input").
   * *Smart Budgeting:* Because Gemini Vision calls are expensive/rate-limited, the system prioritizes which transitions need the LLM. Minor changes use purely OCR-based descriptions. Major changes are combined side-by-side (Left=Before, Right=After) and sent to **Gemini Vision** to ask: *"What action did the user perform to go from the left screen to the right screen?"*
4. **Batched Step Synthesis (Gemini Text):**
   * The system gathers all the transition descriptions (from Vision AI, OCR diffs, and Auth detection).
   * To save API calls, it batches them (8 at a time) and asks Gemini to synthesize them into formal, deduplicated PDD steps ("The system clicks X...", "The system navigates to Y...").
5. **Parallel Section Generation (Gemini Text):**
   * Using the synthesized steps as context, the system spins up parallel threads (`document_sections.py`) to generate the remaining PDD components simultaneously: Purpose, As-Is, To-Be, Prerequisites, Interfaces, and Exception Handling.
6. **Deterministic Flowchart Generation:**
   * Unlike the Audio pipeline, the Video pipeline generates the Graphviz DOT code deterministically. It scans the synthesized steps for keywords like "if", "verify" (makes a Decision Diamond) or "each", "iterate" (makes a Loop block), and builds the flowchart programmatically.
7. **Frame Annotation & Document Assembly:**
   * `frame_annotator.py` uses OpenCV to draw **red bounding boxes and arrows** on the screenshots exactly where the pixel-diff algorithm detected the UI change.
   * Finally, `PDDGenerator` compiles the text, flowchart, and annotated screenshots into the final DOCX file.