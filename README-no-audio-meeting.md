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