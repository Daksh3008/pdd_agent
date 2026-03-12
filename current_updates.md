New file: src/frame_matcher.py     ← ALL matching logic lives here
Modified:  src/frame_extractor.py   ← Extract MORE frames (pool of candidates)
Modified:  src/pdd_agent.py         ← Calls matcher instead of positional assignment

frame_matcher.py will have these independent modules:
├── ocr_extract(frame_path) → str           # OCR text from single frame
├── ocr_batch(frame_paths) → dict           # OCR all frames
├── text_similarity(text1, text2) → float   # Word overlap score
├── score_frame_against_step(frame, step) → float  # Combined OCR + transcript score
├── match_frames_to_steps(frames, steps) → list     # Best assignment
└── fallback_chronological(frames, steps) → list    # If matching fails

Each function is independent, testable, replaceable. If you later want to swap OCR for a vision model, you only change ocr_extract()



New file: src/token_tracker.py      ← ALL token counting lives here
Modified: src/llm_client.py         ← Records tokens per call
Modified: src/pdd_agent.py          ← Prints summary at end

token_tracker.py will have:
├── TokenTracker class
│   ├── record(call_name, prompt, response, duration)
│   ├── get_summary() → dict
│   ├── get_total() → dict
│   └── print_report()
└── estimate_tokens(text) → int     # ~4 chars per token approximation