# app/streamlit_app.py

"""
Unified Streamlit Frontend for PDD Agent.
Two modes:
1. Meeting Recording (Audio+Video) — transcript optional, auto-generated if missing
2. Silent Screen Recording (Video Only) — vision-based analysis
"""

import streamlit as st
import os
import sys
import tempfile

# load .env file first
from dotenv import load_dotenv
load_dotenv()

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.gemini_client import gemini_client
from core.config import config
from pipeline.audio_pipeline import AudioPipeline
from pipeline.video_pipeline import VideoPipeline


def main():
    st.set_page_config(
        page_title="PDD Generation Agent",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 PDD Generation Agent")
    st.markdown(
        "Convert meeting recordings or silent screen recordings into "
        "comprehensive Process Definition Documents."
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🤖 AI Model (Google Gemini)")
        api_ok = gemini_client.is_available()

        if gemini_client.is_configured() and api_ok:
            st.success("✓ Gemini API Connected")
            st.caption(f"Model: `{config.gemini.text_model}`")
            st.caption(f"RPM: {config.gemini.requests_per_minute}")
        elif gemini_client.is_configured() and not api_ok:
            st.warning("⚠ Gemini configured but health-check failed")
            st.caption(gemini_client.last_health_error() or "Unknown error")
        else:
            st.error("✗ Gemini API Not Connected")
            st.caption("Set GEMINI_API_KEY environment variable.")

        st.markdown("---")

        input_mode = st.radio(
            "📥 Input Type",
            [
                "🗣️ Meeting Recording (Audio+Video)",
                "🔇 Silent Screen Recording (Video Only)",
            ],
            index=0,
            help="Meeting Recording: has audio (or provide transcript). "
                 "Silent Recording: no audio, vision-based analysis."
        )

        st.markdown("---")

        # Document type
        doc_type_choice = st.selectbox(
            "📋 Document Type",
            [
                "PDD - Process Definition Document",
                "BRD - Business Requirements Document"
            ],
            index=0
        )
        if doc_type_choice.startswith("PDD"):
            config.document.document_type = "PDD"
            config.document.document_type_full = "Process Definition Document"
        else:
            config.document.document_type = "BRD"
            config.document.document_type_full = "Business Requirements Document"

        # Mode-specific settings
        if input_mode == "🗣️ Meeting Recording (Audio+Video)":
            st.markdown("---")
            st.subheader("🎙️ Audio Settings")
            whisper_model = st.selectbox(
                "Whisper Model",
                ["base", "small", "medium", "large"],
                index=0,
                help="Used only if no transcript is provided. Larger = more accurate but slower."
            )
        else:
            whisper_model = "base"

        if input_mode == "🔇 Silent Screen Recording (Video Only)":
            st.markdown("---")
            st.subheader("🎬 Video Settings")

            ssim_threshold = st.slider(
                "Scene Sensitivity",
                min_value=0.70, max_value=0.95,
                value=config.frame.ssim_threshold, step=0.05,
                help="Lower = more sensitive (detects smaller changes)."
            )

            max_frames = st.number_input(
                "Max Key Frames",
                min_value=10, max_value=150,
                value=config.frame.max_key_frames, step=10
            )

            enable_micro_frames = st.checkbox(
                "Enable Micro-frames",
                value=True,
                help="Extract extra frames around changes for better coverage."
            )

            annotate = st.checkbox(
                "Annotate Screenshots",
                value=config.annotation.enabled,
                help="Draw highlights on screenshots to indicate detected changes."
            )
        else:
            ssim_threshold = config.frame.ssim_threshold
            max_frames = config.frame.max_key_frames
            enable_micro_frames = False
            annotate = config.annotation.enabled

        # PII Redaction toggle
        st.markdown("---")
        st.subheader("🔒 Privacy")
        config.redaction.enabled = st.checkbox(
            "Enable PII Redaction",
            value=config.redaction.enabled,
            help="Black out names, emails, phone numbers in screenshots and text."
        )

    # ── Main Content ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📁 Input")

        project_name = st.text_input(
            "Project Name",
            placeholder="e.g., Monthly Report Generation",
            help="Required for silent videos. Auto-detected for meetings if left blank."
        )

        video_file = None
        transcript_file = None
        transcript_text = None

        # Video upload (both modes)
        st.subheader("🎬 Video File")
        video_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "mkv", "webm"]
        )
        if video_file:
            st.video(video_file)
            size_mb = len(video_file.getvalue()) / (1024 * 1024)
            st.caption(f"📏 {size_mb:.1f} MB")

        # Transcript upload (Meeting Recording mode only)
        if input_mode == "🗣️ Meeting Recording (Audio+Video)":
            st.subheader("📝 Transcript (Optional)")
            st.caption("Provide a transcript to skip Whisper transcription. "
                       "If not provided, audio will be transcribed automatically.")

            transcript_input_method = st.radio(
                "Transcript Input", ["None (Auto-transcribe)", "Upload File", "Paste Text"],
                index=0
            )

            if transcript_input_method == "Upload File":
                transcript_file = st.file_uploader(
                    "Upload Transcript", type=["txt", "srt", "vtt"]
                )
                if transcript_file:
                    preview = transcript_file.read().decode("utf-8", errors="replace")
                    transcript_file.seek(0)
                    with st.expander("Preview", expanded=False):
                        st.text(preview[:1000] + ("..." if len(preview) > 1000 else ""))

            elif transcript_input_method == "Paste Text":
                transcript_text = st.text_area(
                    "Paste Transcript",
                    height=200,
                    placeholder="Paste your transcript here..."
                )

    with col2:
        st.header("📊 Output")
        output_placeholder = st.empty()
        status_placeholder = st.empty()
        download_placeholder = st.empty()
        flowchart_placeholder = st.empty()

    # ── Validation ──
    can_process = False
    missing = ""

    if input_mode == "🗣️ Meeting Recording (Audio+Video)":
        can_process = bool(video_file)
        missing = "Upload a meeting recording video."

    elif input_mode == "🔇 Silent Screen Recording (Video Only)":
        can_process = bool(video_file) and bool(project_name.strip())
        if not project_name.strip():
            missing = "Project Name is required for silent videos."
        elif not video_file:
            missing = "Upload a silent screen recording."

    # ── Process Button ──
    st.markdown("---")
    doc_label = config.document.document_type

    if st.button(
        f"🚀 Generate {doc_label}",
        type="primary",
        use_container_width=True
    ):
        if not can_process:
            st.error(missing)
            return

        if not api_ok:
            st.error("Gemini API is not configured. Please check your API key.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = None
            transcript_path = None

            # Save uploaded video
            if video_file:
                video_path = os.path.join(temp_dir, video_file.name)
                with open(video_path, "wb") as f:
                    f.write(video_file.read())

            # Save uploaded/pasted transcript
            if transcript_file:
                transcript_path = os.path.join(temp_dir, transcript_file.name)
                with open(transcript_path, "wb") as f:
                    f.write(transcript_file.read())
            elif transcript_text and transcript_text.strip():
                transcript_path = os.path.join(temp_dir, "pasted_transcript.txt")
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript_text)

            output_dir = os.path.join(temp_dir, "output")
            result_path = None

            with st.spinner("Processing... This may take a few minutes."):
                try:
                    if input_mode == "🗣️ Meeting Recording (Audio+Video)":
                        status_placeholder.info(
                            "🔄 Running Audio Pipeline "
                            f"({'using provided transcript' if transcript_path else 'auto-transcribing'})..."
                        )
                        agent = AudioPipeline(output_dir)
                        result_path = agent.process(
                            video_path=video_path,
                            project_name=project_name.strip() if project_name else None,
                            whisper_model=whisper_model,
                            transcript_path=transcript_path
                        )

                    elif input_mode == "🔇 Silent Screen Recording (Video Only)":
                        status_placeholder.info("🔄 Running Video Pipeline (Vision AI)...")
                        agent = VideoPipeline(output_dir)
                        result_path = agent.process(
                            video_path=video_path,
                            project_name=project_name.strip(),
                            ssim_threshold=ssim_threshold,
                            max_frames=max_frames,
                            annotate=annotate,
                            enable_micro_frames=enable_micro_frames
                        )

                    # Handle result
                    if result_path and os.path.exists(result_path):
                        output_placeholder.success(f"✅ {doc_label} Generated Successfully!")
                        status_placeholder.empty()

                        with open(result_path, "rb") as f:
                            doc_bytes = f.read()

                        download_placeholder.download_button(
                            label=f"📥 Download {doc_label} Document",
                            data=doc_bytes,
                            file_name=os.path.basename(result_path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )

                        # Display flowchart — check for SVG first, then PNG
                        flowchart_svg = os.path.join(output_dir, "flowchart.svg")
                        flowchart_png = os.path.join(output_dir, "flowchart.png")

                        if os.path.exists(flowchart_svg):
                            try:
                                with open(flowchart_svg, 'r', encoding='utf-8') as f:
                                    svg_content = f.read()
                                flowchart_placeholder.markdown(
                                    f'<div style="text-align: center; background: white; '
                                    f'padding: 20px; border-radius: 8px; '
                                    f'border: 1px solid #e0e0e0;">'
                                    f'{svg_content}</div>',
                                    unsafe_allow_html=True
                                )
                            except Exception as e:
                                # Fall back to image display if SVG rendering fails
                                converted_png = os.path.join(output_dir, "flowchart_converted.png")
                                if os.path.exists(converted_png):
                                    flowchart_placeholder.image(
                                        converted_png, caption="Process Flowchart"
                                    )
                        elif os.path.exists(flowchart_png):
                            flowchart_placeholder.image(
                                flowchart_png, caption="Process Flowchart"
                            )

                        st.info(f"📁 Files saved to: `{config.paths.output_dir}/`")
                    else:
                        output_placeholder.error(f"Failed to generate {doc_label}. Check terminal logs.")

                except Exception as e:
                    output_placeholder.error(f"Error: {str(e)}")
                    status_placeholder.empty()
                    st.exception(e)


if __name__ == "__main__":
    main()