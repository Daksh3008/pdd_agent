# FrontEnd/app.py

"""
Streamlit Frontend for PDD Agent.
Supports: Video only, Transcript only, Video + Transcript.
Saves output to persistent outputs/ directory.
"""

import streamlit as st
import os
import sys
import tempfile

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pdd_agent import PDDAgent
from llm_client import llm_client
from config import path_config, doc_config


def main():
    st.set_page_config(page_title="PDD Agent", page_icon="📄", layout="wide")

    st.title("📄 Meeting to PDD Agent")
    st.markdown(
        "Convert meeting recordings or transcripts into "
        "Process Definition Documents."
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")

        if llm_client.is_available():
            st.success("✓ Ollama LLM Connected")
            st.caption(f"Model: `{llm_client.config.model}`")
            st.caption(f"Context: `{llm_client.params.num_ctx}` tokens")
            st.caption(
                f"Timeout: `{llm_client.params.total_timeout}s` per call"
            )
        else:
            st.error("✗ Ollama LLM Not Available")
            st.info(f"Server: `{llm_client.config.base_url}`")

        st.markdown("---")

        input_mode = st.radio(
            "📥 Input Mode",
            [
                "📹 Video Only",
                "📝 Transcript Only",
                "📹 Video + 📝 Transcript"
            ],
            index=0
        )

        st.markdown("---")

        # Document type selection
        doc_type_choice = st.selectbox(
            "📋 Document Type",
            [
                "PDD - Process Definition Document",
                "BRD - Business Requirements Document"
            ],
            index=0
        )
        if doc_type_choice.startswith("PDD"):
            doc_config.document_type = "PDD"
            doc_config.document_type_full = "Process Definition Document"
        else:
            doc_config.document_type = "BRD"
            doc_config.document_type_full = "Business Requirements Document"

        st.markdown("---")

        # Whisper config — only for video modes
        whisper_model = "base"
        if input_mode in ["📹 Video Only", "📹 Video + 📝 Transcript"]:
            whisper_model = st.selectbox(
                "🎙️ Whisper Model",
                ["tiny", "base", "small", "medium", "large", "large-v2"],
                index=1
            )

        # Transcript input method
        transcript_input_method = None
        if input_mode in [
            "📝 Transcript Only", "📹 Video + 📝 Transcript"
        ]:
            transcript_input_method = st.radio(
                "📝 Transcript Input",
                ["Upload File", "Paste Text"],
                index=0
            )

        st.markdown("---")
        st.markdown("### Pipeline")
        st.markdown("""
        1. Extract entities & project name
        2. Generate purpose, summary, I/O
        3. Extract process steps
        4. Create flowchart & extract frames
        5. Produce document
        """)

    # ── Main Content ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📁 Input")

        project_name = st.text_input(
            "Project Name",
            placeholder="Optional — auto-detected from content"
        )

        video_file = None
        transcript_file = None
        transcript_text = None

        # Video upload
        if input_mode in ["📹 Video Only", "📹 Video + 📝 Transcript"]:
            st.subheader("🎬 Video")
            video_file = st.file_uploader(
                "Upload Video",
                type=["mp4", "avi", "mov", "mkv", "webm"]
            )
            if video_file:
                st.video(video_file)

        # Transcript input
        if input_mode in [
            "📝 Transcript Only", "📹 Video + 📝 Transcript"
        ]:
            st.subheader("📝 Transcript")

            if transcript_input_method == "Upload File":
                transcript_file = st.file_uploader(
                    "Upload Transcript", type=["txt", "srt", "vtt"]
                )
                if transcript_file:
                    preview = transcript_file.read().decode(
                        "utf-8", errors="replace"
                    )
                    transcript_file.seek(0)
                    with st.expander("Preview", expanded=False):
                        st.text(preview[:3000])
                    st.success(f"✓ {len(preview):,} characters loaded")

            elif transcript_input_method == "Paste Text":
                transcript_text = st.text_area(
                    "Paste Transcript",
                    height=300,
                    placeholder=(
                        "Paste transcript here...\n"
                        "[0.00 - 5.23] Text..."
                    )
                )
                if transcript_text:
                    st.caption(f"{len(transcript_text):,} characters")

    with col2:
        st.header("📊 Output")
        output_placeholder = st.empty()
        status_placeholder = st.empty()
        download_placeholder = st.empty()
        flowchart_placeholder = st.empty()

    # ── Validation ──
    can_process = False
    missing = ""

    if input_mode == "📹 Video Only":
        can_process = bool(video_file)
        missing = "Upload a video file."
    elif input_mode == "📝 Transcript Only":
        can_process = (
            bool(transcript_file)
            or bool(transcript_text and transcript_text.strip())
        )
        missing = "Upload a transcript or paste text."
    elif input_mode == "📹 Video + 📝 Transcript":
        has_transcript = (
            bool(transcript_file)
            or bool(transcript_text and transcript_text.strip())
        )
        can_process = bool(video_file) and has_transcript
        if not video_file and not has_transcript:
            missing = "Upload both video and transcript."
        elif not video_file:
            missing = "Upload a video file."
        else:
            missing = "Upload a transcript or paste text."

    # ── Process Button ──
    st.markdown("---")

    doc_type_label = doc_config.document_type

    if st.button(
        f"🚀 Generate {doc_type_label}",
        type="primary",
        use_container_width=True
    ):
        if not can_process:
            st.error(missing)
            return

        if not llm_client.is_available():
            st.error("Ollama LLM not available.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = None
            transcript_path = None

            # Save video
            if video_file:
                video_path = os.path.join(temp_dir, video_file.name)
                with open(video_path, "wb") as f:
                    f.write(video_file.read())

            # Save transcript
            if transcript_file:
                transcript_path = os.path.join(
                    temp_dir, transcript_file.name
                )
                with open(transcript_path, "wb") as f:
                    f.write(transcript_file.read())
            elif transcript_text and transcript_text.strip():
                transcript_path = os.path.join(
                    temp_dir, "pasted_transcript.txt"
                )
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript_text)

            # Process
            output_dir = os.path.join(temp_dir, "output")
            agent = PDDAgent(output_dir)

            with st.spinner(
                "Processing... This may take 10-20 minutes."
            ):
                try:
                    result_path = None

                    if input_mode == "📹 Video Only":
                        status_placeholder.info(
                            "🔄 Video → Audio → Transcription "
                            "→ LLM → Document"
                        )
                        result_path = agent.process_video(
                            video_path=video_path,
                            project_name=project_name or None,
                            whisper_model=whisper_model,
                            transcript_path=None
                        )

                    elif input_mode == "📝 Transcript Only":
                        status_placeholder.info(
                            "🔄 Transcript → LLM → Document"
                        )
                        result_path = agent.process_transcript(
                            transcript_path=transcript_path,
                            project_name=project_name or None,
                            video_path=None
                        )

                    elif input_mode == "📹 Video + 📝 Transcript":
                        status_placeholder.info(
                            "🔄 Transcript + Video → LLM → Document"
                        )
                        result_path = agent.process_video(
                            video_path=video_path,
                            project_name=project_name or None,
                            whisper_model=whisper_model,
                            transcript_path=transcript_path
                        )

                    if result_path and os.path.exists(result_path):
                        output_placeholder.success(
                            f"✅ {doc_type_label} Generated!"
                        )
                        status_placeholder.empty()

                        with open(result_path, "rb") as f:
                            doc_bytes = f.read()

                        download_placeholder.download_button(
                            label=(
                                f"📥 Download {doc_type_label} Document"
                            ),
                            data=doc_bytes,
                            file_name=os.path.basename(result_path),
                            mime=(
                                "application/vnd.openxmlformats-"
                                "officedocument.wordprocessingml.document"
                            ),
                            use_container_width=True
                        )

                        # Show flowchart
                        flowchart_path = os.path.join(
                            output_dir, "flowchart.png"
                        )
                        if os.path.exists(flowchart_path):
                            flowchart_placeholder.image(
                                flowchart_path,
                                caption="Process Flowchart"
                            )

                        st.info(
                            f"📁 Also saved to: "
                            f"`{path_config.output_dir}/`"
                        )
                    else:
                        output_placeholder.error(
                            f"Failed to generate {doc_type_label}."
                        )

                except Exception as e:
                    output_placeholder.error(f"Error: {str(e)}")
                    status_placeholder.empty()
                    st.exception(e)


if __name__ == "__main__":
    main()