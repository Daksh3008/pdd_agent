# FrontEnd/no_audio_meeting.py

"""
Streamlit Frontend for PDD Agent (Silent Screen Recording).
Processes silent screen recordings into Process Definition Documents.
"""

import streamlit as st
import os
import sys
import tempfile

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pdd_no_audio.sop_agent import SOPAgent
from pdd_no_audio.clients.text_llm import text_client
from pdd_no_audio.clients.vision_llm import vision_client
from pdd_no_audio.config import (
    path_config, doc_config, frame_config,
    annotation_config, text_config, vision_config
)


def main():
    st.set_page_config(
        page_title="PDD Agent — Silent Screen Recording",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 Silent Screen Recording → PDD Generator")
    st.markdown(
        "Convert silent screen recordings into "
        "Process Definition Documents with detailed operation analysis."
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🤖 Models")
        vision_ok = vision_client.is_available()
        text_ok = text_client.is_available()

        if vision_ok:
            st.success(f"✓ Vision: `{vision_config.model}`")
        else:
            st.error(f"✗ Vision: `{vision_config.model}`")

        if text_ok:
            st.success(f"✓ Text: `{text_config.model}`")
        else:
            st.error(f"✗ Text: `{text_config.model}`")

        st.caption(f"Server: `{text_config.base_url}`")

        st.markdown("---")

        st.subheader("🎬 Frame Extraction")

        ssim_threshold = st.slider(
            "Scene Sensitivity",
            min_value=0.70, max_value=0.95,
            value=frame_config.ssim_threshold, step=0.05,
            help="Lower = more sensitive (detects smaller changes)."
        )

        max_frames = st.number_input(
            "Max Key Frames",
            min_value=5, max_value=80,
            value=frame_config.max_key_frames, step=5
        )

        st.markdown("---")

        st.subheader("🖼️ Screenshots")
        annotate = st.checkbox(
            "Annotate Screenshots",
            value=annotation_config.enabled,
            help="Draw red boxes and arrows to highlight actions."
        )

        st.markdown("---")

        st.subheader("📋 Pipeline")
        st.markdown("""
        1. **Extract key frames** — SSIM scene detection
        2. **Read screen text** — Tesseract OCR
        3. **Analyze screens** — Vision AI (llama3.2-vision)
        4. **Generate PDD** — Text AI (qwen2.5) with operation detection
        5. **Build document** — Formatted PDD .docx
        """)

        st.markdown("---")
        st.caption("⏱ Processing may take 30–90 minutes.")

    # ── Main Content ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📁 Input")

        project_name = st.text_input(
            "Project Name *",
            placeholder="e.g., Monthly Report Generation Process",
            help="Required. Descriptive name for the PDD."
        )

        st.subheader("🎬 Video")
        video_file = st.file_uploader(
            "Upload Silent Screen Recording",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Silent screen recording showing the process."
        )

        if video_file:
            st.video(video_file)
            size_mb = len(video_file.getvalue()) / (1024 * 1024)
            st.caption(f"📏 {size_mb:.1f} MB")

    with col2:
        st.header("📊 Output")
        output_placeholder = st.empty()
        status_placeholder = st.empty()
        download_placeholder = st.empty()
        flowchart_placeholder = st.empty()

    # ── Validation ──
    can_process = bool(video_file) and bool(project_name and project_name.strip())
    missing = ""
    if not project_name or not project_name.strip():
        missing = "Enter a project name."
    elif not video_file:
        missing = "Upload a video file."

    st.markdown("---")

    if st.button("🚀 Generate PDD", type="primary", use_container_width=True):
        if not can_process:
            st.error(missing)
            return

        if not vision_ok:
            st.error(f"Vision model ({vision_config.model}) not available.")
            return

        if not text_ok:
            st.error(f"Text model ({text_config.model}) not available.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, video_file.name)
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            output_dir = os.path.join(temp_dir, "output")
            agent = SOPAgent(output_dir)

            with st.spinner("Processing... This may take 30–90 minutes."):
                try:
                    status_placeholder.info("🔄 Generating PDD from screen recording...")

                    result_path = agent.process_video(
                        video_path=video_path,
                        project_name=project_name.strip(),
                        ssim_threshold=ssim_threshold,
                        max_frames=int(max_frames),
                        annotate=annotate
                    )

                    if result_path and os.path.exists(result_path):
                        output_placeholder.success("✅ PDD Generated!")
                        status_placeholder.empty()

                        with open(result_path, "rb") as f:
                            doc_bytes = f.read()

                        download_placeholder.download_button(
                            label="📥 Download PDD Document",
                            data=doc_bytes,
                            file_name=os.path.basename(result_path),
                            mime=(
                                "application/vnd.openxmlformats-"
                                "officedocument.wordprocessingml.document"
                            ),
                            use_container_width=True
                        )

                        flowchart_path = os.path.join(output_dir, "flowchart.png")
                        if os.path.exists(flowchart_path):
                            flowchart_placeholder.image(
                                flowchart_path, caption="Process Flowchart"
                            )

                        st.info(f"📁 Also saved to: `{path_config.output_dir}/`")
                    else:
                        output_placeholder.error("Failed to generate PDD.")

                except Exception as e:
                    output_placeholder.error(f"Error: {str(e)}")
                    status_placeholder.empty()
                    st.exception(e)


if __name__ == "__main__":
    main()