# frontend/app.py

"""
Streamlit Frontend for PDD Agent.
Provides web interface for video upload and PDD generation.
"""

import streamlit as st
import os
import sys
import tempfile

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pdd_agent import PDDAgent
from llm_client import llm_client


def main():
    st.set_page_config(
        page_title="PDD Agent",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Meeting to PDD Agent")
    st.markdown("Convert meeting recordings into Process Definition Documents automatically.")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Status
        if llm_client.is_available():
            st.success("✓ Ollama LLM Connected")
        else:
            st.error("✗ Ollama LLM Not Available")
            st.info("Ensure Ollama is running at the configured address.")
        
        st.markdown("---")
        
        # Whisper model selection
        whisper_model = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium", "large", "large-v2"],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This agent:
        1. Extracts audio from video
        2. Transcribes using Whisper
        3. Generates process summary
        4. Creates flowchart
        5. Extracts key frames
        6. Produces PDD document
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Input")
        
        project_name = st.text_input(
            "Project Name",
            placeholder="Enter project name (optional - will be auto-detected)",
            help="Leave empty to auto-detect from meeting content"
        )
        
        video_file = st.file_uploader(
            "Upload Meeting Video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Upload the meeting recording to process"
        )
        
        if video_file:
            st.video(video_file)
    
    with col2:
        st.header("📊 Output")
        output_placeholder = st.empty()
        progress_placeholder = st.empty()
        download_placeholder = st.empty()
    
    # Process button
    if st.button("🚀 Generate PDD", type="primary", use_container_width=True):
        if not video_file:
            st.error("Please upload a video file first.")
            return
        
        if not llm_client.is_available():
            st.error("Cannot process: Ollama LLM is not available.")
            return
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video
            video_path = os.path.join(temp_dir, video_file.name)
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            
            # Initialize agent
            output_dir = os.path.join(temp_dir, "output")
            agent = PDDAgent(output_dir)
            
            # Process with progress updates
            with progress_placeholder:
                with st.spinner("Processing video... This may take several minutes."):
                    try:
                        result_path = agent.process_video(
                            video_path,
                            project_name if project_name else None,
                            whisper_model
                        )
                        
                        if result_path and os.path.exists(result_path):
                            output_placeholder.success("✓ PDD Document Generated Successfully!")
                            
                            # Read file for download
                            with open(result_path, "rb") as f:
                                doc_bytes = f.read()
                            
                            download_placeholder.download_button(
                                label="📥 Download PDD Document",
                                data=doc_bytes,
                                file_name=os.path.basename(result_path),
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True
                            )
                            
                            # Show flowchart if exists
                            flowchart_path = os.path.join(output_dir, "flowchart.png")
                            if os.path.exists(flowchart_path):
                                st.image(flowchart_path, caption="Generated Flowchart")
                        else:
                            output_placeholder.error("Failed to generate PDD document.")
                            
                    except Exception as e:
                        output_placeholder.error(f"Error: {str(e)}")
                        st.exception(e)


if __name__ == "__main__":
    main()