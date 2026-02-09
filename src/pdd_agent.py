# src/pdd_agent.py

"""
PDD Agent - Main Orchestrator.
Converts meeting videos into Process Definition Documents.
"""

import os
import time
from typing import Optional, List, Tuple, Dict

from config import path_config, whisper_config
from llm_client import llm_client
from video_to_audio import convert_video_to_audio
from transcribe_audio import transcribe_audio, read_transcript
from llm_tasks import (
    extract_entities,
    get_process_summary,
    get_project_name,
    get_inputs_outputs,
    get_document_purpose,
    get_applications_table,
    generate_dot_code,
    identify_key_timestamps,
    paraphrase_batch
)
from flowchart_generator import generate_flowchart_from_dot
from frame_extractor import extract_frames_with_transcripts, extract_frame
from pdd_document import PDDGenerator


class PDDAgent:
    """Agent that converts meeting videos to PDD documents."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the PDD Agent."""
        self.output_dir = output_dir or path_config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not llm_client.is_available():
            print("⚠ Warning: Ollama LLM is not available. Some features may fail.")
    
    def _extract_smart_frames(
        self,
        video_path: str,
        transcript_path: str,
        transcript: str,
        frames_dir: str
    ) -> List[Tuple[str, str]]:
        """
        Extract frames at meaningful process moments.
        Uses LLM to identify key action points instead of simple keyword matching.
        """
        print("    Identifying key process moments...")
        
        # Get LLM-identified key moments
        key_moments = identify_key_timestamps(transcript, transcript_path)
        
        if not key_moments:
            print("    Falling back to keyword-based extraction...")
            return extract_frames_with_transcripts(
                video_path, transcript_path, frames_dir
            )
        
        print(f"    Found {len(key_moments)} key process moments")
        
        # Extract frames at identified moments
        frame_pairs = []
        
        for moment in key_moments:
            timestamp = moment["timestamp"]
            description = moment.get("description", "Process step")
            
            frame_path = extract_frame(video_path, timestamp, frames_dir)
            if frame_path:
                frame_pairs.append((frame_path, description))
        
        return frame_pairs
    
    def process_video(
        self,
        video_path: str,
        project_name: str = None,
        whisper_model: str = None
    ) -> Optional[str]:
        """
        Process a video file and generate PDD document.
        """
        print("=" * 60)
        print("PDD Agent - Processing Video")
        print("=" * 60)
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return None
        
        # Step 1: Extract audio
        print("\n[Step 1/6] Extracting audio from video...")
        audio_path = convert_video_to_audio(video_path, self.output_dir)
        if not audio_path:
            print("Failed to extract audio. Aborting.")
            return None
        
        # Step 2: Transcribe audio
        print("\n[Step 2/6] Transcribing audio...")
        whisper_model = whisper_model or whisper_config.model_name
        transcript_path = transcribe_audio(
            audio_path,
            self.output_dir,
            model_name=whisper_model
        )
        if not transcript_path:
            print("Failed to transcribe audio. Aborting.")
            return None
        
        transcript = read_transcript(transcript_path)
        if not transcript:
            print("Failed to read transcript. Aborting.")
            return None
        
        print(f"  Transcript length: {len(transcript):,} characters")
        
        # Step 3: Extract information using LLM
        print("\n[Step 3/6] Extracting process information (LLM)...")
        
        # Extract entities first (for correct spelling)
        print("  - Extracting entities...")
        entities = extract_entities(transcript)
        if entities["companies"]:
            print(f"    Companies: {', '.join(entities['companies'])}")
        if entities["applications"]:
            print(f"    Applications: {', '.join(entities['applications'])}")
        
        # Get project name
        if not project_name:
            print("  - Detecting project name...")
            project_name = get_project_name(transcript)
        print(f"  ✓ Project Name: {project_name}")
        
        # Get document purpose
        print("  - Generating document purpose...")
        document_purpose = get_document_purpose(transcript, project_name)
        print(f"    ✓ Document purpose generated")
        
        # Get process summary
        print("  - Generating process summary...")
        process_summary = get_process_summary(transcript, entities)
        print(f"    ✓ Summary: {len(process_summary)} chars")
        
        # Get inputs/outputs
        print("  - Extracting inputs and outputs...")
        inputs_outputs = get_inputs_outputs(transcript, entities)
        print(f"    ✓ I/O extracted")
        
        # Get applications table
        print("  - Extracting applications used...")
        applications = get_applications_table(transcript, entities)
        print(f"    ✓ Found {len(applications)} applications")
        
        # Step 4: Generate flowchart
        print("\n[Step 4/6] Generating flowchart...")
        dot_code = generate_dot_code(transcript, entities)
        
        flowchart_path = os.path.join(self.output_dir, "flowchart")
        if dot_code:
            print("  - Rendering flowchart...")
            flowchart_result = generate_flowchart_from_dot(dot_code, flowchart_path)
            flowchart_path = flowchart_result if flowchart_result else ""
            if flowchart_result:
                print(f"    ✓ Flowchart saved")
        else:
            print("  ⚠ Could not generate flowchart")
            flowchart_path = ""
        
        # Step 5: Extract key frames (smart extraction)
        print("\n[Step 5/6] Extracting process screenshots...")
        frames_dir = os.path.join(self.output_dir, "frames")
        
        frame_pairs = self._extract_smart_frames(
            video_path,
            transcript_path,
            transcript,
            frames_dir
        )
        
        # Paraphrase frame descriptions
        if frame_pairs:
            print(f"  - Improving {len(frame_pairs)} step descriptions...")
            texts = [text for _, text in frame_pairs]
            improved_texts = paraphrase_batch(texts)
            
            frame_pairs = [
                (frame_pairs[i][0], improved_texts[i] if i < len(improved_texts) else frame_pairs[i][1])
                for i in range(len(frame_pairs))
            ]
            print(f"    ✓ Descriptions improved")
        
        # Step 6: Generate PDD document
        print("\n[Step 6/6] Generating PDD document...")
        
        # Clean project name for filename
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name[:50] if safe_name else "PDD"
        doc_path = os.path.join(self.output_dir, f"{safe_name}_PDD.docx")
        
        generator = PDDGenerator()
        generator.generate(
            project_name=project_name,
            process_summary=process_summary,
            inputs_outputs=inputs_outputs,
            flowchart_path=flowchart_path,
            document_purpose=document_purpose,
            applications=applications,
            output_path=doc_path
        )
        
        # Append frames to document
        if frame_pairs:
            print(f"  - Adding {len(frame_pairs)} process steps with screenshots...")
            generator.append_frames_with_text(doc_path, frame_pairs)
        
        print("\n" + "=" * 60)
        print("✓ PDD Generation Complete!")
        print(f"  Output: {doc_path}")
        print("=" * 60)
        
        return doc_path


def generate_pdd_from_video(
    video_path: str,
    project_name: str = None,
    output_dir: str = None
) -> Optional[str]:
    """Convenience function to generate PDD from video."""
    agent = PDDAgent(output_dir)
    return agent.process_video(video_path, project_name)


if __name__ == "__main__":
    print("PDD Agent - Meeting to PDD Converter")
    print("-" * 40)
    
    video_path = input("Enter video file path: ").strip()
    project_name = input("Enter project name (Enter to auto-detect): ").strip() or None
    
    result = generate_pdd_from_video(video_path, project_name)
    
    if result:
        print(f"\n✓ Success! Document: {result}")
    else:
        print("\n✗ Failed to generate document.")