# main.py
"""
Main entry point for PDD Agent.
CLI interface for video and transcript processing.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pdd_agent import generate_pdd_from_video, generate_pdd_from_transcript
from llm_client import llm_client


def main():
    parser = argparse.ArgumentParser(
        description="PDD Agent - Convert meeting videos/transcripts to PDD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video meeting.mp4
  python main.py video meeting.mp4 -t transcript.txt
  python main.py transcript transcript.txt
  python main.py transcript transcript.txt -v meeting.mp4
  python main.py video meeting.mp4 -n "Ivanti Patching"
  python main.py --check
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Processing mode")
    
    # Video mode
    vp = subparsers.add_parser("video", help="Process video file")
    vp.add_argument("input_file", help="Video file path")
    vp.add_argument("-t", "--transcript", help="Existing transcript (skips Whisper)")
    vp.add_argument("-n", "--name", help="Project name (auto-detected if omitted)")
    vp.add_argument("-o", "--output", default="./outputs", help="Output directory")
    vp.add_argument("-w", "--whisper-model", default=None,
                    choices=["tiny", "base", "small", "medium", "large", "large-v2"])
    
    # Transcript mode
    tp = subparsers.add_parser("transcript", help="Process transcript file")
    tp.add_argument("input_file", help="Transcript file path")
    tp.add_argument("-v", "--video", help="Optional video for screenshots")
    tp.add_argument("-n", "--name", help="Project name (auto-detected if omitted)")
    tp.add_argument("-o", "--output", default="./outputs", help="Output directory")
    
    parser.add_argument("--check", action="store_true", help="Check LLM availability")
    
    args = parser.parse_args()
    
    if args.check:
        if llm_client.is_available():
            print(f"✓ Ollama connected")
            print(f"  Model: {llm_client.config.model}")
            print(f"  Server: {llm_client.config.base_url}")
            print(f"  Context: {llm_client.params.num_ctx}")
            print(f"  Timeout: {llm_client.params.timeout}s per call")
            info = llm_client.get_model_info()
            if info:
                details = info.get('details', {})
                print(f"  Family: {details.get('family', '?')}")
                print(f"  Parameters: {details.get('parameter_size', '?')}")
            return 0
        else:
            print(f"✗ Cannot connect to {llm_client.config.base_url}")
            return 1
    
    if not args.command:
        parser.print_help()
        return 1
    
    if not llm_client.is_available():
        print(f"Error: Ollama not available at {llm_client.config.base_url}")
        return 1
    
    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        return 1
    
    if args.command == "video":
        if args.transcript and not os.path.exists(args.transcript):
            print(f"Warning: Transcript not found: {args.transcript}, using Whisper")
            args.transcript = None
        
        result = generate_pdd_from_video(
            video_path=args.input_file,
            project_name=args.name,
            output_dir=args.output,
            transcript_path=args.transcript
        )
    
    elif args.command == "transcript":
        if args.video and not os.path.exists(args.video):
            print(f"Warning: Video not found: {args.video}, skipping screenshots")
            args.video = None
        
        result = generate_pdd_from_transcript(
            transcript_path=args.input_file,
            project_name=args.name,
            output_dir=args.output,
            video_path=args.video
        )
    
    else:
        parser.print_help()
        return 1
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())