# main.py
"""
Main entry point for PDD Agent.
Provides command-line interface for video processing.
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pdd_agent import generate_pdd_from_video
from llm_client import llm_client


def main():
    parser = argparse.ArgumentParser(
        description="PDD Agent - Convert meeting videos to Process Definition Documents"
    )
    
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to the video file to process"
    )
    
    parser.add_argument(
        "-n", "--name",
        help="Project name (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="./outputs",
        help="Output directory (default: ./outputs)"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if Ollama LLM is available"
    )
    
    args = parser.parse_args()
    
    # Check LLM availability
    if args.check:
        if llm_client.is_available():
            print("✓ Ollama LLM is available and connected.")
            return 0
        else:
            print("✗ Ollama LLM is not available.")
            print("  Ensure Ollama is running and accessible at the configured address.")
            return 1
    
    # Require video argument for processing
    if not args.video:
        parser.print_help()
        return 1
    
    # Check video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Check LLM before processing
    if not llm_client.is_available():
        print("Error: Ollama LLM is not available. Cannot process video.")
        print("Please ensure Ollama is running and try again.")
        return 1
    
    # Process video
    result = generate_pdd_from_video(
        args.video,
        args.name,
        args.output
    )
    
    if result:
        print(f"\nPDD document generated: {result}")
        return 0
    else:
        print("\nFailed to generate PDD document.")
        return 1


if __name__ == "__main__":
    sys.exit(main())