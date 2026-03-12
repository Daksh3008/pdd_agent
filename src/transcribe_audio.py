# src/transcribe_audio.py

"""
Audio transcription using OpenAI's Whisper model.
Generates timestamped transcript from audio files.
Auto-detects language when not specified.
"""

import os
from typing import Optional
import whisper
from config import whisper_config


def transcribe_audio(
    audio_path: str,
    output_dir: str = None,
    model_name: str = None,
    language: str = None,
    task: str = None
) -> Optional[str]:
    """
    Transcribe audio file using Whisper model.

    Language detection:
    - If language is None (default), Whisper auto-detects the language.
    - If detected language is English, task is set to "transcribe".
    - If detected language is non-English, task is set to "translate" (to English).
    - If language is explicitly set, uses that language and the specified task.

    Args:
        audio_path: Path to the audio file.
        output_dir: Directory to save transcript.
        model_name: Whisper model to use (tiny, base, small, medium, large).
        language: Source language code (None = auto-detect).
        task: 'transcribe' or 'translate' (None = auto-select).

    Returns:
        Path to the transcript file, or None if failed.
    """
    model_name = model_name or whisper_config.model_name
    language = language or whisper_config.language  # None = auto-detect
    task = task or whisper_config.task
    output_dir = output_dir or os.path.dirname(audio_path)

    # Generate transcript filename
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")

    # Skip if already exists
    if os.path.exists(transcript_path):
        print(f"Transcript already exists: {transcript_path}")
        return transcript_path

    try:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)

        # ── Auto-detect language if not specified ──
        if language is None:
            print("Detecting language...")
            # Load first 30 seconds for detection
            audio_data = whisper.load_audio(audio_path)
            audio_segment = whisper.pad_or_trim(audio_data)
            mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)
            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            print(f"Detected language: {detected_lang} (confidence: {confidence:.2%})")

            language = detected_lang

            # Auto-select task based on detected language
            if task == "transcribe":
                # User explicitly wants transcription, keep it
                pass
            elif detected_lang == "en":
                task = "transcribe"
                print("  → English detected, using transcribe mode")
            else:
                task = "translate"
                print(f"  → Non-English ({detected_lang}) detected, translating to English")
        else:
            print(f"Using specified language: {language}, task: {task}")

        # ── Transcribe ──
        print(f"Transcribing: {audio_path} (lang={language}, task={task})")
        result = model.transcribe(
            audio_path,
            word_timestamps=False,
            task=task,
            language=language
        )

        # Write transcript with timestamps
        with open(transcript_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()
                f.write(f"[{start:.2f} - {end:.2f}] {text}\n")

        print(f"Transcript saved to: {transcript_path}")
        return transcript_path

    except Exception as e:
        print(f"Transcription error: {e}")
        return None


def read_transcript(transcript_path: str) -> str:
    """
    Read transcript file content.

    Args:
        transcript_path: Path to transcript file.

    Returns:
        Transcript text content.
    """
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Transcript file not found: {transcript_path}")
        return ""
    except Exception as e:
        print(f"Error reading transcript: {e}")
        return ""


if __name__ == "__main__":
    test_audio = input("Enter audio path: ").strip()
    if os.path.exists(test_audio):
        result = transcribe_audio(test_audio)
        print(f"Result: {result}")
    else:
        print("Audio file not found.")