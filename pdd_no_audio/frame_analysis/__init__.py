# pdd_no_audio/frame_analysis/__init__.py

"""
Frame analysis modules for PDD Agent.
"""

from pdd_no_audio.frame_analysis.ocr_engine import ocr_frame, ocr_batch
from pdd_no_audio.frame_analysis.vision_describer import (
    analyze_transitions_smart, identify_application
)
from pdd_no_audio.frame_analysis.change_detector import detect_changes_between_frames
from pdd_no_audio.frame_analysis.frame_annotator import annotate_frame