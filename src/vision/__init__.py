"""VeloVision: nucleo de vision (video monocular -> punto de soltura -> arm slot)."""
from .pose_slot import (
    PoseModelUnavailable,
    PoseSequence,
    annotate_frame,
    combined_slot_error_deg,
    detect_release_frame,
    extract_pose_sequence,
    measure_release,
)

__all__ = [
    "PoseModelUnavailable",
    "PoseSequence",
    "annotate_frame",
    "combined_slot_error_deg",
    "detect_release_frame",
    "extract_pose_sequence",
    "measure_release",
]
