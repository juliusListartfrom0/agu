"""Normalized contracts and thin adapters for basketball perception."""

from .adapters import DetectorAdapter, PerceptionAdapterRegistry, PoseAdapter
from .ball_tracking import BallTracker
from .factory import build_official_detector
from .pose import attach_pose_keypoints
from .ultralytics_adapter import UltralyticsDetectorAdapter
from .ultralytics_pose_adapter import UltralyticsPoseAdapter

__all__ = [
    "BallTracker",
    "DetectorAdapter",
    "PerceptionAdapterRegistry",
    "PoseAdapter",
    "UltralyticsDetectorAdapter",
    "UltralyticsPoseAdapter",
    "attach_pose_keypoints",
    "build_official_detector",
]
