"""Normalized contracts and thin adapters for basketball perception."""

from .adapters import DetectorAdapter, PerceptionAdapterRegistry, PoseAdapter
from .ball_tracking import BallTracker, GlobalBallPathSelector
from .factory import build_official_detector
from .pose import attach_pose_keypoints
from .rfdetr_adapter import RFDETRDetectorAdapter
from .transformers_rfdetr_adapter import TransformersRFDETRDetectorAdapter
from .ultralytics_adapter import UltralyticsDetectorAdapter
from .ultralytics_pose_adapter import UltralyticsPoseAdapter

__all__ = [
    "BallTracker",
    "GlobalBallPathSelector",
    "DetectorAdapter",
    "PerceptionAdapterRegistry",
    "PoseAdapter",
    "RFDETRDetectorAdapter",
    "TransformersRFDETRDetectorAdapter",
    "UltralyticsDetectorAdapter",
    "UltralyticsPoseAdapter",
    "attach_pose_keypoints",
    "build_official_detector",
]
