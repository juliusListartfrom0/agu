from __future__ import annotations

from collections.abc import Sequence

from app.analysis.schemas import BoundingBoxResponse, PerceptionDetectionResponse


def attach_pose_keypoints(
    detections: Sequence[PerceptionDetectionResponse],
    pose_detections: Sequence[PerceptionDetectionResponse],
    *,
    maximum_frame_gap: int = 0,
    minimum_iou: float = 0.25,
) -> list[PerceptionDetectionResponse]:
    """Attach pose points to identity-bearing player detections by geometry only."""

    if maximum_frame_gap < 0:
        raise ValueError("maximum_frame_gap must be non-negative")
    if not 0 <= minimum_iou <= 1:
        raise ValueError("minimum_iou must be between zero and one")
    poses = [item for item in pose_detections if item.object_type == "player" and item.keypoints]
    merged: list[PerceptionDetectionResponse] = []
    for detection in detections:
        if detection.object_type != "player":
            merged.append(detection)
            continue
        candidates = [
            pose
            for pose in poses
            if abs(pose.frame - detection.frame) <= maximum_frame_gap
        ]
        scored = [(_box_iou(detection.bbox, pose.bbox), pose) for pose in candidates]
        if not scored:
            merged.append(detection)
            continue
        overlap, pose = max(
            scored,
            key=lambda item: (item[0], item[1].confidence, item[1].detection_id),
        )
        if overlap < minimum_iou:
            merged.append(detection)
            continue
        merged.append(detection.model_copy(update={"keypoints": dict(pose.keypoints)}))
    return merged


def _box_iou(left: BoundingBoxResponse, right: BoundingBoxResponse) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0
