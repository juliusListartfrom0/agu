from __future__ import annotations

import math
from collections.abc import Iterable

from app.analysis.schemas import (
    BallTrackPointResponse,
    BallTrackResponse,
    PerceptionDetectionResponse,
    Point2DResponse,
)


class BallTracker:
    """Small-object trajectory linker with bounded constant-velocity gaps."""

    def __init__(self, *, max_distance_px: float = 90.0, max_gap_frames: int = 4) -> None:
        if max_distance_px <= 0 or max_gap_frames < 0:
            raise ValueError("invalid ball tracker thresholds")
        self.max_distance_px = max_distance_px
        self.max_gap_frames = max_gap_frames

    def track(self, detections: Iterable[PerceptionDetectionResponse]) -> list[BallTrackResponse]:
        balls = sorted(
            (item for item in detections if item.object_type == "basketball"),
            key=lambda item: (item.frame, -item.confidence),
        )
        by_frame: dict[int, list[PerceptionDetectionResponse]] = {}
        for detection in balls:
            by_frame.setdefault(detection.frame, []).append(detection)

        active: list[BallTrackResponse] = []
        finished: list[BallTrackResponse] = []
        next_id = 1
        for frame in sorted(by_frame):
            active, expired = self._expire(active, frame)
            finished.extend(expired)
            candidates = list(by_frame[frame])
            assignments: list[tuple[float, int, int]] = []
            for track_index, track in enumerate(active):
                expected = _predict(track, frame)
                for detection_index, detection in enumerate(candidates):
                    center = _center(detection)
                    distance = math.hypot(center.x - expected.x, center.y - expected.y)
                    if distance <= self.max_distance_px:
                        assignments.append((distance, track_index, detection_index))
            used_tracks: set[int] = set()
            used_detections: set[int] = set()
            for _, track_index, detection_index in sorted(assignments):
                if track_index in used_tracks or detection_index in used_detections:
                    continue
                _append_detection(active[track_index], candidates[detection_index])
                used_tracks.add(track_index)
                used_detections.add(detection_index)

            for detection_index, detection in enumerate(candidates):
                if detection_index in used_detections:
                    continue
                track = BallTrackResponse(track_id=f"ball_{next_id:04d}")
                next_id += 1
                _append_detection(track, detection)
                active.append(track)
        finished.extend(active)
        return sorted(finished, key=lambda track: track.points[0].frame if track.points else 0)

    def interpolate(self, track: BallTrackResponse) -> BallTrackResponse:
        if len(track.points) < 2:
            return track
        output: list[BallTrackPointResponse] = []
        for left, right in zip(track.points, track.points[1:]):
            output.append(left)
            gap = right.frame - left.frame
            if 1 < gap <= self.max_gap_frames + 1:
                for offset in range(1, gap):
                    ratio = offset / gap
                    output.append(
                        BallTrackPointResponse(
                            frame=left.frame + offset,
                            center=Point2DResponse(
                                x=left.center.x + (right.center.x - left.center.x) * ratio,
                                y=left.center.y + (right.center.y - left.center.y) * ratio,
                            ),
                            confidence=min(left.confidence, right.confidence) * 0.7,
                            visible=False,
                            predicted=True,
                        )
                    )
        output.append(track.points[-1])
        return track.model_copy(update={"points": output})

    def _expire(
        self, tracks: list[BallTrackResponse], frame: int
    ) -> tuple[list[BallTrackResponse], list[BallTrackResponse]]:
        active: list[BallTrackResponse] = []
        expired: list[BallTrackResponse] = []
        for track in tracks:
            if frame - track.points[-1].frame > self.max_gap_frames + 1:
                expired.append(track)
            else:
                active.append(track)
        return active, expired


def _center(detection: PerceptionDetectionResponse) -> Point2DResponse:
    return Point2DResponse(
        x=(detection.bbox.x1 + detection.bbox.x2) / 2.0,
        y=(detection.bbox.y1 + detection.bbox.y2) / 2.0,
    )


def _predict(track: BallTrackResponse, frame: int) -> Point2DResponse:
    last = track.points[-1]
    if len(track.points) < 2:
        return last.center
    previous = track.points[-2]
    elapsed = max(1, last.frame - previous.frame)
    ahead = frame - last.frame
    return Point2DResponse(
        x=last.center.x + (last.center.x - previous.center.x) / elapsed * ahead,
        y=last.center.y + (last.center.y - previous.center.y) / elapsed * ahead,
    )


def _append_detection(track: BallTrackResponse, detection: PerceptionDetectionResponse) -> None:
    track.points.append(
        BallTrackPointResponse(
            frame=detection.frame,
            center=_center(detection),
            confidence=detection.confidence,
        )
    )
