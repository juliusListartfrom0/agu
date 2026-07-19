from __future__ import annotations

import math
from dataclasses import dataclass

from app.analysis.schemas import BallTrackPointResponse, BoundingBoxResponse, CourtCalibrationResponse, Point2DResponse


@dataclass(frozen=True)
class ShotAssessment:
    outcome: str
    confidence: float
    outcome_frame: int | None
    evidence: tuple[str, ...]


def assess_shot_trajectory(
    points: list[BallTrackPointResponse],
    rim: BoundingBoxResponse,
    *,
    rim_margin_px: float = 8.0,
) -> ShotAssessment:
    """Classify made/missed/unknown from a time-ordered ball trajectory."""

    ordered = sorted(points, key=lambda point: point.frame)
    if len(ordered) < 2:
        return ShotAssessment("unknown", 0.0, None, ("insufficient_ball_track",))
    rim_y = (rim.y1 + rim.y2) / 2.0
    left = rim.x1 - rim_margin_px
    right = rim.x2 + rim_margin_px
    near_rim = False
    for previous, current in zip(ordered, ordered[1:]):
        if left <= current.center.x <= right and rim.y1 - rim_margin_px <= current.center.y <= rim.y2 + rim_margin_px:
            near_rim = True
        downward_crossing = previous.center.y < rim_y <= current.center.y
        crossing_x = (previous.center.x + current.center.x) / 2.0
        inside_cylinder = (
            rim.x1 <= previous.center.x <= rim.x2 and rim.x1 <= current.center.x <= rim.x2
        )
        if downward_crossing and inside_cylinder:
            confidence = min(previous.confidence, current.confidence)
            if previous.predicted or current.predicted:
                confidence *= 0.7
            return ShotAssessment(
                "made",
                max(0.5, confidence),
                current.frame,
                ("downward_rim_plane_crossing", f"crossing_x={crossing_x:.2f}"),
            )
    final = ordered[-1]
    descended_past_rim = final.center.y > rim.y2 + rim_margin_px
    outside_cylinder = not left <= final.center.x <= right
    if near_rim and descended_past_rim and outside_cylinder:
        confidence = min(0.95, max(0.5, sum(point.confidence for point in ordered) / len(ordered)))
        return ShotAssessment("missed", confidence, final.frame, ("rim_contact_without_crossing",))
    return ShotAssessment("unknown", 0.35 if near_rim else 0.1, None, ("outcome_not_observable",))


def classify_shot_value(
    release_image_point: Point2DResponse,
    calibration: CourtCalibrationResponse,
    *,
    basket_court_point: Point2DResponse = Point2DResponse(x=0.0, y=0.0),
    arc_radius_m: float = 6.75,
) -> tuple[int | None, float, str]:
    """Project release position into court coordinates and classify 2/3."""

    if calibration.status != "valid" or calibration.reprojection_error_px > 8.0:
        return None, 0.0, "court_calibration_not_valid"
    projected = _project(release_image_point, calibration.homography)
    if projected is None:
        return None, 0.0, "homography_projection_failed"
    distance = math.hypot(projected.x - basket_court_point.x, projected.y - basket_court_point.y)
    line_uncertainty_m = max(0.05, calibration.reprojection_error_px / 100.0)
    if abs(distance - arc_radius_m) <= line_uncertainty_m:
        return None, 0.4, f"release_near_three_point_line:{distance:.3f}m"
    value = 3 if distance > arc_radius_m else 2
    confidence = min(0.99, 0.75 + abs(distance - arc_radius_m) / 4.0)
    return value, confidence, f"projected_basket_distance:{distance:.3f}m"


def _project(point: Point2DResponse, matrix: list[list[float]]) -> Point2DResponse | None:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        return None
    denominator = matrix[2][0] * point.x + matrix[2][1] * point.y + matrix[2][2]
    if abs(denominator) < 1e-9:
        return None
    return Point2DResponse(
        x=(matrix[0][0] * point.x + matrix[0][1] * point.y + matrix[0][2]) / denominator,
        y=(matrix[1][0] * point.x + matrix[1][1] * point.y + matrix[1][2]) / denominator,
    )
