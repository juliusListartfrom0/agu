from __future__ import annotations

from types import SimpleNamespace

from app.analysis.perception import (
    PerceptionAdapterRegistry,
    UltralyticsDetectorAdapter,
    UltralyticsPoseAdapter,
    attach_pose_keypoints,
    build_official_detector,
)
from app.analysis.schemas import BoundingBoxResponse, PerceptionDetectionResponse
from app.config import Settings


class _Array:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _FakeModel:
    names = {0: "person", 32: "sports ball", 99: "chair"}

    def predict(self, **kwargs):
        return [
            SimpleNamespace(
                names=self.names,
                boxes=SimpleNamespace(
                    xyxy=_Array([[0, 0, 20, 40], [30, 10, 36, 16], [5, 5, 10, 10]]),
                    conf=_Array([0.9, 0.8, 0.99]),
                    cls=_Array([0, 32, 99]),
                ),
            )
        ]


class _TrackedFakeModel(_FakeModel):
    def track(self, **kwargs):
        result = self.predict(**kwargs)[0]
        result.boxes.id = _Array([7, 8, 9])
        return [result]


class _FakePoseModel:
    def predict(self, **kwargs):
        points = [[float(index + 1), float(index + 2)] for index in range(17)]
        confidence = [0.9] * 17
        confidence[9] = 0.1
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=_Array([[1, 2, 21, 42]]),
                    conf=_Array([0.88]),
                ),
                keypoints=SimpleNamespace(
                    xy=_Array([points]),
                    conf=_Array([confidence]),
                ),
            )
        ]


class _PromptFakeModel(_FakeModel):
    names = {0: "basketball rim"}

    def __init__(self):
        self.prompts = []

    def set_classes(self, prompts):
        self.prompts.append(prompts)

    def predict(self, **kwargs):
        return [
            SimpleNamespace(
                names=self.names,
                boxes=SimpleNamespace(
                    xyxy=_Array([[10, 20, 30, 24]]),
                    conf=_Array([0.7]),
                    cls=_Array([0]),
                ),
            )
        ]


def test_ultralytics_adapter_normalizes_supported_basketball_classes() -> None:
    adapter = UltralyticsDetectorAdapter(model_path="unused.pt", model=_FakeModel())
    detections = list(adapter.detect([object()], [12]))

    assert [item.object_type for item in detections] == ["player", "basketball"]
    assert all(item.frame == 12 and item.backend == "ultralytics_yolo" for item in detections)


def test_ultralytics_adapter_configures_open_vocabulary_prompts_once() -> None:
    model = _PromptFakeModel()
    adapter = UltralyticsDetectorAdapter(
        model_path="unused.pt",
        model=model,
        class_prompts=("basketball hoop",),
    )

    first = list(adapter.detect([object()], [12]))
    second = list(adapter.detect([object()], [13]))

    assert model.prompts == [["basketball hoop"]]
    assert [item.object_type for item in first + second] == ["rim", "rim"]


def test_perception_registry_reports_and_resolves_available_adapter() -> None:
    adapter = UltralyticsDetectorAdapter(model_path="unused.pt", model=_FakeModel())
    registry = PerceptionAdapterRegistry()
    registry.register(adapter)

    assert registry.diagnostics() == {"ultralytics_yolo": True}
    assert registry.get("ultralytics_yolo") is adapter


def test_ultralytics_tracking_exposes_stable_raw_player_ids() -> None:
    adapter = UltralyticsDetectorAdapter(
        model_path="unused.pt", model=_TrackedFakeModel(), tracking=True
    )
    detections = list(adapter.detect([object()], [12]))

    assert detections[0].track_id == "track-7"
    assert detections[0].player_id == "raw-player-track-7"
    assert detections[0].detection_id == "ultralytics_yolo:12:0"


def test_official_detector_factory_is_off_by_default_and_requires_model_path() -> None:
    assert build_official_detector(Settings(_env_file=None)) is None

    settings = Settings(_env_file=None, official_detector_backend="ultralytics_yolo")
    try:
        build_official_detector(settings)
    except ValueError as exc:
        assert "official_detector_model_path" in str(exc)
    else:
        raise AssertionError("missing model path should fail")


def test_ultralytics_pose_adapter_normalizes_coco_keypoints() -> None:
    adapter = UltralyticsPoseAdapter(
        model_path="unused.pt",
        model=_FakePoseModel(),
        keypoint_confidence=0.2,
    )

    detections = list(adapter.estimate([object()], [30]))

    assert len(detections) == 1
    assert detections[0].frame == 30
    assert detections[0].backend == "ultralytics_pose"
    assert detections[0].keypoints["left_shoulder"].x == 6.0
    assert "left_wrist" not in detections[0].keypoints
    assert detections[0].keypoints["right_wrist"].y == 12.0


def test_attach_pose_keypoints_preserves_detector_identity() -> None:
    player = PerceptionDetectionResponse(
        detection_id="detector:12:0",
        frame=12,
        object_type="player",
        bbox=BoundingBoxResponse(x1=0, y1=0, x2=20, y2=40),
        confidence=0.9,
        track_id="track-7",
        player_id="raw-player-track-7",
        team_id="raw-dark",
        backend="detector",
    )
    pose = list(
        UltralyticsPoseAdapter(model_path="unused.pt", model=_FakePoseModel()).estimate(
            [object()], [12]
        )
    )[0]

    merged = attach_pose_keypoints([player], [pose], minimum_iou=0.5)

    assert merged[0].player_id == player.player_id
    assert merged[0].team_id == player.team_id
    assert merged[0].backend == player.backend
    assert merged[0].keypoints["left_elbow"].x == 8.0
