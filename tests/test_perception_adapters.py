from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.analysis.perception import (
    PerceptionAdapterRegistry,
    RFDETRDetectorAdapter,
    TransformersRFDETRDetectorAdapter,
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


class _RFDetections:
    xyxy = _Array([[1, 2, 11, 12], [2, 3, 9, 10], [0, 0, 4, 4]])
    confidence = _Array([0.91, 0.82, 0.99])
    class_id = _Array([0, 1, 4])


class _RFDETRFakeModel:
    def __init__(self):
        self.model = SimpleNamespace(class_names=None)
        self.frames = []
        self.kwargs = {}

    def predict(self, frames, **kwargs):
        self.frames = frames
        self.kwargs = kwargs
        return [_RFDetections()]


class _TransformersBatch(dict):
    def __init__(self):
        super().__init__(pixel_values=np.zeros((2, 3, 384, 384), dtype="float32"))
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _TransformersRFDETRFakeProcessor:
    def __init__(self):
        self.images = []
        self.return_tensors = None
        self.target_sizes = None
        self.threshold = None
        self.batch = _TransformersBatch()

    def __call__(self, *, images, return_tensors):
        self.images = images
        self.return_tensors = return_tensors
        return self.batch

    def post_process_object_detection(self, outputs, *, target_sizes, threshold):
        self.target_sizes = target_sizes
        self.threshold = threshold
        return [
            {
                "boxes": _Array([[1, 2, 11, 12], [2, 3, 9, 10]]),
                "scores": _Array([0.91, 0.82]),
                "labels": _Array([0, 3]),
            },
            {
                "boxes": _Array([[4, 5, 14, 15]]),
                "scores": _Array([0.73]),
                "labels": _Array([1]),
            },
        ]


class _TransformersRFDETRFakeModel:
    config = SimpleNamespace(id2label={0: "ball", 1: "player", 3: "scoreboard"})

    def __init__(self):
        self.device = None
        self.eval_called = False
        self.inputs = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **inputs):
        self.inputs = inputs
        return SimpleNamespace()


def test_ultralytics_adapter_normalizes_supported_basketball_classes() -> None:
    adapter = UltralyticsDetectorAdapter(model_path="unused.pt", model=_FakeModel())
    detections = list(adapter.detect([object()], [12]))

    assert [item.object_type for item in detections] == ["player", "basketball"]
    assert all(item.frame == 12 and item.backend == "ultralytics_yolo" for item in detections)


def test_rfdetr_adapter_converts_bgr_and_drops_background_class() -> None:
    model = _RFDETRFakeModel()
    adapter = RFDETRDetectorAdapter(
        model_path="unused.pth",
        model=model,
        confidence=0.2,
        image_size=704,
    )
    frame = np.array([[[10, 20, 30]]], dtype="uint8")

    detections = list(adapter.detect([frame], [17]))

    assert model.frames[0].tolist() == [[[30, 20, 10]]]
    assert model.kwargs == {
        "threshold": 0.2,
        "shape": (704, 704),
        "include_source_image": False,
    }
    assert model.model.class_names == ["basketball", "rim", "player", "referee"]
    assert [item.object_type for item in detections] == ["basketball", "rim"]
    assert all(item.frame == 17 and item.backend == "rfdetr" for item in detections)


def test_transformers_rfdetr_adapter_batches_rgb_and_normalizes_ball_class() -> None:
    model = _TransformersRFDETRFakeModel()
    processor = _TransformersRFDETRFakeProcessor()
    adapter = TransformersRFDETRDetectorAdapter(
        model_path="unused-local-model",
        model=model,
        processor=processor,
        confidence=0.2,
        device="mps",
    )
    frames = [
        np.array([[[10, 20, 30]]], dtype="uint8"),
        np.array([[[40, 50, 60]]], dtype="uint8"),
    ]

    detections = list(adapter.detect(frames, [17, 18]))

    assert processor.images[0].tolist() == [[[30, 20, 10]]]
    assert processor.images[1].tolist() == [[[60, 50, 40]]]
    assert processor.return_tensors == "pt"
    assert processor.batch.device == "mps"
    assert processor.threshold == 0.2
    assert processor.target_sizes.tolist() == [[1, 1], [1, 1]]
    assert model.device == "mps"
    assert model.eval_called is True
    assert [item.object_type for item in detections] == ["basketball", "player"]
    assert [item.frame for item in detections] == [17, 18]
    assert all(item.backend == "transformers_rfdetr" for item in detections)


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


def test_official_detector_factory_builds_transformers_rfdetr_offline_backend() -> None:
    settings = Settings(
        _env_file=None,
        official_detector_backend="transformers_rfdetr",
        official_detector_model_path="/models/basketball-rfdetr",
        official_detector_device="mps",
        official_detector_confidence=0.15,
    )

    detector = build_official_detector(settings)

    assert isinstance(detector, TransformersRFDETRDetectorAdapter)
    assert detector.model_path == "/models/basketball-rfdetr"
    assert detector.device == "mps"
    assert detector.confidence == 0.15


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


def test_attach_pose_keypoints_uses_nearby_sparse_frame_index() -> None:
    player = PerceptionDetectionResponse(
        detection_id="player",
        frame=100,
        object_type="player",
        bbox=BoundingBoxResponse(x1=0, y1=0, x2=20, y2=40),
        confidence=0.9,
        player_id="tracked-player",
        backend="detector",
    )
    pose = list(
        UltralyticsPoseAdapter(model_path="unused.pt", model=_FakePoseModel()).estimate(
            [object()], [102]
        )
    )[0]
    far_pose = pose.model_copy(update={"frame": 1, "detection_id": "far"})

    merged = attach_pose_keypoints(
        [player], [far_pose, pose], maximum_frame_gap=2, minimum_iou=0.5
    )

    assert merged[0].player_id == "tracked-player"
    assert merged[0].keypoints["left_elbow"].x == 8.0
