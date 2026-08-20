from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables or .env file.

    All fields have stable defaults for the FastAPI and CLI analysis paths.
    arguments so the app can start with zero configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="BASKETBALL_",
        case_sensitive=False,
    )

    # --- Model checkpoint ---
    model_path: str = "model_checkpoints/r2plus1d_v3/"
    base_model_name: str = "best"
    start_epoch: int = 0
    lr: float = 0.0001
    num_classes: int = 10

    # --- Video pipeline ---
    seq_length: int = 16
    vid_stride: int = 8
    action_vid_stride: int = 24
    batch_size: int = 8
    r2plus1d_device: str = "mps_if_available"
    yolo_device: str = "cpu"
    tracking_fps: float = 8.0
    yolo_imgsz: int = 320
    max_players_per_segment: int = 12
    torch_num_threads: int = 10
    progress_log: bool = True
    analysis_timeout_sec: float = 0.0
    tracker_type: str = "YOLO"  # CSRT | YOLO
    tracker_backend: str = "bytetrack"  # bytetrack | botsort | custom
    yolo_tracker_config: str = ""
    yolo_reid_enabled: bool = False
    yolo_reid_model: str = "auto"
    identity_embedding_backend: str = "torchvision_mobilenet_v3_small"
    identity_embedding_weights: str = "default"
    identity_embedding_device: str = "mps_if_available"
    identity_embedding_batch_size: int = 16
    identity_embedding_allow_fallback: bool = True
    face_identity_backend: str = "opencv_sface_if_available"
    face_detection_model_path: str = "model_checkpoints/opencv_face/face_detection_yunet_2023mar.onnx"
    face_recognition_model_path: str = "model_checkpoints/opencv_face/face_recognition_sface_2021dec.onnx"
    face_detection_score_threshold: float = 0.60
    face_identity_allow_fallback: bool = True
    face_gallery_path: str = ""
    face_gallery_similarity_threshold: float = 0.45
    face_gallery_minimum_margin: float = 0.08
    face_enrolled_minimum_quality: float = 0.65
    face_identity_match_threshold: float = 0.45
    face_identity_conflict_threshold: float = 0.30
    yolo_model_name: str = "model_checkpoints/yolov8n.pt"
    default_video: str = "examples/lebron_shoots.mp4"

    # --- Official event perception (experimental, opt-in) ---
    official_stats_enabled: bool = False
    official_detector_backend: str = "off"  # off | ultralytics_yolo | transformers_rfdetr
    official_detector_model_path: str = ""
    official_detector_device: str = "cpu"
    official_detector_imgsz: int = 704
    official_detector_confidence: float = 0.10
    official_player_tracking_enabled: bool = True
    official_player_tracker_config: str = "bytetrack.yaml"
    official_ball_max_distance_px: float = 90.0
    official_ball_max_gap_frames: int = 4
    official_court_max_reprojection_error_px: float = 8.0
    official_shot_candidate_confidence: float = 0.60
    official_rebound_suppression_make_confidence: float = 0.50
    official_stats_ruleset: str = "conservative-amateur-v1"
    official_vlm_enabled: bool = False
    official_vlm_confidence: float = 0.80
    official_vlm_frames: int = 12
    official_vlm_image_width: int = 512
    official_vlm_contact_sheet: bool = False
    official_vlm_context_length: int = 16384
    official_vlm_timeout: float = 180.0
    official_action_owner_model_path: str = ""
    official_audio_asr_enabled: bool = False
    official_audio_asr_model: str = ""
    official_audio_asr_language: str = "en"

    # --- VLM (Ollama) ---
    vlm_mode: str = "low-confidence"  # off | low-confidence | always
    ollama_model: str = "qwen3-vl:4b"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_timeout: float = 45.0
    scoreboard_ocr_backend: str = "rapidocr_if_available"
    scoreboard_ocr_confidence: float = 0.75
    vlm_frames: int = 1
    vlm_image_width: int = 224
    max_vlm_clips: int = 8
    jersey_number_vlm_enabled: bool = False
    jersey_number_vlm_frames: int = 2
    vlm_identity_merge_enabled: bool = False
    vlm_identity_merge_max_candidates: int = 8
    vlm_identity_merge_confidence: float = 0.78
    vlm_identity_merge_crops_per_side: int = 3

    # --- Confidence thresholds ---
    low_confidence: float = 0.45
    high_confidence: float = 0.70
    smoothing_confidence: float = 0.6

    # --- Output directories ---
    output_dir: str = "analysis_outputs"
    video_output_dir: str = "output_videos"
    allowed_video_roots: str = ""

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8765


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
