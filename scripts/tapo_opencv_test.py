import argparse
import atexit
import csv
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bowl-zone motion and cat detection from RTSP or a local video."
    )
    parser.add_argument(
        "--url",
        help="Explicit RTSP URL. Defaults to RTSP_URL from .env or environment.",
    )
    parser.add_argument(
        "--video",
        help=(
            "Path to a local video file for offline testing. "
            "When set, RTSP is ignored."
        ),
    )
    parser.add_argument(
        "--loop-video",
        action="store_true",
        help="Loop the local video file when it reaches the end.",
    )
    parser.add_argument(
        "--window-name",
        default="Tapo RTSP",
        help="Window title for the preview.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without creating an OpenCV preview window.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Initial preview window width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Initial preview window height in pixels.",
    )
    parser.add_argument(
        "--zone",
        default="0.35,0.45,0.3,0.3",
        help=(
            "Normalized bowl zone as x,y,w,h values between 0 and 1. "
            "Default: center-ish rectangle."
        ),
    )
    parser.add_argument(
        "--zone-polygon",
        help=(
            "Optional normalized polygon for bowl zone as 'x1,y1;x2,y2;...'. "
            "When set, this overrides --zone and supports skewed perspective zones."
        ),
    )
    parser.add_argument(
        "--zone-edit",
        action="store_true",
        help=(
            "Enable mouse editing for zone calibration in the preview window. "
            "Press 'p' to print the current zone value."
        ),
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=1.5,
        help="Motion percentage threshold inside the bowl zone.",
    )
    parser.add_argument(
        "--process-fps",
        type=float,
        default=1.0,
        help="Target processing rate. Use 1.0 for idle-mode polling.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="captures",
        help="Directory for motion-triggered snapshots, relative to the repo root.",
    )
    parser.add_argument(
        "--snapshot-cooldown",
        type=float,
        default=10.0,
        help="Minimum seconds between saved snapshots.",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Disable motion-start snapshot saving.",
    )
    parser.add_argument(
        "--identity-debug-csv",
        help=(
            "Optional CSV path for per-frame identity debug output. "
            "Relative paths are resolved from the repo root."
        ),
    )
    parser.add_argument(
        "--alert-seconds",
        type=float,
        default=4.0,
        help="Trigger an alert when zone activity lasts this many seconds.",
    )
    parser.add_argument(
        "--save-clip-on-alert",
        action="store_true",
        help="Save a short video clip when an alert is triggered.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=10.0,
        help="Clip duration in seconds after alert trigger.",
    )
    parser.add_argument(
        "--cat-model",
        help=(
            "Optional path to a YOLOv8 ONNX model. When set, cat detections "
            "inside the bowl zone become the alert condition."
        ),
    )
    parser.add_argument(
        "--cat-confidence",
        type=float,
        default=0.4,
        help="Minimum confidence for cat detections.",
    )
    parser.add_argument(
        "--cat-class-id",
        type=int,
        default=15,
        help="COCO class id for cat. Default is 15.",
    )
    parser.add_argument(
        "--cat-imgsz",
        type=int,
        default=640,
        help="Inference image size for Ultralytics .pt models.",
    )
    parser.add_argument(
        "--cat-preprocess",
        choices=("none", "night", "night-lite"),
        default="none",
        help="Optional preprocessing for cat inference frames.",
    )
    parser.add_argument(
        "--cat-enter-frames",
        type=int,
        default=2,
        help="Required consecutive cat-in-zone detections before presence starts.",
    )
    parser.add_argument(
        "--cat-hold-seconds",
        type=float,
        default=1.0,
        help="How long to keep cat presence alive after a missed detection.",
    )
    parser.add_argument(
        "--possible-goblin-seconds",
        type=float,
        default=2.0,
        help="Trigger a low-priority possible Goblin event after this much dwell.",
    )
    parser.add_argument(
        "--id-goblin-support-conf",
        type=float,
        default=0.72,
        help="Minimum frame-level identity confidence before a frame counts toward Goblin support.",
    )
    parser.add_argument(
        "--id-goblin-support-margin",
        type=float,
        default=0.18,
        help="Minimum Goblin-vs-Orange evidence margin for a frame to count toward Goblin support.",
    )
    parser.add_argument(
        "--id-goblin-torso-white-min",
        type=float,
        default=0.10,
        help="Minimum torso-core white ratio required for Goblin support.",
    )
    parser.add_argument(
        "--id-goblin-torso-black-blobs-min",
        type=int,
        default=1,
        help="Minimum torso-core black blob count required for Goblin support.",
    )
    parser.add_argument(
        "--id-goblin-periphery-margin-max",
        type=float,
        default=0.03,
        help=(
            "Maximum amount by which periphery Goblin evidence may exceed torso-core evidence "
            "before the frame is downgraded as background contamination."
        ),
    )
    parser.add_argument(
        "--id-lock-margin",
        type=float,
        default=0.11,
        help="Accumulator delta needed to lock an unknown identity to Orange or Goblin.",
    )
    parser.add_argument(
        "--id-switch-margin",
        type=float,
        default=0.24,
        help="Accumulator delta needed to switch a stable identity to the opposite class.",
    )
    parser.add_argument(
        "--id-orange-clear-streak",
        type=int,
        default=3,
        help="How many Orange-strong frames clear accumulated Goblin support.",
    )
    parser.add_argument(
        "--id-goblin-support-window",
        type=int,
        default=6,
        help="Sliding-window size for recent Goblin-support frames.",
    )
    parser.add_argument(
        "--id-confirmed-goblin-support-count",
        type=int,
        default=4,
        help="Required Goblin-support frames inside the sliding window before confirmed Goblin hold can start.",
    )
    parser.add_argument(
        "--id-confirmed-goblin-hold-seconds",
        type=float,
        default=1.0,
        help="How long confirmed Goblin evidence must hold before the confirmed_goblin state becomes active.",
    )
    parser.add_argument(
        "--cat-zone-overlap",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of cat box area that must overlap the zone "
            "(0 to 1). Use >0 to prevent slight edge-touch triggers."
        ),
    )
    parser.add_argument(
        "--cat-detect-mode",
        choices=("auto", "motion", "always"),
        default="always",
        help=(
            "When to run cat inference: auto (always for --video, motion for RTSP), "
            "motion (only while zone motion is active), or always (every processed frame)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference device for OpenCV DNN cat detection.",
    )
    parser.add_argument(
        "--launch-origin",
        default="manual",
        help=(
            "How this run was started (for APP_START logging), "
            "for example: manual, watchdog_ps1, watchdog_sh."
        ),
    )
    return parser.parse_args()


@dataclass
class RuntimeConfig:
    repo_root: Path
    rtsp_url: str | None
    video_path: Path | None
    is_video_source: bool
    zone_rect_edit: list[float]
    zone_polygon_edit: list[list[float]] | None
    snapshot_dir: Path
    event_log_dir: Path
    identity_debug_csv_path: Path | None
    cat_model_path: str | None
    launch_origin: str


@dataclass
class DetectorRuntime:
    detector: object | None
    backend: str
    device: str
    cat_detect_mode: str
    mode: str


@dataclass
class CaptureState:
    video_native_fps: float = 0.0
    video_skip_accumulator: float = 0.0
    rtsp_retrieve_failures: int = 0
    frame_interval: float = 0.0
    next_frame_at: float = 0.0


@dataclass
class ActivityState:
    motion_state: bool = False
    motion_started_at: float = 0.0
    last_snapshot_at: float = 0.0
    current_motion_snapshot_name: str = ""
    alert_fired: bool = False
    cat_present_state: bool = False
    cat_started_at: float = 0.0
    cat_seen_streak: int = 0
    cat_last_seen_at: float = 0.0
    current_dwell_subject: str = "Zone activity"


@dataclass
class IdentityPresenceState:
    seen_streak: int = 0
    last_seen_at: float = 0.0
    present: bool = False
    started_at: float = 0.0
    alert_fired: bool = False


@dataclass
class IdentityRuntimeState:
    orange_acc: float = 0.0
    goblin_acc: float = 0.0
    support_frames: int = 0
    stable_label: str = "unknown"
    stable_conf: float = 0.0
    goblin_support_history: deque[bool] = field(default_factory=deque)
    confirmed_goblin_hold_started_at: float = 0.0
    orange_stronger_streak: int = 0
    possible_goblin_alerted_this_session: bool = False
    presence: dict[str, IdentityPresenceState] = field(default_factory=dict)


@dataclass
class OutputRuntimeState:
    clip_writer: cv2.VideoWriter | None = None
    clip_end_at: float = 0.0
    active_clip_path: Path | None = None
    clip_dir: Path | None = None
    identity_debug_file: object | None = None
    identity_debug_writer: csv.DictWriter | None = None


@dataclass
class RunLifecycleState:
    started_at: float
    frame_count: int = 0
    reason: str = "normal_exit"
    ended: bool = False


@dataclass
class EditorState:
    frame_w: int = 0
    frame_h: int = 0
    dragging: bool = False
    drag_kind: str = ""
    start_mouse: tuple[int, int] = (0, 0)
    start_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    start_polygon: list[list[float]] = field(default_factory=list)
    active_point: int = -1


@dataclass
class RuntimeState:
    capture: CaptureState
    activity: ActivityState
    identity: IdentityRuntimeState
    output: OutputRuntimeState
    lifecycle: RunLifecycleState
    previous_gray: np.ndarray | None = None


@dataclass
class FrameObservation:
    x1: int
    y1: int
    x2: int
    y2: int
    zone_mask: np.ndarray
    zone_polygon_px: np.ndarray | None
    motion_percent: float
    motion_active: bool
    cat_detected_in_zone: bool
    cat_detections: list[tuple[tuple[int, int, int, int], float]]
    cat_identities: list[tuple[str, float]]
    cat_identity_sample_label: str
    cat_identity_sample_conf: float
    cat_identity_sample_orange: float
    cat_identity_sample_goblin: float
    cat_identity_sample_evidence: "IdentityEvidence | None"
    orange_detected_in_zone: bool
    goblin_detected_in_zone: bool
    goblin_support_frame: bool


@dataclass
class IdentityFrameResult:
    runtime_identity: str
    runtime_conf: float
    chosen_label: str
    chosen_conf: float
    evidence: "IdentityEvidence | None"
    goblin_support_frame: bool
    detected_flags: dict[str, bool]
    presence_flags: dict[str, bool]


def parse_zone(zone_text: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in zone_text.split(",")]
    if len(parts) != 4:
        raise ValueError("Zone must contain four comma-separated values: x,y,w,h")

    values = tuple(float(part) for part in parts)
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Zone values must be between 0 and 1")
    return values


def parse_zone_polygon(zone_polygon_text: str) -> list[tuple[float, float]]:
    raw_points = [chunk.strip() for chunk in zone_polygon_text.split(";") if chunk.strip()]
    if len(raw_points) < 3:
        raise ValueError("Zone polygon must contain at least 3 points.")

    points: list[tuple[float, float]] = []
    for raw_point in raw_points:
        parts = [part.strip() for part in raw_point.split(",")]
        if len(parts) != 2:
            raise ValueError("Each polygon point must be 'x,y'.")
        x, y = float(parts[0]), float(parts[1])
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Polygon point values must be between 0 and 1.")
        points.append((x, y))
    return points


def zone_to_pixels(
    frame_width: int, frame_height: int, zone: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    x, y, w, h = zone
    x1 = max(0, min(int(x * frame_width), frame_width - 1))
    y1 = max(0, min(int(y * frame_height), frame_height - 1))
    x2 = max(x1 + 1, min(int((x + w) * frame_width), frame_width))
    y2 = max(y1 + 1, min(int((y + h) * frame_height), frame_height))
    return x1, y1, x2, y2


def pixels_to_zone(
    frame_width: int, frame_height: int, x1: int, y1: int, x2: int, y2: int
) -> tuple[float, float, float, float]:
    fw = max(frame_width, 1)
    fh = max(frame_height, 1)
    return (
        max(0.0, min(x1 / fw, 1.0)),
        max(0.0, min(y1 / fh, 1.0)),
        max(0.0, min((x2 - x1) / fw, 1.0)),
        max(0.0, min((y2 - y1) / fh, 1.0)),
    )


def zone_polygon_to_pixels(
    frame_width: int, frame_height: int, points: list[tuple[float, float]]
) -> np.ndarray:
    px_points: list[list[int]] = []
    for x, y in points:
        px = max(0, min(int(round(x * frame_width)), frame_width - 1))
        py = max(0, min(int(round(y * frame_height)), frame_height - 1))
        px_points.append([px, py])
    return np.array(px_points, dtype=np.int32)


def polygon_bbox(points: np.ndarray, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(points)
    x1 = max(0, min(x, frame_width - 1))
    y1 = max(0, min(y, frame_height - 1))
    x2 = max(x1 + 1, min(x + w, frame_width))
    y2 = max(y1 + 1, min(y + h, frame_height))
    return x1, y1, x2, y2


def format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def event_log_file_for_timestamp(event_log_dir: Path, ts: float) -> Path:
    day_name = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    return event_log_dir / f"{day_name}.log"


def append_event_log(event_log_dir: Path, message: str, ts: float | None = None) -> None:
    event_ts = ts if ts is not None else time.time()
    event_log_path = event_log_file_for_timestamp(event_log_dir, event_ts)
    with event_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


def box_intersects_zone(
    box: tuple[int, int, int, int], zone_box: tuple[int, int, int, int]
) -> bool:
    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = zone_box
    return not (x2 <= zx1 or x1 >= zx2 or y2 <= zy1 or y1 >= zy2)


def box_zone_overlap_ratio(
    box: tuple[int, int, int, int], zone_box: tuple[int, int, int, int]
) -> float:
    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = zone_box
    ix1 = max(x1, zx1)
    iy1 = max(y1, zy1)
    ix2 = min(x2, zx2)
    iy2 = min(y2, zy2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = float(iw * ih)
    box_area = float(max(1, (x2 - x1) * (y2 - y1)))
    return inter_area / box_area


def box_polygon_overlap_ratio(
    box: tuple[int, int, int, int], polygon_points: np.ndarray
) -> float:
    x1, y1, x2, y2 = box
    box_poly = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    zone_poly = cv2.convexHull(polygon_points.astype(np.float32))
    try:
        area, _intersection = cv2.intersectConvexConvex(zone_poly, box_poly)
    except cv2.error:
        return 0.0
    box_area = float(max(1, (x2 - x1) * (y2 - y1)))
    return float(max(0.0, area)) / box_area


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_detector(model_path: str | None) -> tuple[object | None, str]:
    if not model_path:
        return None, "none"

    suffix = Path(model_path).suffix.lower()
    if suffix == ".pt":
        return YOLO(model_path), "ultralytics"
    if suffix == ".onnx":
        return cv2.dnn.readNetFromONNX(model_path), "opencv_dnn"

    raise ValueError(f"Unsupported model format: {suffix}")


def configure_detector_device(detector: object | None, backend: str, device: str) -> str:
    if detector is None:
        return "cpu"

    if backend == "ultralytics":
        if device == "cuda":
            if torch.cuda.is_available():
                return "cuda:0"
            print(
                "CUDA was requested but PyTorch CUDA is unavailable. Falling back to CPU.",
                file=sys.stderr,
            )
        return "cpu"

    if backend != "opencv_dnn":
        return "cpu"

    if device == "cuda":
        try:
            detector = detector  # type: ignore[assignment]
            detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            return "cuda"
        except cv2.error:
            print(
                "CUDA backend is not available in this OpenCV build. Falling back to CPU.",
                file=sys.stderr,
            )

    detector = detector  # type: ignore[assignment]
    detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return "cpu"


def detect_cats(
    detector: object | None,
    backend: str,
    device: str,
    frame: np.ndarray,
    confidence_threshold: float,
    cat_class_id: int,
    cat_imgsz: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    if detector is None:
        return []

    if backend == "ultralytics":
        results = detector.predict(
            source=frame,
            conf=confidence_threshold,
            classes=[cat_class_id],
            device=device,
            verbose=False,
            imgsz=cat_imgsz,
        )
        detections: list[tuple[tuple[int, int, int, int], float]] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        for box, score in zip(xyxy, confs):
            x1, y1, x2, y2 = [int(v) for v in box]
            detections.append(((x1, y1, x2, y2), float(score)))
        return detections

    detector = detector  # type: ignore[assignment]

    input_size = 640
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1 / 255.0,
        size=(input_size, input_size),
        swapRB=True,
        crop=False,
    )
    detector.setInput(blob)
    outputs = detector.forward()
    predictions = np.squeeze(outputs)
    if predictions.ndim != 2:
        return []
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
    if predictions.shape[1] < 5 + cat_class_id:
        return []

    frame_height, frame_width = frame.shape[:2]
    scale_x = frame_width / input_size
    scale_y = frame_height / input_size

    boxes: list[list[int]] = []
    scores: list[float] = []

    # YOLOv8 ONNX commonly outputs cx, cy, w, h, then per-class scores.
    for row in predictions:
        if row.shape[0] <= 4 + cat_class_id:
            continue
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if class_id != cat_class_id or score < confidence_threshold:
            continue

        cx, cy, width, height = row[:4]
        x1 = int((cx - width / 2) * scale_x)
        y1 = int((cy - height / 2) * scale_y)
        w = int(width * scale_x)
        h = int(height * scale_y)
        boxes.append([x1, y1, w, h])
        scores.append(score)

    if not boxes:
        return []

    indexes = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, 0.45)
    if len(indexes) == 0:
        return []

    detections: list[tuple[tuple[int, int, int, int], float]] = []
    for idx in np.array(indexes).flatten():
        x, y, w, h = boxes[int(idx)]
        detections.append(((x, y, x + w, y + h), scores[int(idx)]))
    return detections


def preprocess_cat_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return frame

    if mode == "night-lite":
        # Lightweight path for IR: boost local contrast without heavy denoising.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # IR night frames benefit from denoising + local contrast boost.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


@dataclass
class IdentityEvidence:
    orange_ratio: float
    white_ratio: float
    black_ratio: float
    black_blob_count: int
    torso_core_white_ratio: float
    torso_core_black_ratio: float
    torso_core_black_blob_count: int
    torso_core_goblin_evidence: float
    periphery_goblin_evidence: float
    blue_spill_ratio: float
    low_light: bool
    orange_evidence: float
    goblin_evidence: float
    contamination_downgraded: bool


@dataclass
class RegionCoatStats:
    white_ratio: float
    black_ratio: float
    orange_ratio: float
    blue_spill_ratio: float
    black_blob_count: int
    valid_pixels: int


@dataclass
class ZoneCandidate:
    score: float
    label: str
    label_conf: float
    evidence: IdentityEvidence
    goblin_support_frame: bool


def count_meaningful_blobs(
    binary_mask: np.ndarray, valid_pixels: int, min_area_ratio: float
) -> int:
    if valid_pixels <= 0 or not np.any(binary_mask):
        return 0
    binary_u8 = binary_mask.astype(np.uint8) * 255
    comp_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary_u8, connectivity=8
    )
    min_blob_area = max(4, int(valid_pixels * min_area_ratio))
    blob_count = 0
    for comp_idx in range(1, comp_count):
        area = int(stats[comp_idx, cv2.CC_STAT_AREA])
        if area >= min_blob_area:
            blob_count += 1
    return blob_count


def is_goblin_support_frame(
    label: str,
    label_conf: float,
    evidence: IdentityEvidence,
    *,
    min_conf: float,
    min_margin: float,
    min_torso_black_blobs: int,
    min_torso_white_ratio: float,
) -> bool:
    if label != "white_black_dotted":
        return False
    if evidence.contamination_downgraded:
        return False
    return (
        label_conf >= min_conf
        and evidence.goblin_evidence >= evidence.orange_evidence + min_margin
        and evidence.torso_core_black_blob_count >= min_torso_black_blobs
        and evidence.torso_core_white_ratio >= min_torso_white_ratio
    )


def classify_cat_identity(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    contamination_periphery_margin_max: float,
) -> tuple[str, float, IdentityEvidence]:
    x1, y1, x2, y2 = box
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 12 or box_h < 12:
        empty = IdentityEvidence(
            0.0,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            False,
            0.0,
            0.0,
            False,
        )
        return "unknown", 0.0, empty

    # Shrink toward center/body to reduce door/wall background leakage.
    inset_x = max(1, int(box_w * 0.16))
    inset_y = max(1, int(box_h * 0.14))
    rx1 = min(x2 - 1, x1 + inset_x)
    ry1 = min(y2 - 1, y1 + inset_y)
    rx2 = max(rx1 + 1, x2 - inset_x)
    ry2 = max(ry1 + 1, y2 - inset_y)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        empty = IdentityEvidence(
            0.0,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            False,
            0.0,
            0.0,
            False,
        )
        return "unknown", 0.0, empty

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    roi_h, roi_w = v.shape
    yy, xx = np.ogrid[:roi_h, :roi_w]
    cx = (roi_w - 1) / 2.0
    cy = (roi_h - 1) / 2.0
    # Body-only focus mask (ellipse) to prefer torso over bbox edges.
    a = max(1.0, roi_w * 0.46)
    b = max(1.0, roi_h * 0.46)
    body_mask = (((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2) <= 1.0

    dist_left = xx.astype(np.float32)
    dist_right = (roi_w - 1 - xx).astype(np.float32)
    dist_top = yy.astype(np.float32)
    dist_bottom = (roi_h - 1 - yy).astype(np.float32)
    edge_dist = np.minimum(
        np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom)
    )
    boundary_band = max(2, int(min(roi_w, roi_h) * 0.12))
    boundary_mask = edge_dist <= float(boundary_band)

    # Ignore the darkest boundary band (likely background/shadow contamination).
    base_valid = (v > 20) & body_mask
    boundary_valid = boundary_mask & base_valid
    if np.any(boundary_valid):
        boundary_values = v[boundary_valid]
        dark_cutoff = float(np.percentile(boundary_values, 30))
        dark_boundary_mask = boundary_mask & (v <= dark_cutoff)
    else:
        dark_boundary_mask = np.zeros_like(base_valid, dtype=bool)

    valid_mask = base_valid & (~dark_boundary_mask)
    valid_pixels = int(np.count_nonzero(valid_mask))
    if valid_pixels < 80:
        # Fallback when ROI is tiny or overly filtered.
        valid_mask = base_valid
        valid_pixels = max(int(np.count_nonzero(valid_mask)), 1)
    torso_core_rect = np.zeros_like(valid_mask, dtype=bool)
    core_x1 = int(round(roi_w * 0.225))
    core_x2 = int(round(roi_w * 0.775))
    core_y1 = int(round(roi_h * 0.20))
    core_y2 = int(round(roi_h * 0.80))
    torso_core_rect[core_y1:core_y2, core_x1:core_x2] = True
    torso_core_mask = valid_mask & torso_core_rect
    if np.count_nonzero(torso_core_mask) < 40:
        torso_core_mask = valid_mask & body_mask & torso_core_rect
    periphery_mask = valid_mask & (~torso_core_mask)

    def compute_region_stats(region_mask: np.ndarray) -> RegionCoatStats:
        region_pixels = int(np.count_nonzero(region_mask))
        if region_pixels <= 0:
            return RegionCoatStats(0.0, 0.0, 0.0, 0.0, 0, 0)
        blue_spill_mask = (
            region_mask
            & (h >= 92)
            & (h <= 138)
            & (s >= 80)
            & (v >= 70)
        )
        white_mask = region_mask & (~blue_spill_mask) & (s < 65) & (v > 95)
        black_mask = region_mask & (v < 62)
        orange_mask = region_mask & (h >= 4) & (h <= 26) & (s > 85) & (v > 50)
        min_area_ratio = 0.010 if region_pixels < 140 else 0.006
        black_blob_count = count_meaningful_blobs(
            black_mask, region_pixels, min_area_ratio
        )
        return RegionCoatStats(
            white_ratio=float(np.count_nonzero(white_mask)) / region_pixels,
            black_ratio=float(np.count_nonzero(black_mask)) / region_pixels,
            orange_ratio=float(np.count_nonzero(orange_mask)) / region_pixels,
            blue_spill_ratio=float(np.count_nonzero(blue_spill_mask)) / region_pixels,
            black_blob_count=black_blob_count,
            valid_pixels=region_pixels,
        )

    overall_stats = compute_region_stats(valid_mask)
    torso_core_stats = compute_region_stats(torso_core_mask)
    periphery_stats = compute_region_stats(periphery_mask)

    mean_v = float(np.mean(v[valid_mask])) if np.any(valid_mask) else 0.0
    low_light = mean_v < 92.0
    white_ratio = overall_stats.white_ratio
    black_ratio = overall_stats.black_ratio
    orange_ratio = overall_stats.orange_ratio
    black_blob_count = overall_stats.black_blob_count

    orange_evidence = max(
        0.0,
        orange_ratio * 2.2
        - black_ratio * 1.4
        - max(white_ratio - orange_ratio * 1.2, 0.0) * 0.35
        + (0.08 if black_blob_count == 0 else 0.0)
        - (0.12 if black_blob_count >= 2 else 0.0),
    )
    torso_core_goblin_evidence = max(
        0.0,
        torso_core_stats.white_ratio * 1.45
        + torso_core_stats.black_ratio * 2.6
        + min(torso_core_stats.black_blob_count * 0.10, 0.20)
        - torso_core_stats.orange_ratio * 1.0
        - torso_core_stats.blue_spill_ratio * 0.9,
    )
    periphery_goblin_evidence = max(
        0.0,
        periphery_stats.white_ratio * 1.15
        + periphery_stats.black_ratio * 2.1
        + min(periphery_stats.black_blob_count * 0.06, 0.12)
        - periphery_stats.orange_ratio * 0.8
        - periphery_stats.blue_spill_ratio * 1.0,
    )
    goblin_evidence = max(
        0.0,
        white_ratio * 1.15
        + black_ratio * 2.15
        + min(black_blob_count * 0.07, 0.21)
        + min(torso_core_stats.white_ratio * 0.70, 0.18)
        + min(torso_core_stats.black_blob_count * 0.10, 0.18)
        - orange_ratio * 1.05
        + (0.05 if (low_light and torso_core_stats.black_blob_count >= 1) else 0.0)
        - (0.12 if orange_ratio > 0.22 else 0.0),
    )
    contamination_downgraded = (
        periphery_stats.valid_pixels > 0
        and periphery_goblin_evidence
        > torso_core_goblin_evidence + contamination_periphery_margin_max
    )
    evidence = IdentityEvidence(
        orange_ratio=orange_ratio,
        white_ratio=white_ratio,
        black_ratio=black_ratio,
        black_blob_count=black_blob_count,
        torso_core_white_ratio=torso_core_stats.white_ratio,
        torso_core_black_ratio=torso_core_stats.black_ratio,
        torso_core_black_blob_count=torso_core_stats.black_blob_count,
        torso_core_goblin_evidence=min(1.0, torso_core_goblin_evidence),
        periphery_goblin_evidence=min(1.0, periphery_goblin_evidence),
        blue_spill_ratio=overall_stats.blue_spill_ratio,
        low_light=low_light,
        orange_evidence=min(1.0, orange_evidence),
        goblin_evidence=min(1.0, goblin_evidence),
        contamination_downgraded=contamination_downgraded,
    )

    # In low-light warm scenes (e.g. yellow TV cast), Goblin may look orange-ish.
    # Require stronger patch evidence so Orange does not flip from weak dark noise.
    goblin_low_light_patchy = (
        low_light
        and black_ratio >= 0.045
        and torso_core_stats.black_blob_count >= 1
        and torso_core_stats.white_ratio >= 0.10
        and orange_ratio <= 0.23
        and black_ratio >= orange_ratio * 0.38
    )
    if goblin_low_light_patchy:
        if goblin_evidence < orange_evidence + 0.08 or contamination_downgraded:
            return "unknown", min(0.55, 0.18 + goblin_evidence * 0.65), evidence
        score = min(
            0.99,
            0.38
            + min(white_ratio * 1.0, 0.22)
            + min(black_ratio * 3.4, 0.35)
            + min(torso_core_stats.black_blob_count * 0.10, 0.18),
        )
        return "white_black_dotted", score, evidence

    # Orange guard: require weak black evidence before forcing Orange.
    if (
        orange_ratio >= 0.12
        and orange_ratio >= white_ratio * 1.1
        and black_ratio < 0.05
        and torso_core_stats.black_blob_count <= 1
    ):
        score = min(0.99, 0.42 + orange_ratio * 2.5 - min(black_ratio * 0.7, 0.12))
        return "orange", score, evidence

    # Goblin needs visible white+black coat structure.
    goblin_patchy = (
        white_ratio >= 0.15
        and black_ratio >= 0.04
        and black_blob_count >= 2
        and orange_ratio <= 0.26
        and torso_core_stats.white_ratio >= 0.10
        and torso_core_stats.black_blob_count >= 1
    )
    goblin_high_contrast = (
        white_ratio >= 0.23
        and black_ratio >= 0.05
        and black_blob_count >= 1
        and orange_ratio <= 0.30
        and torso_core_stats.white_ratio >= 0.10
    )
    if goblin_patchy or goblin_high_contrast:
        if goblin_evidence < orange_evidence + 0.08 or contamination_downgraded:
            return "unknown", min(0.55, 0.18 + goblin_evidence * 0.65), evidence
        score = min(
            0.99,
            0.4
            + min(white_ratio * 1.2, 0.28)
            + min(black_ratio * 3.0, 0.32)
            + min(torso_core_stats.black_blob_count * 0.09, 0.18),
        )
        return "white_black_dotted", score, evidence

    # Only call Orange when black evidence is weak.
    if (
        orange_ratio >= 0.14
        and orange_ratio >= white_ratio * 0.78
        and (black_ratio < 0.05 or white_ratio < 0.12)
        and torso_core_stats.black_blob_count <= 1
    ):
        score = min(0.99, 0.4 + orange_ratio * 2.6)
        return "orange", score, evidence

    # Soft Orange fallback: useful when warm casts dilute saturation and we would
    # otherwise remain unknown despite clearly weak Goblin evidence.
    if (
        orange_evidence >= max(goblin_evidence * 1.25, 0.18)
        and torso_core_stats.black_blob_count <= 1
        and black_ratio < 0.07
    ):
        score = min(0.92, 0.34 + orange_evidence * 0.9)
        return "orange", score, evidence

    return "unknown", max(0.0, min(0.6, orange_ratio + white_ratio * 0.4)), evidence


def create_identity_presence_map() -> dict[str, IdentityPresenceState]:
    return {
        "orange": IdentityPresenceState(),
        "possible_goblin": IdentityPresenceState(),
        "confirmed_goblin": IdentityPresenceState(),
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.video and args.url:
        raise ValueError("Use either --video or --url, not both.")
    if args.process_fps <= 0:
        raise ValueError("--process-fps must be greater than 0.")
    if args.snapshot_cooldown < 0:
        raise ValueError("--snapshot-cooldown must be 0 or greater.")
    if args.alert_seconds < 0:
        raise ValueError("--alert-seconds must be 0 or greater.")
    if args.clip_seconds <= 0:
        raise ValueError("--clip-seconds must be greater than 0.")
    if not 0 <= args.cat_confidence <= 1:
        raise ValueError("--cat-confidence must be between 0 and 1.")
    if args.cat_enter_frames <= 0:
        raise ValueError("--cat-enter-frames must be greater than 0.")
    if not 0 <= args.cat_zone_overlap <= 1:
        raise ValueError("--cat-zone-overlap must be between 0 and 1.")
    if args.cat_imgsz <= 0:
        raise ValueError("--cat-imgsz must be greater than 0.")
    if args.cat_hold_seconds < 0:
        raise ValueError("--cat-hold-seconds must be 0 or greater.")
    if args.possible_goblin_seconds <= 0:
        raise ValueError("--possible-goblin-seconds must be greater than 0.")
    if args.headless and args.zone_edit:
        raise ValueError("--zone-edit cannot be used together with --headless.")
    if args.id_goblin_support_conf < 0:
        raise ValueError("--id-goblin-support-conf must be 0 or greater.")
    if args.id_goblin_support_margin < 0:
        raise ValueError("--id-goblin-support-margin must be 0 or greater.")
    if args.id_goblin_torso_white_min < 0:
        raise ValueError("--id-goblin-torso-white-min must be 0 or greater.")
    if args.id_goblin_torso_black_blobs_min < 0:
        raise ValueError("--id-goblin-torso-black-blobs-min must be 0 or greater.")
    if args.id_goblin_periphery_margin_max < 0:
        raise ValueError("--id-goblin-periphery-margin-max must be 0 or greater.")
    if args.id_lock_margin < 0:
        raise ValueError("--id-lock-margin must be 0 or greater.")
    if args.id_switch_margin < 0:
        raise ValueError("--id-switch-margin must be 0 or greater.")
    if args.id_orange_clear_streak < 1:
        raise ValueError("--id-orange-clear-streak must be at least 1.")
    if args.id_goblin_support_window < 1:
        raise ValueError("--id-goblin-support-window must be at least 1.")
    if args.id_confirmed_goblin_support_count < 1:
        raise ValueError("--id-confirmed-goblin-support-count must be at least 1.")
    if args.id_confirmed_goblin_support_count > args.id_goblin_support_window:
        raise ValueError(
            "--id-confirmed-goblin-support-count cannot exceed --id-goblin-support-window."
        )
    if args.id_confirmed_goblin_hold_seconds < 0:
        raise ValueError("--id-confirmed-goblin-hold-seconds must be 0 or greater.")


def resolve_runtime_config(
    repo_root: Path, args: argparse.Namespace
) -> RuntimeConfig:
    validate_args(args)
    rtsp_url = args.url or os.environ.get("RTSP_URL")
    video_path: Path | None = None
    is_video_source = bool(args.video)
    if args.video:
        candidate = Path(args.video)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        video_path = candidate
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")
    elif not rtsp_url:
        raise ValueError(
            "Missing input source. Set RTSP_URL in .env, pass --url, or pass --video."
        )

    zone = parse_zone(args.zone)
    zone_polygon_norm: list[tuple[float, float]] | None = None
    if args.zone_polygon:
        zone_polygon_norm = parse_zone_polygon(args.zone_polygon)
        print(
            "Using polygon zone: "
            + ";".join(f"{x:.4f},{y:.4f}" for x, y in zone_polygon_norm)
        )
    else:
        print(
            "Using rectangle zone: "
            f"{zone[0]:.4f},{zone[1]:.4f},{zone[2]:.4f},{zone[3]:.4f}"
        )

    snapshot_dir = repo_root / args.snapshot_dir
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    event_log_dir = repo_root / "event-log"
    event_log_dir.mkdir(parents=True, exist_ok=True)

    identity_debug_csv_path: Path | None = None
    if args.identity_debug_csv:
        debug_candidate = Path(args.identity_debug_csv)
        if not debug_candidate.is_absolute():
            debug_candidate = repo_root / debug_candidate
        identity_debug_csv_path = debug_candidate.resolve()
        identity_debug_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cat_model_path = None
    if args.cat_model:
        cat_model_path = str((repo_root / args.cat_model).resolve())

    launch_origin = (args.launch_origin or "manual").strip().lower().replace(" ", "_")
    if not launch_origin:
        launch_origin = "manual"

    return RuntimeConfig(
        repo_root=repo_root,
        rtsp_url=rtsp_url,
        video_path=video_path,
        is_video_source=is_video_source,
        zone_rect_edit=list(zone),
        zone_polygon_edit=(
            [[x, y] for x, y in zone_polygon_norm]
            if zone_polygon_norm is not None
            else None
        ),
        snapshot_dir=snapshot_dir,
        event_log_dir=event_log_dir,
        identity_debug_csv_path=identity_debug_csv_path,
        cat_model_path=cat_model_path,
        launch_origin=launch_origin,
    )


def open_capture(config: RuntimeConfig) -> cv2.VideoCapture:
    cap = (
        cv2.VideoCapture(str(config.video_path))
        if config.is_video_source
        else cv2.VideoCapture(config.rtsp_url, cv2.CAP_FFMPEG)
    )
    if not cap.isOpened():
        if config.is_video_source:
            raise RuntimeError(f"OpenCV could not open video file: {config.video_path}")
        raise RuntimeError(
            "OpenCV could not open the RTSP stream. Check the camera IP, credentials, "
            "and whether OpenCV has FFmpeg support."
        )
    if not config.is_video_source:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def setup_detector_runtime(
    config: RuntimeConfig, args: argparse.Namespace
) -> DetectorRuntime:
    detector, detector_backend = load_detector(config.cat_model_path)
    detector_device = configure_detector_device(detector, detector_backend, args.device)
    if args.cat_detect_mode == "auto":
        cat_detect_mode = "always" if config.is_video_source else "motion"
    else:
        cat_detect_mode = args.cat_detect_mode
    mode = (
        "motion-only"
        if detector is None
        else f"cat-in-zone ({cat_detect_mode})"
    )
    return DetectorRuntime(
        detector=detector,
        backend=detector_backend,
        device=detector_device,
        cat_detect_mode=cat_detect_mode,
        mode=mode,
    )


def initialize_capture_state(
    cap: cv2.VideoCapture, config: RuntimeConfig, args: argparse.Namespace
) -> CaptureState:
    capture_state = CaptureState(
        frame_interval=1.0 / args.process_fps,
        next_frame_at=time.perf_counter(),
    )
    if config.is_video_source:
        raw_video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if np.isfinite(raw_video_fps) and raw_video_fps > 0:
            capture_state.video_native_fps = raw_video_fps
    return capture_state


def initialize_runtime_state(
    args: argparse.Namespace, snapshot_dir: Path
) -> RuntimeState:
    output_state = OutputRuntimeState(clip_dir=snapshot_dir / "clips")
    if args.save_clip_on_alert and output_state.clip_dir is not None:
        output_state.clip_dir.mkdir(parents=True, exist_ok=True)
    return RuntimeState(
        capture=CaptureState(),
        activity=ActivityState(),
        identity=IdentityRuntimeState(
            goblin_support_history=deque(maxlen=args.id_goblin_support_window),
            presence=create_identity_presence_map(),
        ),
        output=output_state,
        lifecycle=RunLifecycleState(started_at=time.time()),
    )


def setup_identity_debug_writer(
    output_state: OutputRuntimeState, identity_debug_csv_path: Path | None
) -> None:
    if identity_debug_csv_path is None:
        return
    output_state.identity_debug_file = identity_debug_csv_path.open(
        "w", encoding="utf-8", newline=""
    )
    output_state.identity_debug_writer = csv.DictWriter(
        output_state.identity_debug_file,
        fieldnames=[
            "source",
            "frame_index",
            "source_seconds",
            "wall_timestamp",
            "chosen_identity",
            "runtime_identity",
            "confidence",
            "orange_evidence",
            "goblin_evidence",
            "torso_core_white_ratio",
            "torso_core_black_blob_count",
            "goblin_support_frame",
            "contamination_downgraded",
        ],
    )
    output_state.identity_debug_writer.writeheader()


def source_text_for_config(config: RuntimeConfig) -> str:
    return str(config.video_path) if config.is_video_source else "RTSP stream"


def setup_preview_window(args: argparse.Namespace) -> None:
    if args.headless:
        return
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, args.width, args.height)


def print_zone_value(
    zone_rect_edit: list[float], zone_polygon_edit: list[list[float]] | None
) -> None:
    if zone_polygon_edit is not None:
        text = ";".join(f"{p[0]:.4f},{p[1]:.4f}" for p in zone_polygon_edit)
        print(f"ZONE_POLYGON={text}")
        return
    print(
        "ZONE="
        f"{zone_rect_edit[0]:.4f},{zone_rect_edit[1]:.4f},"
        f"{zone_rect_edit[2]:.4f},{zone_rect_edit[3]:.4f}"
    )


def reset_identity_presence(presence: dict[str, IdentityPresenceState]) -> None:
    for state in presence.values():
        state.seen_streak = 0
        state.last_seen_at = 0.0
        state.present = False
        state.started_at = 0.0
        state.alert_fired = False


def reset_identity_runtime(
    identity_state: IdentityRuntimeState,
    args: argparse.Namespace,
) -> None:
    identity_state.orange_acc = 0.0
    identity_state.goblin_acc = 0.0
    identity_state.support_frames = 0
    identity_state.stable_label = "unknown"
    identity_state.stable_conf = 0.0
    identity_state.goblin_support_history = deque(maxlen=args.id_goblin_support_window)
    identity_state.confirmed_goblin_hold_started_at = 0.0
    identity_state.orange_stronger_streak = 0
    identity_state.possible_goblin_alerted_this_session = False
    reset_identity_presence(identity_state.presence)


def describe_identity_subject(identity_label: str) -> str:
    if identity_label == "orange":
        return "Orange cat in zone"
    if identity_label == "possible_goblin":
        return "Possible Goblin in zone"
    if identity_label == "confirmed_goblin":
        return "White-black cat in zone"
    return "Cat in zone"


def build_mode_text(detector_runtime: DetectorRuntime) -> str:
    return (
        "motion-only"
        if detector_runtime.detector is None
        else f"cat-in-zone ({detector_runtime.cat_detect_mode})"
    )


def print_runtime_banner(
    args: argparse.Namespace,
    config: RuntimeConfig,
    detector_runtime: DetectorRuntime,
) -> None:
    source_text = source_text_for_config(config)
    mode = build_mode_text(detector_runtime)
    print(
        f"Connected to {source_text}. Processing at {args.process_fps:.2f} FPS in "
        f"{mode} mode using {detector_runtime.backend} on "
        f"{detector_runtime.device.upper()}. "
        + ("Headless mode active." if args.headless else "Press q to quit.")
    )


def log_run_start(
    event_log_dir: Path,
    lifecycle: RunLifecycleState,
    config: RuntimeConfig,
    detector_runtime: DetectorRuntime,
    args: argparse.Namespace,
) -> None:
    run_start_message = (
        f"[{format_timestamp(lifecycle.started_at)}] APP_START: "
        f"source={source_text_for_config(config)} "
        f"mode={build_mode_text(detector_runtime)} fps={args.process_fps:.2f} "
        f"launch_origin={config.launch_origin}"
    )
    append_event_log(event_log_dir, run_start_message, lifecycle.started_at)
    print(run_start_message)


def finalize_run_log(
    event_log_dir: Path,
    lifecycle: RunLifecycleState,
) -> None:
    if lifecycle.ended:
        return
    run_ended_at = time.time()
    run_end_message = (
        f"[{format_timestamp(run_ended_at)}] APP_END: reason={lifecycle.reason} "
        f"duration={run_ended_at - lifecycle.started_at:.1f}s "
        f"frames={lifecycle.frame_count}"
    )
    append_event_log(event_log_dir, run_end_message, run_ended_at)
    print(run_end_message)
    lifecycle.ended = True


def start_clip_writer(
    output_state: OutputRuntimeState,
    args: argparse.Namespace,
    frame_width: int,
    frame_height: int,
    event_ts: float,
    alert_cat: str,
    event_prefix: str,
) -> None:
    if (
        not args.save_clip_on_alert
        or output_state.clip_writer is not None
        or output_state.clip_dir is None
    ):
        return
    clip_name = (
        f"{event_prefix}_"
        + datetime.fromtimestamp(event_ts).strftime("%Y-%m-%d_%H%M%S")
        + f"_{alert_cat}.mp4"
    )
    clip_path = output_state.clip_dir / clip_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    clip_writer = cv2.VideoWriter(
        str(clip_path),
        fourcc,
        max(args.process_fps, 1.0),
        (frame_width, frame_height),
    )
    if clip_writer.isOpened():
        output_state.clip_writer = clip_writer
        output_state.active_clip_path = clip_path
        output_state.clip_end_at = event_ts + args.clip_seconds
        print(f"[{format_timestamp(event_ts)}] Clip recording started: {clip_path}")
        return
    clip_writer.release()
    print(
        f"[{format_timestamp(event_ts)}] Failed to start clip writer: {clip_path}",
        file=sys.stderr,
    )


def close_clip_writer(output_state: OutputRuntimeState, event_ts: float) -> None:
    if output_state.clip_writer is None:
        return
    output_state.clip_writer.release()
    if output_state.active_clip_path is not None:
        print(f"[{format_timestamp(event_ts)}] Clip saved: {output_state.active_clip_path}")
    output_state.clip_writer = None
    output_state.active_clip_path = None


def emit_alert(
    event_log_dir: Path,
    output_state: OutputRuntimeState,
    args: argparse.Namespace,
    frame_width: int,
    frame_height: int,
    subject: str,
    alert_cat: str,
    duration_s: float,
    event_ts: float,
    current_motion_snapshot_name: str,
    event_prefix: str = "ALERT",
) -> None:
    if event_prefix == "ALERT":
        alert_message = (
            f"[{format_timestamp(event_ts)}] ALERT: {subject.lower()} lasted "
            f"{duration_s:.1f}s"
        )
    else:
        alert_message = (
            f"[{format_timestamp(event_ts)}] {event_prefix}: "
            f"{subject.lower()} lasted {duration_s:.1f}s"
        )
    if current_motion_snapshot_name:
        alert_message += f" snapshot={current_motion_snapshot_name}"
    print(alert_message)
    append_event_log(event_log_dir, alert_message, event_ts)
    start_clip_writer(
        output_state,
        args,
        frame_width,
        frame_height,
        event_ts,
        alert_cat,
        event_prefix,
    )


def build_frame_observation(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
    detector_runtime: DetectorRuntime,
    args: argparse.Namespace,
    zone_mask: np.ndarray,
    zone_polygon_px: np.ndarray | None,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[np.ndarray, FrameObservation]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    observation = FrameObservation(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        zone_mask=zone_mask,
        zone_polygon_px=zone_polygon_px,
        motion_percent=0.0,
        motion_active=False,
        cat_detected_in_zone=False,
        cat_detections=[],
        cat_identities=[],
        cat_identity_sample_label="unknown",
        cat_identity_sample_conf=0.0,
        cat_identity_sample_orange=0.0,
        cat_identity_sample_goblin=0.0,
        cat_identity_sample_evidence=None,
        orange_detected_in_zone=False,
        goblin_detected_in_zone=False,
        goblin_support_frame=False,
    )
    if previous_gray is None:
        return gray, observation

    delta = cv2.absdiff(previous_gray, gray)
    zone_delta = delta[y1:y2, x1:x2]
    _, thresh = cv2.threshold(zone_delta, 25, 255, cv2.THRESH_BINARY)
    zone_mask_roi = zone_mask[y1:y2, x1:x2]
    masked_motion = cv2.bitwise_and(thresh, thresh, mask=zone_mask_roi)
    changed_pixels = cv2.countNonZero(masked_motion)
    total_pixels = max(cv2.countNonZero(zone_mask_roi), 1)
    observation.motion_percent = changed_pixels * 100.0 / total_pixels
    observation.motion_active = observation.motion_percent >= args.motion_threshold

    should_run_cat = (
        detector_runtime.detector is not None
        and (
            detector_runtime.cat_detect_mode == "always"
            or observation.motion_active
        )
    )
    if not should_run_cat:
        return gray, observation

    cat_frame = preprocess_cat_frame(frame, args.cat_preprocess)
    observation.cat_detections = detect_cats(
        detector_runtime.detector,
        detector_runtime.backend,
        detector_runtime.device,
        cat_frame,
        args.cat_confidence,
        args.cat_class_id,
        args.cat_imgsz,
    )
    zone_candidates: list[ZoneCandidate] = []
    for box, score in observation.cat_detections:
        overlap_ratio = (
            box_polygon_overlap_ratio(box, zone_polygon_px)
            if zone_polygon_px is not None
            else box_zone_overlap_ratio(box, (x1, y1, x2, y2))
        )
        label, label_conf, evidence = classify_cat_identity(
            frame,
            box,
            contamination_periphery_margin_max=args.id_goblin_periphery_margin_max,
        )
        observation.cat_identities.append((label, label_conf))
        if overlap_ratio < args.cat_zone_overlap:
            continue
        goblin_support_frame = is_goblin_support_frame(
            label,
            label_conf,
            evidence,
            min_conf=args.id_goblin_support_conf,
            min_margin=args.id_goblin_support_margin,
            min_torso_black_blobs=args.id_goblin_torso_black_blobs_min,
            min_torso_white_ratio=args.id_goblin_torso_white_min,
        )
        candidate = ZoneCandidate(
            score=score,
            label=label,
            label_conf=label_conf,
            evidence=evidence,
            goblin_support_frame=goblin_support_frame,
        )
        zone_candidates.append(candidate)
        if label == "orange" and label_conf >= 0.35:
            observation.orange_detected_in_zone = True
        if goblin_support_frame:
            observation.goblin_detected_in_zone = True

    observation.cat_detected_in_zone = len(zone_candidates) > 0
    if zone_candidates:
        best_candidate = max(
            zone_candidates,
            key=lambda item: item.score * max(item.label_conf, 0.2),
        )
        observation.cat_identity_sample_label = best_candidate.label
        observation.cat_identity_sample_conf = best_candidate.label_conf
        observation.cat_identity_sample_orange = best_candidate.evidence.orange_evidence
        observation.cat_identity_sample_goblin = best_candidate.evidence.goblin_evidence
        observation.cat_identity_sample_evidence = best_candidate.evidence
        observation.goblin_support_frame = best_candidate.goblin_support_frame
    return gray, observation


def update_identity_runtime(
    args: argparse.Namespace,
    identity_state: IdentityRuntimeState,
    output_state: OutputRuntimeState,
    observation: FrameObservation,
    event_now: float,
    cap: cv2.VideoCapture,
    config: RuntimeConfig,
    frame_count: int,
    run_started_at: float,
) -> IdentityFrameResult:
    cat_identity_in_zone = "unknown"
    cat_identity_conf = 0.0
    if observation.cat_detected_in_zone:
        identity_state.support_frames += 1
        identity_state.orange_acc = (
            identity_state.orange_acc * 0.90 + observation.cat_identity_sample_orange
        )
        identity_state.goblin_acc = (
            identity_state.goblin_acc * 0.90 + observation.cat_identity_sample_goblin
        )
        identity_state.goblin_support_history.append(observation.goblin_support_frame)
    else:
        identity_state.orange_acc *= 0.97
        identity_state.goblin_acc *= 0.97

    min_support_frames = 2
    delta = identity_state.orange_acc - identity_state.goblin_acc
    dominant_acc = max(identity_state.orange_acc, identity_state.goblin_acc)
    if identity_state.support_frames >= min_support_frames:
        if identity_state.stable_label == "orange":
            if delta <= -args.id_switch_margin:
                identity_state.stable_label = "white_black_dotted"
        elif identity_state.stable_label == "white_black_dotted":
            if delta >= args.id_switch_margin:
                identity_state.stable_label = "orange"
        else:
            if delta >= args.id_lock_margin:
                identity_state.stable_label = "orange"
            elif delta <= -args.id_lock_margin:
                identity_state.stable_label = "white_black_dotted"
            elif (
                observation.cat_identity_sample_label == "orange"
                and observation.cat_identity_sample_conf >= 0.48
                and delta >= -0.08
                and identity_state.goblin_acc < 0.40
            ):
                identity_state.stable_label = "orange"
            elif (
                observation.cat_identity_sample_label == "white_black_dotted"
                and observation.cat_identity_sample_conf >= 0.58
                and delta <= 0.02
                and identity_state.orange_acc < 0.35
            ):
                identity_state.stable_label = "white_black_dotted"

    if identity_state.stable_label == "unknown":
        identity_state.stable_conf = max(
            0.0,
            min(
                0.7,
                max(observation.cat_identity_sample_conf * 0.8, dominant_acc * 0.55),
            ),
        )
    else:
        identity_state.stable_conf = min(
            0.99,
            0.45 + min(dominant_acc * 0.65, 0.35) + min(abs(delta), 0.25),
        )

    if observation.cat_detected_in_zone:
        if (
            observation.cat_identity_sample_label == "orange"
            and observation.cat_identity_sample_conf >= 0.48
            and observation.cat_identity_sample_orange
            >= observation.cat_identity_sample_goblin - 0.04
        ):
            identity_state.orange_stronger_streak += 1
        elif observation.goblin_support_frame:
            identity_state.orange_stronger_streak = 0
        else:
            identity_state.orange_stronger_streak = max(
                0, identity_state.orange_stronger_streak - 1
            )

    if identity_state.orange_stronger_streak >= args.id_orange_clear_streak:
        identity_state.goblin_support_history.clear()
        identity_state.confirmed_goblin_hold_started_at = 0.0

    recent_goblin_support = sum(1 for flag in identity_state.goblin_support_history if flag)
    if (
        recent_goblin_support >= args.id_confirmed_goblin_support_count
        and len(identity_state.goblin_support_history)
        >= args.id_confirmed_goblin_support_count
        and identity_state.stable_label == "white_black_dotted"
        and identity_state.orange_stronger_streak < args.id_orange_clear_streak
    ):
        if identity_state.confirmed_goblin_hold_started_at == 0.0:
            identity_state.confirmed_goblin_hold_started_at = event_now
    else:
        identity_state.confirmed_goblin_hold_started_at = 0.0

    confirmed_goblin_detected_now = (
        identity_state.confirmed_goblin_hold_started_at > 0.0
        and event_now - identity_state.confirmed_goblin_hold_started_at
        >= args.id_confirmed_goblin_hold_seconds
    )
    possible_goblin_detected_now = (
        recent_goblin_support >= 2
        and not confirmed_goblin_detected_now
        and identity_state.stable_label == "white_black_dotted"
        and identity_state.orange_stronger_streak < args.id_orange_clear_streak
    )

    detected_flags = {
        "orange": observation.orange_detected_in_zone,
        "possible_goblin": possible_goblin_detected_now,
        "confirmed_goblin": confirmed_goblin_detected_now,
    }
    for identity_label, detected_now in detected_flags.items():
        state = identity_state.presence[identity_label]
        previously_present = state.present
        if detected_now:
            state.seen_streak += 1
            state.last_seen_at = event_now
        else:
            state.seen_streak = 0

        min_start_frames = args.cat_enter_frames if identity_label == "orange" else 1
        should_start_identity = state.seen_streak >= min_start_frames
        should_hold_identity = (
            identity_label != "possible_goblin"
            and previously_present
            and event_now - state.last_seen_at <= args.cat_hold_seconds
        )
        present_now = should_start_identity or should_hold_identity
        state.present = present_now

        if present_now and not previously_present:
            state.started_at = event_now
            state.alert_fired = False
            print(
                f"[{format_timestamp(event_now)}] "
                f"{describe_identity_subject(identity_label)} entered"
            )
        elif not present_now and previously_present:
            print(
                f"[{format_timestamp(event_now)}] "
                f"{describe_identity_subject(identity_label)} left "
                f"(duration={event_now - state.started_at:.1f}s)"
            )
            state.alert_fired = False

    confirmed_present_now = identity_state.presence["confirmed_goblin"].present
    possible_present_now = identity_state.presence["possible_goblin"].present
    if confirmed_present_now or confirmed_goblin_detected_now:
        cat_identity_in_zone = "confirmed_goblin"
        cat_identity_conf = max(identity_state.stable_conf, args.id_goblin_support_conf)
    elif possible_present_now or possible_goblin_detected_now:
        cat_identity_in_zone = "possible_goblin"
        cat_identity_conf = max(identity_state.stable_conf, 0.58)
    else:
        cat_identity_in_zone = identity_state.stable_label
        cat_identity_conf = identity_state.stable_conf

    if output_state.identity_debug_writer is not None and observation.cat_detected_in_zone:
        source_seconds = (
            float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if config.is_video_source
            else max(0.0, event_now - run_started_at)
        )
        debug_evidence = observation.cat_identity_sample_evidence
        if debug_evidence is not None:
            output_state.identity_debug_writer.writerow(
                {
                    "source": source_text_for_config(config),
                    "frame_index": frame_count,
                    "source_seconds": f"{source_seconds:.3f}",
                    "wall_timestamp": format_timestamp(event_now),
                    "chosen_identity": observation.cat_identity_sample_label,
                    "runtime_identity": cat_identity_in_zone,
                    "confidence": f"{observation.cat_identity_sample_conf:.4f}",
                    "orange_evidence": f"{debug_evidence.orange_evidence:.4f}",
                    "goblin_evidence": f"{debug_evidence.goblin_evidence:.4f}",
                    "torso_core_white_ratio": (
                        f"{debug_evidence.torso_core_white_ratio:.4f}"
                    ),
                    "torso_core_black_blob_count": (
                        debug_evidence.torso_core_black_blob_count
                    ),
                    "goblin_support_frame": int(observation.goblin_support_frame),
                    "contamination_downgraded": int(
                        debug_evidence.contamination_downgraded
                    ),
                }
            )

    return IdentityFrameResult(
        runtime_identity=cat_identity_in_zone,
        runtime_conf=cat_identity_conf,
        chosen_label=observation.cat_identity_sample_label,
        chosen_conf=observation.cat_identity_sample_conf,
        evidence=observation.cat_identity_sample_evidence,
        goblin_support_frame=observation.goblin_support_frame,
        detected_flags=detected_flags,
        presence_flags={
            label: state.present for label, state in identity_state.presence.items()
        },
    )


def render_preview(
    frame: np.ndarray,
    args: argparse.Namespace,
    detector_runtime: DetectorRuntime,
    observation: FrameObservation,
    identity_state: IdentityRuntimeState,
    identity_result: IdentityFrameResult | None,
    tracked_active: bool,
    cat_in_zone: bool,
    current_dwell_subject: str,
    tracked_duration: float,
    fps: float,
    alert_fired: bool,
    zone_polygon_edit: list[list[float]] | None,
    cat_seen_streak: int,
) -> None:
    zone_color = (0, 0, 255) if tracked_active else (0, 255, 255)
    if observation.zone_polygon_px is not None:
        cv2.polylines(
            frame,
            [observation.zone_polygon_px],
            isClosed=True,
            color=zone_color,
            thickness=2,
        )
        if args.zone_edit:
            for pt in observation.zone_polygon_px:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 255, 255), -1)
    else:
        cv2.rectangle(
            frame,
            (observation.x1, observation.y1),
            (observation.x2, observation.y2),
            zone_color,
            2,
        )

    for idx, ((bx1, by1, bx2, by2), score) in enumerate(observation.cat_detections):
        overlap_ratio = (
            box_polygon_overlap_ratio((bx1, by1, bx2, by2), observation.zone_polygon_px)
            if observation.zone_polygon_px is not None
            else box_zone_overlap_ratio(
                (bx1, by1, bx2, by2),
                (observation.x1, observation.y1, observation.x2, observation.y2),
            )
        )
        in_zone = overlap_ratio >= args.cat_zone_overlap
        color = (0, 0, 255) if in_zone else (255, 0, 0)
        thickness = 3 if in_zone else 2
        identity_label = "unknown"
        identity_conf = 0.0
        if idx < len(observation.cat_identities):
            identity_label, identity_conf = observation.cat_identities[idx]
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, thickness)
        cv2.putText(
            frame,
            f"cat {score:.2f} {identity_label} {identity_conf:.2f}",
            (bx1, max(20, by1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Zone motion: {observation.motion_percent:.2f}%", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Cat in zone" if cat_in_zone else ("Motion detected" if observation.motion_active else "Idle"), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Dwell: {tracked_duration:.1f}s / {args.alert_seconds:.1f}s", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Mode: {build_mode_text(detector_runtime)} ({detector_runtime.backend}, {detector_runtime.device})", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if detector_runtime.detector is not None and identity_result is not None:
        cv2.putText(frame, f"Cat raw: {'yes' if observation.cat_detected_in_zone else 'no'} streak={cat_seen_streak}", (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Identity: {identity_result.runtime_identity} ({identity_result.runtime_conf:.2f})", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Goblin support: {sum(1 for flag in identity_state.goblin_support_history if flag)}/{len(identity_state.goblin_support_history)}", (10, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Zone: {'polygon' if zone_polygon_edit is not None else 'rect'} overlap>={args.cat_zone_overlap:.2f}", (10, 310 if detector_runtime.detector is not None else 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if args.zone_edit:
        cv2.putText(frame, "Zone edit: drag zone/points, press p to print, q to quit", (10, 345 if detector_runtime.detector is not None else 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if alert_fired:
        cv2.putText(frame, "ALERT", (10, 380 if detector_runtime.detector is not None else 310), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    if not args.headless:
        cv2.imshow(args.window_name, frame)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    args = parse_args()
    try:
        config = resolve_runtime_config(repo_root, args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        cap = open_capture(config)
        detector_runtime = setup_detector_runtime(config, args)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except cv2.error as exc:
        print(f"Failed to load cat model: {exc}", file=sys.stderr)
        return 1

    capture_state = initialize_capture_state(cap, config, args)
    runtime_state = initialize_runtime_state(args, config.snapshot_dir)
    runtime_state.capture = capture_state
    zone_rect_edit = config.zone_rect_edit
    zone_polygon_edit = config.zone_polygon_edit
    current_dwell_subject = "Zone activity"

    setup_preview_window(args)

    editor_state = EditorState()

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if not args.zone_edit:
            return
        frame_w = editor_state.frame_w
        frame_h = editor_state.frame_h
        if frame_w <= 0 or frame_h <= 0:
            return

        if zone_polygon_edit is not None:
            points_px = zone_polygon_to_pixels(
                frame_w, frame_h, [(p[0], p[1]) for p in zone_polygon_edit]
            )
            if event == cv2.EVENT_LBUTTONDOWN:
                best_idx = -1
                best_dist2 = 1e12
                for idx, pt in enumerate(points_px):
                    dx = int(pt[0]) - x
                    dy = int(pt[1]) - y
                    dist2 = float(dx * dx + dy * dy)
                    if dist2 < best_dist2:
                        best_dist2 = dist2
                        best_idx = idx
                if best_idx >= 0 and best_dist2 <= 24 * 24:
                    editor_state.dragging = True
                    editor_state.drag_kind = "poly_point"
                    editor_state.active_point = best_idx
                else:
                    inside = cv2.pointPolygonTest(
                        points_px.astype(np.float32), (float(x), float(y)), False
                    )
                    if inside >= 0:
                        editor_state.dragging = True
                        editor_state.drag_kind = "poly_move"
                        editor_state.start_mouse = (x, y)
                        editor_state.start_polygon = [
                            [p[0], p[1]] for p in zone_polygon_edit
                        ]
            elif event == cv2.EVENT_MOUSEMOVE and editor_state.dragging:
                drag_kind = editor_state.drag_kind
                if drag_kind == "poly_point":
                    idx = editor_state.active_point
                    if 0 <= idx < len(zone_polygon_edit):
                        zone_polygon_edit[idx][0] = clamp01(x / max(frame_w - 1, 1))
                        zone_polygon_edit[idx][1] = clamp01(y / max(frame_h - 1, 1))
                elif drag_kind == "poly_move":
                    smx, smy = editor_state.start_mouse
                    dx = (x - smx) / max(frame_w - 1, 1)
                    dy = (y - smy) / max(frame_h - 1, 1)
                    start_poly = editor_state.start_polygon
                    for i, pt in enumerate(start_poly):
                        zone_polygon_edit[i][0] = clamp01(float(pt[0]) + dx)
                        zone_polygon_edit[i][1] = clamp01(float(pt[1]) + dy)
            elif event == cv2.EVENT_LBUTTONUP and editor_state.dragging:
                editor_state.dragging = False
                editor_state.drag_kind = ""
                editor_state.active_point = -1
                editor_state.start_polygon = []
                print_zone_value(zone_rect_edit, zone_polygon_edit)
            return

        x1, y1, x2, y2 = zone_to_pixels(frame_w, frame_h, tuple(zone_rect_edit))
        if event == cv2.EVENT_LBUTTONDOWN:
            near_corner = abs(x - x2) <= 14 and abs(y - y2) <= 14
            in_rect = x1 <= x <= x2 and y1 <= y <= y2
            if near_corner:
                editor_state.dragging = True
                editor_state.drag_kind = "rect_resize"
                editor_state.start_rect = (x1, y1, x2, y2)
            elif in_rect:
                editor_state.dragging = True
                editor_state.drag_kind = "rect_move"
                editor_state.start_mouse = (x, y)
                editor_state.start_rect = (x1, y1, x2, y2)
        elif event == cv2.EVENT_MOUSEMOVE and editor_state.dragging:
            drag_kind = editor_state.drag_kind
            sx1, sy1, sx2, sy2 = editor_state.start_rect
            if drag_kind == "rect_move":
                smx, smy = editor_state.start_mouse
                dx = x - smx
                dy = y - smy
                w = sx2 - sx1
                h = sy2 - sy1
                nx1 = max(0, min(sx1 + dx, frame_w - max(1, w)))
                ny1 = max(0, min(sy1 + dy, frame_h - max(1, h)))
                nx2 = nx1 + w
                ny2 = ny1 + h
                zone_rect_edit[:] = pixels_to_zone(frame_w, frame_h, nx1, ny1, nx2, ny2)
            elif drag_kind == "rect_resize":
                nx2 = max(sx1 + 12, min(x, frame_w - 1))
                ny2 = max(sy1 + 12, min(y, frame_h - 1))
                zone_rect_edit[:] = pixels_to_zone(
                    frame_w, frame_h, sx1, sy1, nx2, ny2
                )
        elif event == cv2.EVENT_LBUTTONUP and editor_state.dragging:
            editor_state.dragging = False
            editor_state.drag_kind = ""
            print_zone_value(zone_rect_edit, zone_polygon_edit)

    if args.zone_edit and not args.headless:
        cv2.setMouseCallback(args.window_name, on_mouse)

    print_runtime_banner(args, config, detector_runtime)
    setup_identity_debug_writer(
        runtime_state.output, config.identity_debug_csv_path
    )
    atexit.register(lambda: finalize_run_log(config.event_log_dir, runtime_state.lifecycle))
    log_run_start(
        config.event_log_dir,
        runtime_state.lifecycle,
        config,
        detector_runtime,
        args,
    )

    # Keep the existing loop behavior intact while the new grouped runtime
    # state is phased in around it.
    is_video_source = config.is_video_source
    video_path = config.video_path
    detector = detector_runtime.detector
    detector_backend = detector_runtime.backend
    detector_device = detector_runtime.device
    cat_detect_mode = detector_runtime.cat_detect_mode
    mode = build_mode_text(detector_runtime)
    snapshot_dir = config.snapshot_dir
    source_text = source_text_for_config(config)
    frame_count = runtime_state.lifecycle.frame_count
    started_at = runtime_state.lifecycle.started_at
    run_started_at = started_at
    previous_gray = runtime_state.previous_gray
    frame_interval = runtime_state.capture.frame_interval
    next_frame_at = runtime_state.capture.next_frame_at
    motion_state = runtime_state.activity.motion_state
    motion_started_at = runtime_state.activity.motion_started_at
    last_snapshot_at = runtime_state.activity.last_snapshot_at
    current_motion_snapshot_name = runtime_state.activity.current_motion_snapshot_name
    alert_fired = runtime_state.activity.alert_fired
    cat_present_state = runtime_state.activity.cat_present_state
    cat_started_at = runtime_state.activity.cat_started_at
    cat_seen_streak = runtime_state.activity.cat_seen_streak
    cat_last_seen_at = runtime_state.activity.cat_last_seen_at
    identity_orange_acc = runtime_state.identity.orange_acc
    identity_goblin_acc = runtime_state.identity.goblin_acc
    identity_support_frames = runtime_state.identity.support_frames
    identity_stable_label = runtime_state.identity.stable_label
    identity_stable_conf = runtime_state.identity.stable_conf
    goblin_support_history = runtime_state.identity.goblin_support_history
    confirmed_goblin_hold_started_at = runtime_state.identity.confirmed_goblin_hold_started_at
    orange_stronger_streak = runtime_state.identity.orange_stronger_streak
    possible_goblin_alerted_this_session = runtime_state.identity.possible_goblin_alerted_this_session
    identity_presence = runtime_state.identity.presence
    video_native_fps = runtime_state.capture.video_native_fps
    video_skip_accumulator = runtime_state.capture.video_skip_accumulator
    rtsp_retrieve_failures = runtime_state.capture.rtsp_retrieve_failures
    identity_debug_writer = runtime_state.output.identity_debug_writer

    while True:
        now = time.perf_counter()
        if now < next_frame_at:
            if not config.is_video_source:
                # For live RTSP, keep grabbing during idle time so retrieve()
                # returns the freshest frame instead of a buffered stale frame.
                if not cap.grab():
                    print(
                        "Stream grab failed. The camera may have disconnected.",
                        file=sys.stderr,
                    )
                    runtime_state.lifecycle.reason = "stream_grab_failed"
                    break
            if not args.headless and cv2.waitKey(1) & 0xFF == ord("q"):
                runtime_state.lifecycle.reason = "quit_key"
                break
            continue

        if is_video_source:
            if video_native_fps > args.process_fps:
                # Keep local-file playback near real-time by dropping only the
                # extra frames required between processed frames.
                video_skip_accumulator += video_native_fps / args.process_fps
                span = int(video_skip_accumulator)
                frames_to_skip = max(span - 1, 0)
                if span > 0:
                    video_skip_accumulator -= span

                skip_failed = False
                for _ in range(frames_to_skip):
                    if not cap.grab():
                        skip_failed = True
                        break
                if skip_failed:
                    ok = False
                    frame = None
                else:
                    ok, frame = cap.read()
            else:
                ok, frame = cap.read()

            if not ok:
                if args.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    video_skip_accumulator = 0.0
                    previous_gray = None
                    motion_state = False
                    cat_present_state = False
                    cat_seen_streak = 0
                    identity_orange_acc = 0.0
                    identity_goblin_acc = 0.0
                    identity_support_frames = 0
                    identity_stable_label = "unknown"
                    identity_stable_conf = 0.0
                    goblin_support_history.clear()
                    confirmed_goblin_hold_started_at = 0.0
                    orange_stronger_streak = 0
                    possible_goblin_alerted_this_session = False
                    for state in identity_presence.values():
                        state.seen_streak = 0
                        state.last_seen_at = 0.0
                        state.present = False
                        state.started_at = 0.0
                        state.alert_fired = False
                    alert_fired = False
                    current_motion_snapshot_name = ""
                    continue
                print("Video ended.", file=sys.stderr)
                runtime_state.lifecycle.reason = "video_ended"
                break
        else:
            if not cap.grab():
                print(
                    "Stream grab failed. The camera may have disconnected.",
                    file=sys.stderr,
                )
                runtime_state.lifecycle.reason = "stream_grab_failed"
                break
            # If inference is slower than incoming RTSP frames, aggressively
            # drop additional buffered frames to keep latency near real-time.
            overdue = max(0.0, now - next_frame_at)
            extra_grabs = min(15, int(overdue / frame_interval) + 2)
            for _ in range(extra_grabs):
                if not cap.grab():
                    break
            ok, frame = cap.retrieve()
            if not ok:
                rtsp_retrieve_failures += 1
                if rtsp_retrieve_failures >= 30:
                    print(
                        "Stream retrieve failed repeatedly. The camera may have disconnected.",
                        file=sys.stderr,
                    )
                    runtime_state.lifecycle.reason = "stream_retrieve_failed"
                    break
                # Transient FFmpeg decode gaps can happen on RTSP; skip this tick.
                next_frame_at = now + frame_interval
                runtime_state.capture.next_frame_at = next_frame_at
                continue
            rtsp_retrieve_failures = 0

        next_frame_at = now + frame_interval
        runtime_state.capture.next_frame_at = next_frame_at

        frame_count += 1
        runtime_state.lifecycle.frame_count = frame_count
        elapsed = max(time.time() - started_at, 1e-6)
        fps = frame_count / elapsed
        frame_height, frame_width = frame.shape[:2]
        clip_frame = frame.copy()
        editor_state.frame_w = frame_width
        editor_state.frame_h = frame_height
        if zone_polygon_edit is not None:
            zone_polygon_px = zone_polygon_to_pixels(
                frame_width, frame_height, [(p[0], p[1]) for p in zone_polygon_edit]
            )
            x1, y1, x2, y2 = polygon_bbox(zone_polygon_px, frame_width, frame_height)
            zone_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.fillPoly(zone_mask, [zone_polygon_px], 255)
        else:
            x1, y1, x2, y2 = zone_to_pixels(
                frame_width, frame_height, tuple(zone_rect_edit)
            )
            zone_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.rectangle(zone_mask, (x1, y1), (x2, y2), 255, thickness=-1)
            zone_polygon_px = None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_percent = 0.0
        motion_active = False
        cat_detected_in_zone = False
        cat_detections: list[tuple[tuple[int, int, int, int], float]] = []
        cat_identities: list[tuple[str, float]] = []
        cat_identity_sample_label = "unknown"
        cat_identity_sample_conf = 0.0
        cat_identity_sample_orange = 0.0
        cat_identity_sample_goblin = 0.0
        cat_identity_sample_evidence: IdentityEvidence | None = None
        orange_detected_in_zone = False
        goblin_detected_in_zone = False
        goblin_support_frame = False
        cat_identity_in_zone = "unknown"
        cat_identity_conf = 0.0
        if previous_gray is not None:
            delta = cv2.absdiff(previous_gray, gray)
            zone_delta = delta[y1:y2, x1:x2]
            _, thresh = cv2.threshold(zone_delta, 25, 255, cv2.THRESH_BINARY)
            zone_mask_roi = zone_mask[y1:y2, x1:x2]
            masked_motion = cv2.bitwise_and(thresh, thresh, mask=zone_mask_roi)
            changed_pixels = cv2.countNonZero(masked_motion)
            total_pixels = max(cv2.countNonZero(zone_mask_roi), 1)
            motion_percent = changed_pixels * 100.0 / total_pixels
            motion_active = motion_percent >= args.motion_threshold
            should_run_cat = (
                detector is not None
                and (cat_detect_mode == "always" or motion_active)
            )
            if should_run_cat:
                cat_frame = preprocess_cat_frame(frame, args.cat_preprocess)
                try:
                    cat_detections = detect_cats(
                        detector,
                        detector_backend,
                        detector_device,
                        cat_frame,
                        args.cat_confidence,
                        args.cat_class_id,
                        args.cat_imgsz,
                    )
                except cv2.error as exc:
                    if detector_backend == "opencv_dnn" and detector_device == "cuda":
                        print(
                            "CUDA inference failed during forward(); falling back to CPU.",
                            file=sys.stderr,
                        )
                        print(f"OpenCV error: {exc}", file=sys.stderr)
                        detector_device = configure_detector_device(
                            detector, detector_backend, "cpu"
                        )
                        cat_detections = detect_cats(
                            detector,
                            detector_backend,
                            detector_device,
                            cat_frame,
                            args.cat_confidence,
                            args.cat_class_id,
                            args.cat_imgsz,
                        )
                    else:
                        raise
                if zone_polygon_px is not None:
                    zone_candidates: list[ZoneCandidate] = []
                    for box, score in cat_detections:
                        overlap_ratio = box_polygon_overlap_ratio(box, zone_polygon_px)
                        label, label_conf, evidence = classify_cat_identity(
                            frame,
                            box,
                            contamination_periphery_margin_max=args.id_goblin_periphery_margin_max,
                        )
                        cat_identities.append((label, label_conf))
                        if overlap_ratio >= args.cat_zone_overlap:
                            candidate = ZoneCandidate(
                                score=score,
                                label=label,
                                label_conf=label_conf,
                                evidence=evidence,
                                goblin_support_frame=is_goblin_support_frame(
                                    label,
                                    label_conf,
                                    evidence,
                                    min_conf=args.id_goblin_support_conf,
                                    min_margin=args.id_goblin_support_margin,
                                    min_torso_black_blobs=args.id_goblin_torso_black_blobs_min,
                                    min_torso_white_ratio=args.id_goblin_torso_white_min,
                                ),
                            )
                            zone_candidates.append(candidate)
                            if label == "orange" and label_conf >= 0.35:
                                orange_detected_in_zone = True
                            if candidate.goblin_support_frame:
                                goblin_detected_in_zone = True
                    cat_detected_in_zone = len(zone_candidates) > 0
                    if zone_candidates:
                        best_candidate = max(
                            zone_candidates,
                            key=lambda item: item.score * max(item.label_conf, 0.2),
                        )
                        cat_identity_sample_label = best_candidate.label
                        cat_identity_sample_conf = best_candidate.label_conf
                        cat_identity_sample_orange = best_candidate.evidence.orange_evidence
                        cat_identity_sample_goblin = best_candidate.evidence.goblin_evidence
                        cat_identity_sample_evidence = best_candidate.evidence
                        goblin_support_frame = best_candidate.goblin_support_frame
                else:
                    zone_candidates: list[ZoneCandidate] = []
                    for box, score in cat_detections:
                        overlap_ratio = box_zone_overlap_ratio(box, (x1, y1, x2, y2))
                        label, label_conf, evidence = classify_cat_identity(
                            frame,
                            box,
                            contamination_periphery_margin_max=args.id_goblin_periphery_margin_max,
                        )
                        cat_identities.append((label, label_conf))
                        if overlap_ratio >= args.cat_zone_overlap:
                            candidate = ZoneCandidate(
                                score=score,
                                label=label,
                                label_conf=label_conf,
                                evidence=evidence,
                                goblin_support_frame=is_goblin_support_frame(
                                    label,
                                    label_conf,
                                    evidence,
                                    min_conf=args.id_goblin_support_conf,
                                    min_margin=args.id_goblin_support_margin,
                                    min_torso_black_blobs=args.id_goblin_torso_black_blobs_min,
                                    min_torso_white_ratio=args.id_goblin_torso_white_min,
                                ),
                            )
                            zone_candidates.append(candidate)
                            if label == "orange" and label_conf >= 0.35:
                                orange_detected_in_zone = True
                            if candidate.goblin_support_frame:
                                goblin_detected_in_zone = True
                    cat_detected_in_zone = len(zone_candidates) > 0
                    if zone_candidates:
                        best_candidate = max(
                            zone_candidates,
                            key=lambda item: item.score * max(item.label_conf, 0.2),
                        )
                        cat_identity_sample_label = best_candidate.label
                        cat_identity_sample_conf = best_candidate.label_conf
                        cat_identity_sample_orange = best_candidate.evidence.orange_evidence
                        cat_identity_sample_goblin = best_candidate.evidence.goblin_evidence
                        cat_identity_sample_evidence = best_candidate.evidence
                        goblin_support_frame = best_candidate.goblin_support_frame

        previous_gray = gray

        event_now = time.time()
        if detector is not None:
            if cat_detected_in_zone:
                cat_seen_streak += 1
                cat_last_seen_at = event_now
            else:
                cat_seen_streak = 0

            cat_should_start = cat_seen_streak >= args.cat_enter_frames
            cat_should_hold = (
                cat_present_state
                and event_now - cat_last_seen_at <= args.cat_hold_seconds
            )
            cat_in_zone = cat_should_start or cat_should_hold

            if cat_in_zone:
                if cat_detected_in_zone:
                    identity_support_frames += 1
                    identity_orange_acc = identity_orange_acc * 0.90 + cat_identity_sample_orange
                    identity_goblin_acc = identity_goblin_acc * 0.90 + cat_identity_sample_goblin
                    goblin_support_history.append(goblin_support_frame)
                else:
                    # Keep identity stable during short detector dropouts.
                    identity_orange_acc *= 0.97
                    identity_goblin_acc *= 0.97

                min_support_frames = 2
                lock_margin = args.id_lock_margin
                switch_margin = args.id_switch_margin
                delta = identity_orange_acc - identity_goblin_acc
                dominant_acc = max(identity_orange_acc, identity_goblin_acc)

                if identity_support_frames >= min_support_frames:
                    if identity_stable_label == "orange":
                        if delta <= -switch_margin:
                            identity_stable_label = "white_black_dotted"
                    elif identity_stable_label == "white_black_dotted":
                        if delta >= switch_margin:
                            identity_stable_label = "orange"
                    else:
                        if delta >= lock_margin:
                            identity_stable_label = "orange"
                        elif delta <= -lock_margin:
                            identity_stable_label = "white_black_dotted"
                        elif (
                            cat_identity_sample_label == "orange"
                            and cat_identity_sample_conf >= 0.48
                            and delta >= -0.08
                            and identity_goblin_acc < 0.40
                        ):
                            identity_stable_label = "orange"
                        elif (
                            cat_identity_sample_label == "white_black_dotted"
                            and cat_identity_sample_conf >= 0.58
                            and delta <= 0.02
                            and identity_orange_acc < 0.35
                        ):
                            identity_stable_label = "white_black_dotted"

                if identity_stable_label == "unknown":
                    identity_stable_conf = max(
                        0.0,
                        min(
                            0.7,
                            max(cat_identity_sample_conf * 0.8, dominant_acc * 0.55),
                        ),
                    )
                else:
                    identity_stable_conf = min(
                        0.99,
                        0.45 + min(dominant_acc * 0.65, 0.35) + min(abs(delta), 0.25),
                    )

                if cat_detected_in_zone:
                    if (
                        cat_identity_sample_label == "orange"
                        and cat_identity_sample_conf >= 0.48
                        and cat_identity_sample_orange >= cat_identity_sample_goblin - 0.04
                    ):
                        orange_stronger_streak += 1
                    elif goblin_support_frame:
                        orange_stronger_streak = 0
                    else:
                        orange_stronger_streak = max(0, orange_stronger_streak - 1)
                if orange_stronger_streak >= args.id_orange_clear_streak:
                    goblin_support_history.clear()
                    confirmed_goblin_hold_started_at = 0.0
                recent_goblin_support = sum(1 for flag in goblin_support_history if flag)
                if (
                    recent_goblin_support >= args.id_confirmed_goblin_support_count
                    and len(goblin_support_history) >= args.id_confirmed_goblin_support_count
                    and identity_stable_label == "white_black_dotted"
                    and orange_stronger_streak < args.id_orange_clear_streak
                ):
                    if confirmed_goblin_hold_started_at == 0.0:
                        confirmed_goblin_hold_started_at = event_now
                else:
                    confirmed_goblin_hold_started_at = 0.0
                confirmed_goblin_detected_now = (
                    confirmed_goblin_hold_started_at > 0.0
                    and event_now - confirmed_goblin_hold_started_at
                    >= args.id_confirmed_goblin_hold_seconds
                )
                possible_goblin_detected_now = (
                    recent_goblin_support >= 2
                    and not confirmed_goblin_detected_now
                    and identity_stable_label == "white_black_dotted"
                    and orange_stronger_streak < args.id_orange_clear_streak
                )

                identity_detected_flags = {
                    "orange": orange_detected_in_zone,
                    "possible_goblin": possible_goblin_detected_now,
                    "confirmed_goblin": confirmed_goblin_detected_now,
                }
                identity_subject_map = {
                    "orange": "Orange cat in zone",
                    "possible_goblin": "Possible Goblin in zone",
                    "confirmed_goblin": "White-black cat in zone",
                }
                for identity_label, detected_now in identity_detected_flags.items():
                    state = identity_presence[identity_label]
                    previously_present = state.present
                    if detected_now:
                        state.seen_streak += 1
                        state.last_seen_at = event_now
                    else:
                        state.seen_streak = 0

                    min_start_frames = args.cat_enter_frames if identity_label == "orange" else 1
                    should_start_identity = state.seen_streak >= min_start_frames
                    should_hold_identity = (
                        identity_label != "possible_goblin"
                        and previously_present
                        and event_now - state.last_seen_at <= args.cat_hold_seconds
                    )
                    present_now = should_start_identity or should_hold_identity
                    state.present = present_now

                    if present_now and not previously_present:
                        state.started_at = event_now
                        state.alert_fired = False
                        print(
                            f"[{format_timestamp(event_now)}] "
                            f"{identity_subject_map[identity_label]} entered"
                        )
                    elif not present_now and previously_present:
                        print(
                            f"[{format_timestamp(event_now)}] "
                            f"{identity_subject_map[identity_label]} left "
                            f"(duration={event_now - state.started_at:.1f}s)"
                        )
                        state.alert_fired = False

                confirmed_present_now = identity_presence["confirmed_goblin"].present
                possible_present_now = identity_presence["possible_goblin"].present
                if confirmed_present_now or confirmed_goblin_detected_now:
                    cat_identity_in_zone = "confirmed_goblin"
                    cat_identity_conf = max(identity_stable_conf, args.id_goblin_support_conf)
                elif possible_present_now or possible_goblin_detected_now:
                    cat_identity_in_zone = "possible_goblin"
                    cat_identity_conf = max(identity_stable_conf, 0.58)
                else:
                    cat_identity_in_zone = identity_stable_label
                    cat_identity_conf = identity_stable_conf

                if identity_debug_writer is not None and cat_detected_in_zone:
                    source_seconds = (
                        float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                        if is_video_source
                        else max(0.0, event_now - run_started_at)
                    )
                    debug_evidence = cat_identity_sample_evidence
                    if debug_evidence is not None:
                        identity_debug_writer.writerow(
                            {
                                "source": source_text,
                                "frame_index": frame_count,
                                "source_seconds": f"{source_seconds:.3f}",
                                "wall_timestamp": format_timestamp(event_now),
                                "chosen_identity": cat_identity_sample_label,
                                "runtime_identity": cat_identity_in_zone,
                                "confidence": f"{cat_identity_sample_conf:.4f}",
                                "orange_evidence": f"{debug_evidence.orange_evidence:.4f}",
                                "goblin_evidence": f"{debug_evidence.goblin_evidence:.4f}",
                                "torso_core_white_ratio": (
                                    f"{debug_evidence.torso_core_white_ratio:.4f}"
                                ),
                                "torso_core_black_blob_count": (
                                    debug_evidence.torso_core_black_blob_count
                                ),
                                "goblin_support_frame": int(goblin_support_frame),
                                "contamination_downgraded": int(
                                    debug_evidence.contamination_downgraded
                                ),
                            }
                        )
            else:
                identity_orange_acc = 0.0
                identity_goblin_acc = 0.0
                identity_support_frames = 0
                identity_stable_label = "unknown"
                identity_stable_conf = 0.0
                goblin_support_history.clear()
                confirmed_goblin_hold_started_at = 0.0
                orange_stronger_streak = 0
                for identity_label, state in identity_presence.items():
                    if state.present:
                        subject = (
                            "Orange cat in zone"
                            if identity_label == "orange"
                            else (
                                "Possible Goblin in zone"
                                if identity_label == "possible_goblin"
                                else "White-black cat in zone"
                            )
                        )
                        print(
                            f"[{format_timestamp(event_now)}] {subject} left "
                            f"(duration={event_now - state.started_at:.1f}s)"
                        )
                    state.seen_streak = 0
                    state.last_seen_at = 0.0
                    state.present = False
                    state.started_at = 0.0
                    state.alert_fired = False
        else:
            cat_in_zone = False

        tracked_active = cat_in_zone if detector is not None else motion_active
        if detector is not None:
            orange_present_now = identity_presence["orange"].present
            possible_present_now = identity_presence["possible_goblin"].present
            confirmed_present_now = identity_presence["confirmed_goblin"].present
            if confirmed_present_now:
                current_dwell_subject = "White-black cat in zone"
            elif possible_present_now:
                current_dwell_subject = "Possible Goblin in zone"
            elif cat_identity_in_zone == "orange":
                current_dwell_subject = "Orange cat in zone"
            else:
                current_dwell_subject = "Cat in zone"
        else:
            current_dwell_subject = "Zone activity"

        if motion_active and not motion_state:
            motion_state = True
            motion_started_at = event_now
            current_motion_snapshot_name = ""
            alert_fired = False
            print(
                f"[{format_timestamp(event_now)}] Motion started "
                f"(zone={motion_percent:.2f}%)"
            )
            if detector is None and not args.no_snapshots and (
                args.snapshot_cooldown == 0
                or event_now - last_snapshot_at >= args.snapshot_cooldown
            ):
                snapshot_name = datetime.fromtimestamp(event_now).strftime(
                    "motion_%Y%m%d_%H%M%S.jpg"
                )
                snapshot_path = snapshot_dir / snapshot_name
                if cv2.imwrite(str(snapshot_path), frame):
                    last_snapshot_at = event_now
                    current_motion_snapshot_name = snapshot_name
                    print(f"[{format_timestamp(event_now)}] Snapshot saved: {snapshot_path}")
                else:
                    print(
                        f"[{format_timestamp(event_now)}] Failed to save snapshot: {snapshot_path}",
                        file=sys.stderr,
                    )
        elif not motion_active and motion_state:
            motion_state = False
            print(
                f"[{format_timestamp(event_now)}] Motion ended "
                f"(duration={event_now - motion_started_at:.1f}s)"
            )
        if detector is not None and cat_in_zone and not cat_present_state:
            cat_present_state = True
            cat_started_at = event_now
            alert_fired = False
            print(
                f"[{format_timestamp(event_now)}] Cat entered zone "
                f"(identity={cat_identity_in_zone} conf={cat_identity_conf:.2f})"
            )
            if not args.no_snapshots and (
                args.snapshot_cooldown == 0
                or event_now - last_snapshot_at >= args.snapshot_cooldown
            ):
                snapshot_name = datetime.fromtimestamp(event_now).strftime(
                    "cat_%Y%m%d_%H%M%S.jpg"
                )
                snapshot_path = snapshot_dir / snapshot_name
                if cv2.imwrite(str(snapshot_path), frame):
                    last_snapshot_at = event_now
                    current_motion_snapshot_name = snapshot_name
                    print(f"[{format_timestamp(event_now)}] Snapshot saved: {snapshot_path}")
                else:
                    print(
                        f"[{format_timestamp(event_now)}] Failed to save snapshot: {snapshot_path}",
                        file=sys.stderr,
                    )
        elif detector is not None and not cat_in_zone and cat_present_state:
            cat_present_state = False
            identity_orange_acc = 0.0
            identity_goblin_acc = 0.0
            identity_support_frames = 0
            identity_stable_label = "unknown"
            identity_stable_conf = 0.0
            goblin_support_history.clear()
            confirmed_goblin_hold_started_at = 0.0
            orange_stronger_streak = 0
            possible_goblin_alerted_this_session = False
            print(
                f"[{format_timestamp(event_now)}] Cat left zone "
                f"(duration={event_now - cat_started_at:.1f}s)"
            )

        if not tracked_active:
            alert_fired = False
            if not motion_state:
                current_motion_snapshot_name = ""

        motion_duration = event_now - motion_started_at if motion_state else 0.0
        dwell_duration = event_now - cat_started_at if cat_present_state else 0.0
        tracked_duration = dwell_duration if detector is not None else motion_duration
        if detector is not None:
            alert_fired_any = False
            for identity_label, alert_cat, subject in (
                ("orange", "ORANGE", "Orange cat in zone"),
                ("possible_goblin", "POSSIBLE_GOBLIN", "Cat in zone"),
                ("confirmed_goblin", "GOBLIN", "White-black cat in zone"),
            ):
                state = identity_presence[identity_label]
                if not state.present:
                    state.alert_fired = False
                    continue
                identity_duration = event_now - state.started_at
                if identity_label == "possible_goblin":
                    if (
                        possible_goblin_alerted_this_session
                        or state.alert_fired
                        or identity_duration < args.possible_goblin_seconds
                    ):
                        continue
                    state.alert_fired = True
                    possible_goblin_alerted_this_session = True
                    alert_fired_any = True
                    emit_alert(
                        config.event_log_dir,
                        runtime_state.output,
                        args,
                        frame_width,
                        frame_height,
                        subject,
                        alert_cat,
                        identity_duration,
                        event_now,
                        current_motion_snapshot_name,
                        event_prefix="POSSIBLE_GOBLIN",
                    )
                    continue
                if state.alert_fired or identity_duration < args.alert_seconds:
                    continue
                state.alert_fired = True
                alert_fired_any = True
                emit_alert(
                    config.event_log_dir,
                    runtime_state.output,
                    args,
                    frame_width,
                    frame_height,
                    subject,
                    alert_cat,
                    identity_duration,
                    event_now,
                    current_motion_snapshot_name,
                )
            alert_fired = alert_fired_any
        elif tracked_active and not alert_fired and tracked_duration >= args.alert_seconds:
            alert_fired = True
            emit_alert(
                config.event_log_dir,
                runtime_state.output,
                args,
                frame_width,
                frame_height,
                "Zone activity",
                "MOTION",
                tracked_duration,
                event_now,
                current_motion_snapshot_name,
            )

        preview_observation = FrameObservation(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            zone_mask=zone_mask,
            zone_polygon_px=zone_polygon_px,
            motion_percent=motion_percent,
            motion_active=motion_active,
            cat_detected_in_zone=cat_detected_in_zone,
            cat_detections=cat_detections,
            cat_identities=cat_identities,
            cat_identity_sample_label=cat_identity_sample_label,
            cat_identity_sample_conf=cat_identity_sample_conf,
            cat_identity_sample_orange=cat_identity_sample_orange,
            cat_identity_sample_goblin=cat_identity_sample_goblin,
            cat_identity_sample_evidence=cat_identity_sample_evidence,
            orange_detected_in_zone=orange_detected_in_zone,
            goblin_detected_in_zone=goblin_detected_in_zone,
            goblin_support_frame=goblin_support_frame,
        )
        preview_identity_result = (
            IdentityFrameResult(
                runtime_identity=cat_identity_in_zone,
                runtime_conf=cat_identity_conf,
                chosen_label=cat_identity_sample_label,
                chosen_conf=cat_identity_sample_conf,
                evidence=cat_identity_sample_evidence,
                goblin_support_frame=goblin_support_frame,
                detected_flags={},
                presence_flags={},
            )
            if detector is not None
            else None
        )
        render_preview(
            frame,
            args,
            detector_runtime,
            preview_observation,
            runtime_state.identity,
            preview_identity_result,
            tracked_active,
            cat_in_zone,
            current_dwell_subject,
            tracked_duration,
            fps,
            alert_fired,
            zone_polygon_edit,
            cat_seen_streak,
        )

        if runtime_state.output.clip_writer is not None:
            runtime_state.output.clip_writer.write(clip_frame)
            if event_now >= runtime_state.output.clip_end_at:
                close_clip_writer(runtime_state.output, event_now)

        if not args.headless:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("p") and args.zone_edit:
                print_zone_value(zone_rect_edit, zone_polygon_edit)
            if key == ord("q"):
                runtime_state.lifecycle.reason = "quit_key"
                break

    close_clip_writer(runtime_state.output, time.time())
    if runtime_state.output.identity_debug_file is not None:
        runtime_state.output.identity_debug_file.close()
    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    finalize_run_log(config.event_log_dir, runtime_state.lifecycle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
