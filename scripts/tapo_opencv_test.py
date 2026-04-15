import argparse
import atexit
import os
import sys
import time
from dataclasses import dataclass
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
    low_light: bool
    orange_evidence: float
    goblin_evidence: float


def classify_cat_identity(
    frame: np.ndarray, box: tuple[int, int, int, int]
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
        empty = IdentityEvidence(0.0, 0.0, 0.0, 0, False, 0.0, 0.0)
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
        empty = IdentityEvidence(0.0, 0.0, 0.0, 0, False, 0.0, 0.0)
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
    mean_v = float(np.mean(v[valid_mask])) if np.any(valid_mask) else 0.0
    low_light = mean_v < 92.0
    white_mask = valid_mask & (s < 65) & (v > 95)
    black_mask = valid_mask & (v < 62)
    orange_mask = valid_mask & (h >= 4) & (h <= 26) & (s > 85) & (v > 50)

    white_ratio = float(np.count_nonzero(white_mask)) / valid_pixels
    black_ratio = float(np.count_nonzero(black_mask)) / valid_pixels
    orange_ratio = float(np.count_nonzero(orange_mask)) / valid_pixels

    # Count meaningful dark blobs (patch-like marks), not just tiny noise pixels.
    black_u8 = (black_mask.astype(np.uint8) * 255)
    comp_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        black_u8, connectivity=8
    )
    min_blob_area = max(6, int(valid_pixels * 0.006))
    black_blob_count = 0
    for comp_idx in range(1, comp_count):
        area = int(stats[comp_idx, cv2.CC_STAT_AREA])
        if area >= min_blob_area:
            black_blob_count += 1

    orange_evidence = max(
        0.0,
        orange_ratio * 2.2
        - black_ratio * 1.4
        - max(white_ratio - orange_ratio * 1.2, 0.0) * 0.35
        + (0.08 if black_blob_count == 0 else 0.0)
        - (0.12 if black_blob_count >= 2 else 0.0),
    )
    goblin_evidence = max(
        0.0,
        white_ratio * 1.3
        + black_ratio * 2.3
        + min(black_blob_count * 0.07, 0.21)
        - orange_ratio * 1.0
        + (0.05 if (low_light and black_blob_count >= 2) else 0.0)
        - (0.12 if orange_ratio > 0.22 else 0.0),
    )
    evidence = IdentityEvidence(
        orange_ratio=orange_ratio,
        white_ratio=white_ratio,
        black_ratio=black_ratio,
        black_blob_count=black_blob_count,
        low_light=low_light,
        orange_evidence=min(1.0, orange_evidence),
        goblin_evidence=min(1.0, goblin_evidence),
    )

    # In low-light warm scenes (e.g. yellow TV cast), Goblin may look orange-ish.
    # Require stronger patch evidence so Orange does not flip from weak dark noise.
    goblin_low_light_patchy = (
        low_light
        and black_ratio >= 0.045
        and black_blob_count >= 2
        and white_ratio >= 0.10
        and orange_ratio <= 0.23
        and black_ratio >= orange_ratio * 0.38
    )
    if goblin_low_light_patchy:
        score = min(
            0.99,
            0.38
            + min(white_ratio * 1.1, 0.22)
            + min(black_ratio * 3.6, 0.38)
            + min(black_blob_count * 0.09, 0.18),
        )
        return "white_black_dotted", score, evidence

    # Orange guard: require weak black evidence before forcing Orange.
    if (
        orange_ratio >= 0.12
        and orange_ratio >= white_ratio * 1.1
        and black_ratio < 0.05
        and black_blob_count <= 1
    ):
        score = min(0.99, 0.42 + orange_ratio * 2.5 - min(black_ratio * 0.7, 0.12))
        return "orange", score, evidence

    # Goblin needs visible white+black coat structure.
    goblin_patchy = (
        white_ratio >= 0.15
        and black_ratio >= 0.04
        and black_blob_count >= 2
        and orange_ratio <= 0.26
    )
    goblin_high_contrast = (
        white_ratio >= 0.23
        and black_ratio >= 0.05
        and black_blob_count >= 1
        and orange_ratio <= 0.30
    )
    if goblin_patchy or goblin_high_contrast:
        score = min(
            0.99,
            0.4
            + min(white_ratio * 1.3, 0.3)
            + min(black_ratio * 3.2, 0.35)
            + min(black_blob_count * 0.08, 0.16),
        )
        return "white_black_dotted", score, evidence

    # Only call Orange when black evidence is weak.
    if (
        orange_ratio >= 0.14
        and orange_ratio >= white_ratio * 0.78
        and (black_ratio < 0.05 or white_ratio < 0.12)
        and black_blob_count <= 1
    ):
        score = min(0.99, 0.4 + orange_ratio * 2.6)
        return "orange", score, evidence

    # Soft Orange fallback: useful when warm casts dilute saturation and we would
    # otherwise remain unknown despite clearly weak Goblin evidence.
    if (
        orange_evidence >= max(goblin_evidence * 1.25, 0.18)
        and black_blob_count <= 1
        and black_ratio < 0.07
    ):
        score = min(0.92, 0.34 + orange_evidence * 0.9)
        return "orange", score, evidence

    return "unknown", max(0.0, min(0.6, orange_ratio + white_ratio * 0.4)), evidence


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    args = parse_args()

    if args.video and args.url:
        print("Use either --video or --url, not both.", file=sys.stderr)
        return 1

    rtsp_url = args.url or os.environ.get("RTSP_URL")
    video_path: Path | None = None
    is_video_source = bool(args.video)
    if args.video:
        candidate = Path(args.video)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        video_path = candidate
        if not video_path.exists():
            print(f"Video file not found: {video_path}", file=sys.stderr)
            return 1
    elif not rtsp_url:
        print(
            "Missing input source. Set RTSP_URL in .env, pass --url, or pass --video.",
            file=sys.stderr,
        )
        return 1

    try:
        zone = parse_zone(args.zone)
    except ValueError as exc:
        print(f"Invalid --zone value: {exc}", file=sys.stderr)
        return 1
    zone_polygon_norm: list[tuple[float, float]] | None = None
    if args.zone_polygon:
        try:
            zone_polygon_norm = parse_zone_polygon(args.zone_polygon)
        except ValueError as exc:
            print(f"Invalid --zone-polygon value: {exc}", file=sys.stderr)
            return 1
        print(
            "Using polygon zone: "
            + ";".join(f"{x:.4f},{y:.4f}" for x, y in zone_polygon_norm)
        )
    else:
        print(
            "Using rectangle zone: "
            f"{zone[0]:.4f},{zone[1]:.4f},{zone[2]:.4f},{zone[3]:.4f}"
        )
    zone_rect_edit = [zone[0], zone[1], zone[2], zone[3]]
    zone_polygon_edit = (
        [[x, y] for x, y in zone_polygon_norm] if zone_polygon_norm is not None else None
    )
    if args.process_fps <= 0:
        print("--process-fps must be greater than 0.", file=sys.stderr)
        return 1
    if args.snapshot_cooldown < 0:
        print("--snapshot-cooldown must be 0 or greater.", file=sys.stderr)
        return 1
    if args.alert_seconds < 0:
        print("--alert-seconds must be 0 or greater.", file=sys.stderr)
        return 1
    if args.clip_seconds <= 0:
        print("--clip-seconds must be greater than 0.", file=sys.stderr)
        return 1
    if not 0 <= args.cat_confidence <= 1:
        print("--cat-confidence must be between 0 and 1.", file=sys.stderr)
        return 1
    if args.cat_enter_frames <= 0:
        print("--cat-enter-frames must be greater than 0.", file=sys.stderr)
        return 1
    if not 0 <= args.cat_zone_overlap <= 1:
        print("--cat-zone-overlap must be between 0 and 1.", file=sys.stderr)
        return 1
    if args.cat_imgsz <= 0:
        print("--cat-imgsz must be greater than 0.", file=sys.stderr)
        return 1
    if args.cat_hold_seconds < 0:
        print("--cat-hold-seconds must be 0 or greater.", file=sys.stderr)
        return 1

    snapshot_dir = repo_root / args.snapshot_dir
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    event_log_dir = repo_root / "event-log"
    event_log_dir.mkdir(parents=True, exist_ok=True)
    cat_model_path = None
    if args.cat_model:
        cat_model_path = str((repo_root / args.cat_model).resolve())

    if is_video_source:
        cap = cv2.VideoCapture(str(video_path))
    else:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        if is_video_source:
            print(
                f"OpenCV could not open video file: {video_path}",
                file=sys.stderr,
            )
        else:
            print(
                "OpenCV could not open the RTSP stream. Check the camera IP, "
                "credentials, and whether OpenCV has FFmpeg support.",
                file=sys.stderr,
            )
        return 1
    if not is_video_source:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_native_fps = 0.0
    video_skip_accumulator = 0.0
    if is_video_source:
        raw_video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if np.isfinite(raw_video_fps) and raw_video_fps > 0:
            video_native_fps = raw_video_fps
    try:
        detector, detector_backend = load_detector(cat_model_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except cv2.error as exc:
        print(f"Failed to load cat model: {exc}", file=sys.stderr)
        return 1
    detector_device = configure_detector_device(detector, detector_backend, args.device)
    if args.cat_detect_mode == "auto":
        cat_detect_mode = "always" if is_video_source else "motion"
    else:
        cat_detect_mode = args.cat_detect_mode

    frame_count = 0
    started_at = time.time()
    previous_gray = None
    frame_interval = 1.0 / args.process_fps
    next_frame_at = time.perf_counter()
    motion_state = False
    motion_started_at = 0.0
    last_snapshot_at = 0.0
    current_motion_snapshot_name = ""
    alert_fired = False
    cat_present_state = False
    cat_started_at = 0.0
    cat_seen_streak = 0
    cat_last_seen_at = 0.0
    identity_orange_acc = 0.0
    identity_goblin_acc = 0.0
    identity_support_frames = 0
    identity_stable_label = "unknown"
    identity_stable_conf = 0.0
    identity_presence: dict[str, dict[str, float | int | bool]] = {
        "orange": {
            "seen_streak": 0,
            "last_seen_at": 0.0,
            "present": False,
            "started_at": 0.0,
            "alert_fired": False,
        },
        "white_black_dotted": {
            "seen_streak": 0,
            "last_seen_at": 0.0,
            "present": False,
            "started_at": 0.0,
            "alert_fired": False,
        },
    }
    current_dwell_subject = "Zone activity"
    zone_mask: np.ndarray | None = None
    zone_polygon_px: np.ndarray | None = None
    rtsp_retrieve_failures = 0
    clip_writer: cv2.VideoWriter | None = None
    clip_end_at = 0.0
    active_clip_path: Path | None = None
    clip_dir = snapshot_dir / "clips"
    if args.save_clip_on_alert:
        clip_dir.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, args.width, args.height)

    editor_state: dict[str, object] = {
        "frame_w": 0,
        "frame_h": 0,
        "dragging": False,
        "drag_kind": "",
        "start_mouse": (0, 0),
        "start_rect": (0, 0, 0, 0),
        "start_polygon": [],
        "active_point": -1,
    }

    def print_zone_value() -> None:
        if zone_polygon_edit is not None:
            text = ";".join(f"{p[0]:.4f},{p[1]:.4f}" for p in zone_polygon_edit)
            print(f"ZONE_POLYGON={text}")
        else:
            print(
                "ZONE="
                f"{zone_rect_edit[0]:.4f},{zone_rect_edit[1]:.4f},"
                f"{zone_rect_edit[2]:.4f},{zone_rect_edit[3]:.4f}"
            )

    def emit_alert(
        subject: str, alert_cat: str, duration_s: float, event_ts: float
    ) -> None:
        nonlocal clip_writer, active_clip_path, clip_end_at
        alert_message = (
            f"[{format_timestamp(event_ts)}] ALERT: {subject.lower()} lasted "
            f"{duration_s:.1f}s"
        )
        if current_motion_snapshot_name:
            alert_message += f" snapshot={current_motion_snapshot_name}"
        print(alert_message)
        append_event_log(event_log_dir, alert_message, event_ts)

        if args.save_clip_on_alert and clip_writer is None:
            clip_name = (
                "ALERT_"
                + datetime.fromtimestamp(event_ts).strftime("%Y-%m-%d_%H%M%S")
                + f"_{alert_cat}.mp4"
            )
            clip_path = clip_dir / clip_name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            clip_writer = cv2.VideoWriter(
                str(clip_path),
                fourcc,
                max(args.process_fps, 1.0),
                (frame_width, frame_height),
            )
            if clip_writer.isOpened():
                active_clip_path = clip_path
                clip_end_at = event_ts + args.clip_seconds
                print(f"[{format_timestamp(event_ts)}] Clip recording started: {clip_path}")
            else:
                clip_writer.release()
                clip_writer = None
                active_clip_path = None
                print(
                    f"[{format_timestamp(event_ts)}] Failed to start clip writer: {clip_path}",
                    file=sys.stderr,
                )

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if not args.zone_edit:
            return
        frame_w = int(editor_state["frame_w"])
        frame_h = int(editor_state["frame_h"])
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
                    editor_state["dragging"] = True
                    editor_state["drag_kind"] = "poly_point"
                    editor_state["active_point"] = best_idx
                else:
                    inside = cv2.pointPolygonTest(
                        points_px.astype(np.float32), (float(x), float(y)), False
                    )
                    if inside >= 0:
                        editor_state["dragging"] = True
                        editor_state["drag_kind"] = "poly_move"
                        editor_state["start_mouse"] = (x, y)
                        editor_state["start_polygon"] = [
                            [p[0], p[1]] for p in zone_polygon_edit
                        ]
            elif event == cv2.EVENT_MOUSEMOVE and bool(editor_state["dragging"]):
                drag_kind = str(editor_state["drag_kind"])
                if drag_kind == "poly_point":
                    idx = int(editor_state["active_point"])
                    if 0 <= idx < len(zone_polygon_edit):
                        zone_polygon_edit[idx][0] = clamp01(x / max(frame_w - 1, 1))
                        zone_polygon_edit[idx][1] = clamp01(y / max(frame_h - 1, 1))
                elif drag_kind == "poly_move":
                    smx, smy = editor_state["start_mouse"]  # type: ignore[assignment]
                    dx = (x - int(smx)) / max(frame_w - 1, 1)
                    dy = (y - int(smy)) / max(frame_h - 1, 1)
                    start_poly = editor_state["start_polygon"]  # type: ignore[assignment]
                    for i, pt in enumerate(start_poly):
                        zone_polygon_edit[i][0] = clamp01(float(pt[0]) + dx)
                        zone_polygon_edit[i][1] = clamp01(float(pt[1]) + dy)
            elif event == cv2.EVENT_LBUTTONUP and bool(editor_state["dragging"]):
                editor_state["dragging"] = False
                editor_state["drag_kind"] = ""
                editor_state["active_point"] = -1
                editor_state["start_polygon"] = []
                print_zone_value()
            return

        x1, y1, x2, y2 = zone_to_pixels(frame_w, frame_h, tuple(zone_rect_edit))
        if event == cv2.EVENT_LBUTTONDOWN:
            near_corner = abs(x - x2) <= 14 and abs(y - y2) <= 14
            in_rect = x1 <= x <= x2 and y1 <= y <= y2
            if near_corner:
                editor_state["dragging"] = True
                editor_state["drag_kind"] = "rect_resize"
                editor_state["start_rect"] = (x1, y1, x2, y2)
            elif in_rect:
                editor_state["dragging"] = True
                editor_state["drag_kind"] = "rect_move"
                editor_state["start_mouse"] = (x, y)
                editor_state["start_rect"] = (x1, y1, x2, y2)
        elif event == cv2.EVENT_MOUSEMOVE and bool(editor_state["dragging"]):
            drag_kind = str(editor_state["drag_kind"])
            sx1, sy1, sx2, sy2 = editor_state["start_rect"]  # type: ignore[assignment]
            if drag_kind == "rect_move":
                smx, smy = editor_state["start_mouse"]  # type: ignore[assignment]
                dx = x - int(smx)
                dy = y - int(smy)
                w = int(sx2) - int(sx1)
                h = int(sy2) - int(sy1)
                nx1 = max(0, min(int(sx1) + dx, frame_w - max(1, w)))
                ny1 = max(0, min(int(sy1) + dy, frame_h - max(1, h)))
                nx2 = nx1 + w
                ny2 = ny1 + h
                zone_rect_edit[:] = pixels_to_zone(frame_w, frame_h, nx1, ny1, nx2, ny2)
            elif drag_kind == "rect_resize":
                nx2 = max(int(sx1) + 12, min(x, frame_w - 1))
                ny2 = max(int(sy1) + 12, min(y, frame_h - 1))
                zone_rect_edit[:] = pixels_to_zone(
                    frame_w, frame_h, int(sx1), int(sy1), nx2, ny2
                )
        elif event == cv2.EVENT_LBUTTONUP and bool(editor_state["dragging"]):
            editor_state["dragging"] = False
            editor_state["drag_kind"] = ""
            print_zone_value()

    if args.zone_edit:
        cv2.setMouseCallback(args.window_name, on_mouse)

    if detector is None:
        mode = "motion-only"
    else:
        mode = f"cat-in-zone ({cat_detect_mode})"
    source_text = str(video_path) if is_video_source else "RTSP stream"
    print(
        f"Connected to {source_text}. Processing at {args.process_fps:.2f} FPS in "
        f"{mode} mode using {detector_backend} on {detector_device.upper()}. "
        "Press q to quit."
    )
    run_started_at = time.time()
    run_state = {"reason": "normal_exit", "ended": False}
    launch_origin = (args.launch_origin or "manual").strip().lower().replace(" ", "_")
    if not launch_origin:
        launch_origin = "manual"

    def log_run_end() -> None:
        if run_state["ended"]:
            return
        run_ended_at = time.time()
        run_end_message = (
            f"[{format_timestamp(run_ended_at)}] APP_END: reason={run_state['reason']} "
            f"duration={run_ended_at - run_started_at:.1f}s frames={frame_count}"
        )
        append_event_log(event_log_dir, run_end_message, run_ended_at)
        print(run_end_message)
        run_state["ended"] = True

    atexit.register(log_run_end)
    run_start_message = (
        f"[{format_timestamp(run_started_at)}] APP_START: "
        f"source={source_text} mode={mode} fps={args.process_fps:.2f} "
        f"launch_origin={launch_origin}"
    )
    append_event_log(event_log_dir, run_start_message, run_started_at)
    print(run_start_message)

    while True:
        now = time.perf_counter()
        if now < next_frame_at:
            if not is_video_source:
                # For live RTSP, keep grabbing during idle time so retrieve()
                # returns the freshest frame instead of a buffered stale frame.
                if not cap.grab():
                    print(
                        "Stream grab failed. The camera may have disconnected.",
                        file=sys.stderr,
                    )
                    run_state["reason"] = "stream_grab_failed"
                    break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                run_state["reason"] = "quit_key"
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
                    for state in identity_presence.values():
                        state["seen_streak"] = 0
                        state["last_seen_at"] = 0.0
                        state["present"] = False
                        state["started_at"] = 0.0
                        state["alert_fired"] = False
                    alert_fired = False
                    current_motion_snapshot_name = ""
                    continue
                print("Video ended.", file=sys.stderr)
                run_state["reason"] = "video_ended"
                break
        else:
            if not cap.grab():
                print(
                    "Stream grab failed. The camera may have disconnected.",
                    file=sys.stderr,
                )
                run_state["reason"] = "stream_grab_failed"
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
                    run_state["reason"] = "stream_retrieve_failed"
                    break
                # Transient FFmpeg decode gaps can happen on RTSP; skip this tick.
                next_frame_at = now + frame_interval
                continue
            rtsp_retrieve_failures = 0

        next_frame_at = now + frame_interval

        frame_count += 1
        elapsed = max(time.time() - started_at, 1e-6)
        fps = frame_count / elapsed
        frame_height, frame_width = frame.shape[:2]
        clip_frame = frame.copy()
        editor_state["frame_w"] = frame_width
        editor_state["frame_h"] = frame_height
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
        orange_detected_in_zone = False
        goblin_detected_in_zone = False
        goblin_detect_conf_min = 0.62
        goblin_detect_margin_min = 0.12
        goblin_detect_evidence_min = 0.28
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
                    zone_candidates: list[tuple[float, str, float, float, float]] = []
                    for box, score in cat_detections:
                        overlap_ratio = box_polygon_overlap_ratio(box, zone_polygon_px)
                        label, label_conf, evidence = classify_cat_identity(frame, box)
                        cat_identities.append((label, label_conf))
                        if overlap_ratio >= args.cat_zone_overlap:
                            zone_candidates.append(
                                (
                                    score,
                                    label,
                                    label_conf,
                                    evidence.orange_evidence,
                                    evidence.goblin_evidence,
                                )
                            )
                            if label == "orange" and label_conf >= 0.35:
                                orange_detected_in_zone = True
                            elif (
                                label == "white_black_dotted"
                                and label_conf >= goblin_detect_conf_min
                                and evidence.goblin_evidence >= goblin_detect_evidence_min
                                and evidence.goblin_evidence
                                >= evidence.orange_evidence + goblin_detect_margin_min
                            ):
                                goblin_detected_in_zone = True
                    cat_detected_in_zone = len(zone_candidates) > 0
                    if zone_candidates:
                        _score, label, label_conf, orange_ev, goblin_ev = max(
                            zone_candidates, key=lambda item: item[0] * max(item[2], 0.2)
                        )
                        cat_identity_sample_label = label
                        cat_identity_sample_conf = label_conf
                        cat_identity_sample_orange = orange_ev
                        cat_identity_sample_goblin = goblin_ev
                else:
                    zone_candidates = []
                    for box, score in cat_detections:
                        overlap_ratio = box_zone_overlap_ratio(box, (x1, y1, x2, y2))
                        label, label_conf, evidence = classify_cat_identity(frame, box)
                        cat_identities.append((label, label_conf))
                        if overlap_ratio >= args.cat_zone_overlap:
                            zone_candidates.append(
                                (
                                    score,
                                    label,
                                    label_conf,
                                    evidence.orange_evidence,
                                    evidence.goblin_evidence,
                                )
                            )
                            if label == "orange" and label_conf >= 0.35:
                                orange_detected_in_zone = True
                            elif (
                                label == "white_black_dotted"
                                and label_conf >= goblin_detect_conf_min
                                and evidence.goblin_evidence >= goblin_detect_evidence_min
                                and evidence.goblin_evidence
                                >= evidence.orange_evidence + goblin_detect_margin_min
                            ):
                                goblin_detected_in_zone = True
                    cat_detected_in_zone = len(zone_candidates) > 0
                    if zone_candidates:
                        _score, label, label_conf, orange_ev, goblin_ev = max(
                            zone_candidates, key=lambda item: item[0] * max(item[2], 0.2)
                        )
                        cat_identity_sample_label = label
                        cat_identity_sample_conf = label_conf
                        cat_identity_sample_orange = orange_ev
                        cat_identity_sample_goblin = goblin_ev

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
                else:
                    # Keep identity stable during short detector dropouts.
                    identity_orange_acc *= 0.97
                    identity_goblin_acc *= 0.97

                min_support_frames = 2
                lock_margin = 0.11
                switch_margin = 0.24
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

                cat_identity_in_zone = identity_stable_label
                cat_identity_conf = identity_stable_conf

                identity_detected_flags = {
                    "orange": orange_detected_in_zone,
                    "white_black_dotted": goblin_detected_in_zone,
                }
                identity_subject_map = {
                    "orange": "Orange cat in zone",
                    "white_black_dotted": "White-black cat in zone",
                }
                for identity_label, detected_now in identity_detected_flags.items():
                    state = identity_presence[identity_label]
                    previously_present = bool(state["present"])
                    if detected_now:
                        state["seen_streak"] = int(state["seen_streak"]) + 1
                        state["last_seen_at"] = event_now
                    else:
                        state["seen_streak"] = 0

                    min_start_frames = (
                        max(args.cat_enter_frames, 3)
                        if identity_label == "white_black_dotted"
                        else args.cat_enter_frames
                    )
                    should_start_identity = int(state["seen_streak"]) >= min_start_frames
                    should_hold_identity = previously_present and (
                        event_now - float(state["last_seen_at"]) <= args.cat_hold_seconds
                    )
                    present_now = should_start_identity or should_hold_identity
                    state["present"] = present_now

                    if present_now and not previously_present:
                        state["started_at"] = event_now
                        state["alert_fired"] = False
                        print(
                            f"[{format_timestamp(event_now)}] "
                            f"{identity_subject_map[identity_label]} entered"
                        )
                    elif not present_now and previously_present:
                        print(
                            f"[{format_timestamp(event_now)}] "
                            f"{identity_subject_map[identity_label]} left "
                            f"(duration={event_now - float(state['started_at']):.1f}s)"
                        )
                        state["alert_fired"] = False
            else:
                identity_orange_acc = 0.0
                identity_goblin_acc = 0.0
                identity_support_frames = 0
                identity_stable_label = "unknown"
                identity_stable_conf = 0.0
                for identity_label, state in identity_presence.items():
                    if bool(state["present"]):
                        subject = (
                            "Orange cat in zone"
                            if identity_label == "orange"
                            else "White-black cat in zone"
                        )
                        print(
                            f"[{format_timestamp(event_now)}] {subject} left "
                            f"(duration={event_now - float(state['started_at']):.1f}s)"
                        )
                    state["seen_streak"] = 0
                    state["last_seen_at"] = 0.0
                    state["present"] = False
                    state["started_at"] = 0.0
                    state["alert_fired"] = False
        else:
            cat_in_zone = False

        tracked_active = cat_in_zone if detector is not None else motion_active
        if detector is not None:
            orange_present_now = bool(identity_presence["orange"]["present"])
            goblin_present_now = bool(identity_presence["white_black_dotted"]["present"])
            if orange_present_now and goblin_present_now:
                current_dwell_subject = "Multiple cats in zone"
            elif cat_identity_in_zone == "orange":
                current_dwell_subject = "Orange cat in zone"
            elif cat_identity_in_zone == "white_black_dotted":
                current_dwell_subject = "White-black cat in zone"
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
                ("white_black_dotted", "GOBLIN", "White-black cat in zone"),
            ):
                state = identity_presence[identity_label]
                if not bool(state["present"]):
                    state["alert_fired"] = False
                    continue
                identity_duration = event_now - float(state["started_at"])
                if bool(state["alert_fired"]) or identity_duration < args.alert_seconds:
                    continue
                state["alert_fired"] = True
                alert_fired_any = True
                emit_alert(subject, alert_cat, identity_duration, event_now)
            alert_fired = alert_fired_any
        elif tracked_active and not alert_fired and tracked_duration >= args.alert_seconds:
            alert_fired = True
            emit_alert("Zone activity", "MOTION", tracked_duration, event_now)

        zone_color = (0, 0, 255) if tracked_active else (0, 255, 255)
        if zone_polygon_px is not None:
            cv2.polylines(
                frame, [zone_polygon_px], isClosed=True, color=zone_color, thickness=2
            )
            if args.zone_edit:
                for pt in zone_polygon_px:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 255, 255), -1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), zone_color, 2)
        for idx, ((bx1, by1, bx2, by2), score) in enumerate(cat_detections):
            color = (255, 0, 0)
            thickness = 2
            overlap_ratio = (
                box_polygon_overlap_ratio((bx1, by1, bx2, by2), zone_polygon_px)
                if zone_polygon_px is not None
                else box_zone_overlap_ratio((bx1, by1, bx2, by2), (x1, y1, x2, y2))
            )
            in_zone = overlap_ratio >= args.cat_zone_overlap
            if in_zone:
                color = (0, 0, 255)
                thickness = 3
            identity_label = "unknown"
            identity_conf = 0.0
            if idx < len(cat_identities):
                identity_label, identity_conf = cat_identities[idx]
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

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Zone motion: {motion_percent:.2f}%",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            zone_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Cat in zone" if cat_in_zone else ("Motion detected" if motion_active else "Idle"),
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            zone_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Dwell: {tracked_duration:.1f}s / {args.alert_seconds:.1f}s",
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            zone_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Mode: {mode} ({detector_backend}, {detector_device})",
            (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if detector is not None:
            cv2.putText(
                frame,
                f"Cat raw: {'yes' if cat_detected_in_zone else 'no'} streak={cat_seen_streak}",
                (10, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Identity: {cat_identity_in_zone} ({cat_identity_conf:.2f})",
                (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"Zone: {'polygon' if zone_polygon_edit is not None else 'rect'} overlap>={args.cat_zone_overlap:.2f}",
            (10, 275 if detector is not None else 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if args.zone_edit:
            cv2.putText(
                frame,
                "Zone edit: drag zone/points, press p to print, q to quit",
                (10, 310 if detector is not None else 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if alert_fired:
            cv2.putText(
                frame,
                "ALERT",
                (10, 345 if detector is not None else 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(args.window_name, frame)

        if clip_writer is not None:
            clip_writer.write(clip_frame)
            if event_now >= clip_end_at:
                clip_writer.release()
                if active_clip_path is not None:
                    print(
                        f"[{format_timestamp(event_now)}] Clip saved: {active_clip_path}"
                    )
                clip_writer = None
                active_clip_path = None

        key = cv2.waitKey(1) & 0xFF
        if key == ord("p") and args.zone_edit:
            print_zone_value()
        if key == ord("q"):
            run_state["reason"] = "quit_key"
            break

    if clip_writer is not None:
        clip_writer.release()
        if active_clip_path is not None:
            print(f"[{format_timestamp(time.time())}] Clip saved: {active_clip_path}")
    cap.release()
    cv2.destroyAllWindows()
    log_run_end()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
