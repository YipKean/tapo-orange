from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PetDetection:
    box: tuple[int, int, int, int]
    score: float
    class_id: int


def parse_class_ids(raw_text: str) -> list[int]:
    tokens = [token.strip() for token in raw_text.replace(";", ",").split(",")]
    class_ids: list[int] = []
    for token in tokens:
        if not token:
            continue
        class_id = int(token)
        if class_id < 0:
            raise ValueError("Class ids must be 0 or greater.")
        class_ids.append(class_id)
    return class_ids


def box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = float(iw * ih)
    first_area = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    second_area = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return intersection / max(first_area + second_area - intersection, 1.0)


def merge_detections(
    primary: list[PetDetection],
    fallback: list[PetDetection],
    *,
    iou_threshold: float = 0.55,
) -> list[PetDetection]:
    merged = list(primary)
    for candidate in fallback:
        duplicate = any(
            box_iou(candidate.box, existing.box) >= iou_threshold
            for existing in merged
        )
        if not duplicate:
            merged.append(candidate)
    return merged


def detections_to_legacy(
    detections: list[PetDetection],
) -> list[tuple[tuple[int, int, int, int], float]]:
    return [(detection.box, detection.score) for detection in detections]


def detect_objects(
    detector: object | None,
    backend: str,
    device: str,
    frame: np.ndarray,
    confidence_threshold: float,
    class_ids: list[int],
    image_size: int,
) -> list[PetDetection]:
    if detector is None or not class_ids:
        return []

    if backend == "ultralytics":
        results = detector.predict(
            source=frame,
            conf=confidence_threshold,
            classes=class_ids,
            device=device,
            verbose=False,
            imgsz=image_size,
        )
        detections: list[PetDetection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        for box, score, class_id in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = [int(v) for v in box]
            detections.append(
                PetDetection(
                    box=(x1, y1, x2, y2),
                    score=float(score),
                    class_id=int(class_id),
                )
            )
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
    if predictions.shape[1] <= 4 + max(class_ids):
        return []

    frame_height, frame_width = frame.shape[:2]
    scale_x = frame_width / input_size
    scale_y = frame_height / input_size

    boxes: list[list[int]] = []
    scores: list[float] = []
    chosen_classes: list[int] = []
    allowed_class_ids = set(class_ids)

    for row in predictions:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if class_id not in allowed_class_ids or score < confidence_threshold:
            continue

        cx, cy, width, height = row[:4]
        x1 = int((cx - width / 2) * scale_x)
        y1 = int((cy - height / 2) * scale_y)
        w = int(width * scale_x)
        h = int(height * scale_y)
        boxes.append([x1, y1, w, h])
        scores.append(score)
        chosen_classes.append(class_id)

    if not boxes:
        return []

    indexes = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, 0.45)
    if len(indexes) == 0:
        return []

    detections: list[PetDetection] = []
    for idx in np.array(indexes).flatten():
        box_index = int(idx)
        x, y, w, h = boxes[box_index]
        detections.append(
            PetDetection(
                box=(x, y, x + w, y + h),
                score=scores[box_index],
                class_id=chosen_classes[box_index],
            )
        )
    return detections


def detect_collar_aware_pets(
    detector: object | None,
    backend: str,
    device: str,
    frame: np.ndarray,
    cat_confidence: float,
    cat_class_id: int,
    image_size: int,
    *,
    collar_fallback_enabled: bool,
    collar_fallback_class_ids: list[int],
    collar_fallback_confidence: float,
) -> list[tuple[tuple[int, int, int, int], float]]:
    primary = detect_objects(
        detector,
        backend,
        device,
        frame,
        cat_confidence,
        [cat_class_id],
        image_size,
    )
    if not collar_fallback_enabled:
        return detections_to_legacy(primary)

    fallback_ids = [
        class_id for class_id in collar_fallback_class_ids if class_id != cat_class_id
    ]
    fallback = detect_objects(
        detector,
        backend,
        device,
        frame,
        collar_fallback_confidence,
        fallback_ids,
        image_size,
    )
    return detections_to_legacy(merge_detections(primary, fallback))
