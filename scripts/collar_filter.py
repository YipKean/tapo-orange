from __future__ import annotations

import cv2
import numpy as np


def build_collar_mask(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros(crop_bgr.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    saturated = (sat >= 45) & (val >= 35)
    green_cyan = (hue >= 35) & (hue <= 100)
    blue_purple = (hue >= 101) & (hue <= 145)

    mask = (saturated & (green_cyan | blue_purple)).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return mask

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def suppress_collar_artifacts(crop_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    mask = build_collar_mask(crop_bgr)
    masked_pixels = cv2.countNonZero(mask)
    total_pixels = max(1, crop_bgr.shape[0] * crop_bgr.shape[1])
    mask_ratio = masked_pixels / total_pixels
    if masked_pixels == 0:
        return crop_bgr, 0.0

    # Replace only unusual saturated collar colors. Fur colors and black/white
    # coat evidence are intentionally left untouched for Orange/Goblin identity.
    blurred = cv2.GaussianBlur(crop_bgr, (21, 21), 0)
    filtered = crop_bgr.copy()
    filtered[mask > 0] = blurred[mask > 0]
    return filtered, mask_ratio
