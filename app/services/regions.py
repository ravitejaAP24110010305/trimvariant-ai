"""Editable-region identification and mask building (Milestone 1)."""
from __future__ import annotations

import base64
import binascii
import io
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.schemas import RegionSpec


class RegionError(ValueError):
    """Raised when a region cannot be turned into a usable mask."""


# Approximate starting regions for the UI. They are only hints - refine with the brush.
REGION_PRESETS = [
    {"id": "auto_wheels", "label": "Wheels (auto-detect, side view)", "view": "side",
     "region": {"type": "auto_wheels"}},
    {"id": "front_grille", "label": "Front grille (approx.)", "view": "front",
     "region": {"type": "box", "box": [0.30, 0.50, 0.70, 0.68]}},
    {"id": "front_lower_bumper", "label": "Front lower bumper (approx.)", "view": "front",
     "region": {"type": "box", "box": [0.20, 0.68, 0.80, 0.82]}},
    {"id": "side_roof", "label": "Roof area (approx.)", "view": "side",
     "region": {"type": "box", "box": [0.28, 0.12, 0.72, 0.30]}},
    {"id": "side_door", "label": "Door / body panel (approx.)", "view": "side",
     "region": {"type": "box", "box": [0.30, 0.38, 0.68, 0.62]}},
    {"id": "rear_spoiler", "label": "Rear spoiler area (approx.)", "view": "rear",
     "region": {"type": "box", "box": [0.25, 0.22, 0.75, 0.34]}},
]


def detect_wheels(image: Image.Image, max_wheels: int = 2) -> list[tuple[float, float, float]]:
    """Find wheel circles (cx, cy, r) in pixels on a side-view vehicle using a Hough transform.

    Works best on clean side/three-quarter-side shots. For other angles draw the mask manually.
    """
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    gray = cv2.medianBlur(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w * 0.2,
        param1=120, param2=35, minRadius=int(w * 0.04), maxRadius=int(w * 0.16),
    )
    if circles is None:
        raise RegionError("No wheels detected. Use the brush or a box to mark the wheel areas.")
    # Hough returns candidates strongest-first. Wheels sit in the lower part of the picture.
    candidates = [c for c in circles[0].tolist() if c[1] > h * 0.45]
    if not candidates:
        raise RegionError("No wheel-like circles found in the lower half of the image.")
    best = candidates[0]
    chosen = [best]
    for c in candidates[1:]:
        if len(chosen) >= max_wheels:
            break
        similar_row = abs(c[1] - best[1]) < 0.08 * h
        similar_size = abs(c[2] - best[2]) < 0.45 * best[2]
        far_apart = abs(c[0] - best[0]) > 1.2 * best[2]
        if similar_row and similar_size and far_apart:
            chosen.append(c)
    return [(float(cx), float(cy), float(r)) for cx, cy, r in chosen]


def decode_mask_png(data: str, size: tuple[int, int]) -> np.ndarray:
    """Decode a base64/data-URL PNG (drawn in the UI) into a binary 0/255 mask of `size`."""
    payload = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        raw = base64.b64decode(payload, validate=False)
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError) as exc:
        raise RegionError(f"Could not decode the drawn mask: {exc}") from exc
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    arr = np.asarray(img)
    lum = arr[..., :3].mean(axis=-1)
    visible = (arr[..., 3] > 127) & (lum > 127)
    return (visible.astype(np.uint8)) * 255


def build_mask(size: tuple[int, int], regions: Iterable[RegionSpec], image: Image.Image | None = None) -> np.ndarray:
    """Union of all regions -> uint8 mask (H x W) with 255 = editable."""
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    count = 0
    for r in regions:
        count += 1
        if r.type == "box":
            x0, y0, x1, y1 = r.box  # type: ignore[misc]
            p0 = (int(round(min(x0, x1) * (w - 1))), int(round(min(y0, y1) * (h - 1))))
            p1 = (int(round(max(x0, x1) * (w - 1))), int(round(max(y0, y1) * (h - 1))))
            cv2.rectangle(mask, p0, p1, 255, thickness=-1)
        elif r.type == "polygon":
            pts = np.array([[int(round(x * (w - 1))), int(round(y * (h - 1)))] for x, y in r.points],  # type: ignore[union-attr]
                           dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        elif r.type == "ellipse":
            cx, cy, rx, ry = r.ellipse  # type: ignore[misc]
            cv2.ellipse(mask, (int(cx * (w - 1)), int(cy * (h - 1))),
                        (max(1, int(rx * w)), max(1, int(ry * h))), 0, 0, 360, 255, thickness=-1)
        elif r.type == "auto_wheels":
            if image is None:
                raise RegionError("auto_wheels needs the source image.")
            for cx, cy, rad in detect_wheels(image):
                cv2.circle(mask, (int(cx), int(cy)), max(1, int(rad * r.wheel_scale)), 255, thickness=-1)
        elif r.type == "mask_png":
            mask = np.maximum(mask, decode_mask_png(r.data or "", (w, h)))
    if count == 0:
        raise RegionError("At least one region is required.")
    return mask


def refine_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Grow the mask slightly so the edit fully covers the original feature's edges."""
    if dilate_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    return cv2.dilate(mask, k)


def mask_coverage(mask: np.ndarray) -> float:
    return float((mask > 127).mean())


def feather_alpha(mask: np.ndarray, feather_px: int) -> np.ndarray:
    """Soft blend weights (float32 0..1). Fully 1 inside the mask, fading out beyond its edge."""
    binary = (mask > 127).astype(np.float32)
    if feather_px <= 0:
        return binary
    k = feather_px * 2 + 1
    blurred = cv2.GaussianBlur(binary, (k, k), sigmaX=max(0.5, feather_px / 2))
    return np.maximum(blurred, binary)
