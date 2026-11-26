"""Vehicle image validation and preparation (Milestone 1)."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import Settings

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a usable vehicle image."""


def validate_upload(filename: str, data: bytes, settings: Settings) -> Image.Image:
    """Validate an upload and return it as an upright RGB image."""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext or 'unknown'}'. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ImageValidationError(f"File is larger than the {settings.max_upload_mb} MB limit.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageValidationError(f"The file is not a valid image: {exc}") from exc
    if min(img.size) < settings.min_image_side:
        raise ImageValidationError(
            f"Image is too small ({img.size[0]}x{img.size[1]}); the shorter side must be at least "
            f"{settings.min_image_side}px."
        )
    return img.convert("RGB")


def prepare_for_model(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest side <= max_side and snap dimensions to multiples of 8
    (a requirement of the Stable Diffusion VAE)."""
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    new_w = max(64, int(round(w * scale / 8)) * 8)
    new_h = max(64, int(round(h * scale / 8)) * 8)
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def canny_edges(img: Image.Image, low: int = 100, high: int = 200) -> np.ndarray:
    """Canny edge map (H x W, uint8 0/255) - the structural signal for ControlNet."""
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, low, high)


def build_control_image(img: Image.Image, mask: np.ndarray | None, keep_inner_edges: bool,
                        low: int = 100, high: int = 200) -> Image.Image:
    """3-channel canny control image.

    By default edges *inside* the editable region are removed so ControlNet preserves the
    surrounding geometry (body lines, wheel arches, window frames) while the feature itself
    is free to change. Set keep_inner_edges=True to also hold the original feature outline.
    """
    edges = canny_edges(img, low, high)
    if mask is not None and not keep_inner_edges:
        edges = edges.copy()
        edges[mask > 127] = 0
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8), mode="L")
