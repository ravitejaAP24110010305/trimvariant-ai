"""Variant analysis and quality-control checkpoints (Milestone 4).

Two views of every result are scored:
  * RAW   - what the diffusion model returned (shows how well the model itself preserved the scene;
            this is the fair way to compare plain inpainting against ControlNet guidance)
  * FINAL - raw output pasted back through a feathered mask (what would ship)
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from app.config import Settings
from app.services.image_prep import canny_edges

logger = logging.getLogger(__name__)

METRIC_LABELS = {
    "coverage": "Edited area (fraction of image)",
    "raw_outside_ssim": "Background SSIM (raw model output)",
    "raw_outside_mae": "Background mean abs. error (raw)",
    "raw_edge_f1": "Structure edge F1 (raw)",
    "final_background_ssim": "Background SSIM (final)",
    "inside_change": "Feature change magnitude",
    "halo_ssim": "Halo SSIM (raw, ring just outside the edit)",
    "clip_score": "CLIP prompt similarity",
}


def _ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    win = min(7, min(gray_a.shape) // 2 * 2 - 1)
    _, smap = structural_similarity(gray_a, gray_b, data_range=255, full=True, win_size=max(3, win))
    return smap


def _masked_mean(values: np.ndarray, region: np.ndarray) -> float | None:
    if not region.any():
        return None
    return float(values[region].mean())


def _edge_f1(orig: Image.Image, other: Image.Image, region: np.ndarray, low: int, high: int) -> float | None:
    """F1 of canny edges inside `region`, tolerant to 1px shifts (VAE re-encoding jitters edges)."""
    if not region.any():
        return None
    e1 = (canny_edges(orig, low, high) > 0) & region
    e2 = (canny_edges(other, low, high) > 0) & region
    if not e1.any() and not e2.any():
        return 1.0
    if not e1.any() or not e2.any():
        return 0.0
    k = np.ones((3, 3), np.uint8)
    d1 = cv2.dilate(e1.astype(np.uint8), k) > 0
    d2 = cv2.dilate(e2.astype(np.uint8), k) > 0
    precision = float((e2 & d1).sum() / e2.sum())
    recall = float((e1 & d2).sum() / e1.sum())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


_clip_cache: dict = {}


def clip_score(image: Image.Image, prompt: str, model_name: str) -> float | None:
    """Optional cosine similarity between the edited crop and the prompt (needs transformers + torch)."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if "model" not in _clip_cache:
            _clip_cache["model"] = CLIPModel.from_pretrained(model_name).eval()
            _clip_cache["proc"] = CLIPProcessor.from_pretrained(model_name)
        model, proc = _clip_cache["model"], _clip_cache["proc"]
        inputs = proc(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((img_emb * txt_emb).sum())
    except Exception as exc:  # optional metric: never fail the job
        logger.warning("CLIP score unavailable: %s", exc)
        return None


def evaluate(original: Image.Image, raw: Image.Image, final: Image.Image, mask: np.ndarray,
             settings: Settings, prompt: str | None = None, feather_px: int = 8) -> dict:
    """Compute metrics for one generated variant. `mask` is uint8, 255 = edited region."""
    o = np.asarray(original.convert("RGB"))
    r = np.asarray(raw.convert("RGB").resize(original.size)) if raw.size != original.size else np.asarray(raw.convert("RGB"))
    f = np.asarray(final.convert("RGB"))
    inside = mask > 127
    outside = ~inside
    def grow(px: int) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        return cv2.dilate(mask, kernel) > 127

    # Ring 5-20px outside the edit, beyond the SSIM window, where the model should have changed nothing.
    ring = grow(20) & ~grow(5)
    far_outside = ~grow(2 * feather_px)

    smap_raw = _ssim_map(o, r)
    smap_final = _ssim_map(o, f)
    abs_raw = np.abs(o.astype(np.float32) - r.astype(np.float32)).mean(axis=-1) / 255.0
    abs_final = np.abs(o.astype(np.float32) - f.astype(np.float32)).mean(axis=-1) / 255.0

    metrics = {
        "coverage": float(inside.mean()),
        "raw_outside_ssim": _masked_mean(smap_raw, outside),
        "raw_outside_mae": _masked_mean(abs_raw, outside),
        "raw_edge_f1": _edge_f1(original, raw.resize(original.size) if raw.size != original.size else raw,
                                outside, settings.canny_low, settings.canny_high),
        "final_background_ssim": _masked_mean(smap_final, far_outside),
        "inside_change": _masked_mean(abs_final, inside),
        "halo_ssim": _masked_mean(smap_raw, ring),
        "clip_score": None,
    }
    if settings.enable_clip and prompt and inside.any():
        ys, xs = np.where(inside)
        pad = 16
        box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
               min(final.size[0], xs.max() + pad), min(final.size[1], ys.max() + pad))
        metrics["clip_score"] = clip_score(final.crop(box), prompt, settings.clip_model)
    return {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in metrics.items()}


def run_qc(metrics: dict, settings: Settings) -> dict:
    """Automatic QC checkpoints. A human reviewer still approves/rejects in the UI."""
    checks = []

    def add(name: str, label: str, value, op: str, threshold: float, help_text: str):
        if value is None:
            ok = False
        elif op == ">=":
            ok = value >= threshold
        else:
            ok = value <= threshold
        checks.append({"name": name, "label": label, "value": value, "op": op,
                       "threshold": threshold, "passed": bool(ok), "help": help_text})

    add("coverage_min", "Edit region is large enough", metrics.get("coverage"), ">=",
        settings.qc_min_coverage, "Mask covers too little of the image to be a meaningful feature edit.")
    add("coverage_max", "Edit region is not excessive", metrics.get("coverage"), "<=",
        settings.qc_max_coverage, "Mask covers too much; the vehicle/scene may be re-generated.")
    add("feature_changed", "Feature visibly changed", metrics.get("inside_change"), ">=",
        settings.qc_min_inside_change, "The edited area barely differs from the original - try a stronger prompt.")
    add("background_preserved", "Background preserved (raw output)", metrics.get("raw_outside_ssim"), ">=",
        settings.qc_min_outside_ssim, "The model altered regions outside the mask.")
    add("structure_preserved", "Vehicle structure preserved (edge F1)", metrics.get("raw_edge_f1"), ">=",
        settings.qc_min_edge_f1, "Body lines/geometry drifted - increase ControlNet scale or shrink the mask.")
    add("halo_clean", "No colour bleed / halo around the edit", metrics.get("halo_ssim"), ">=",
        settings.qc_min_halo_ssim, "The model altered pixels just outside the mask - tighten the mask or raise ControlNet scale.")
    failed = [c for c in checks if not c["passed"]]
    return {"passed": not failed, "failed": [c["name"] for c in failed], "checks": checks}


def summarize_by_mode(variants: list[dict]) -> dict[str, dict]:
    """Mean metrics per generation mode - the inpaint vs ControlNet comparison."""
    keys = ["raw_outside_ssim", "raw_outside_mae", "raw_edge_f1", "inside_change", "halo_ssim"]
    out: dict[str, dict] = {}
    for mode in ("inpaint", "controlnet"):
        rows = [v for v in variants if v["mode"] == mode and v["status"] == "done" and v.get("metrics")]
        summary: dict = {"count": len(rows)}
        for key in keys:
            vals = [v["metrics"][key] for v in rows if v["metrics"].get(key) is not None]
            summary[key] = round(sum(vals) / len(vals), 4) if vals else None
        summary["qc_pass_rate"] = (
            round(sum(1 for v in rows if v.get("qc", {}).get("passed")) / len(rows), 3) if rows else None
        )
        out[mode] = summary
    return out
