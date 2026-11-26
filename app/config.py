"""Central configuration. Values come from environment variables or a `.env` file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency). Real env vars win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_dir: Path
    db_path: Path
    # models
    inpaint_model: str
    controlnet_model: str
    device: str  # auto | cuda | mps | cpu
    hf_token: str | None
    local_files_only: bool
    cpu_offload: bool
    keep_both_models: bool
    enable_clip: bool
    clip_model: str
    # images
    max_image_side: int
    min_image_side: int
    max_upload_mb: int
    # generation defaults
    default_steps: int
    default_guidance: float
    default_strength: float
    default_controlnet_scale: float
    canny_low: int
    canny_high: int
    mask_dilate_px: int
    mask_feather_px: int
    # quality-control thresholds (starting values - calibrate on your own images)
    qc_min_outside_ssim: float
    qc_min_edge_f1: float
    qc_min_inside_change: float
    qc_min_halo_ssim: float
    qc_min_coverage: float
    qc_max_coverage: float

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def prepared_dir(self) -> Path:
        return self.data_dir / "prepared"

    @property
    def variants_dir(self) -> Path:
        return self.data_dir / "variants"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.prepared_dir, self.variants_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(_str("TRIMVARIANT_DATA_DIR", str(BASE_DIR / "data" / "runtime")))
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir
    return Settings(
        base_dir=BASE_DIR,
        data_dir=data_dir,
        db_path=data_dir / "trimvariant.db",
        inpaint_model=_str("INPAINT_MODEL", "stable-diffusion-v1-5/stable-diffusion-inpainting"),
        controlnet_model=_str("CONTROLNET_MODEL", "lllyasviel/control_v11p_sd15_canny"),
        device=_str("DEVICE", "auto").lower(),
        hf_token=os.environ.get("HF_TOKEN") or None,
        local_files_only=_bool("LOCAL_FILES_ONLY", False),
        cpu_offload=_bool("CPU_OFFLOAD", False),
        keep_both_models=_bool("KEEP_BOTH_MODELS", True),
        enable_clip=_bool("ENABLE_CLIP_SCORE", False),
        clip_model=_str("CLIP_MODEL", "openai/clip-vit-base-patch32"),
        max_image_side=_int("MAX_IMAGE_SIDE", 768),
        min_image_side=_int("MIN_IMAGE_SIDE", 256),
        max_upload_mb=_int("MAX_UPLOAD_MB", 15),
        default_steps=_int("DEFAULT_STEPS", 30),
        default_guidance=_float("DEFAULT_GUIDANCE", 7.5),
        default_strength=_float("DEFAULT_STRENGTH", 1.0),
        default_controlnet_scale=_float("DEFAULT_CONTROLNET_SCALE", 1.0),
        canny_low=_int("CANNY_LOW", 100),
        canny_high=_int("CANNY_HIGH", 200),
        mask_dilate_px=_int("MASK_DILATE_PX", 4),
        mask_feather_px=_int("MASK_FEATHER_PX", 8),
        qc_min_outside_ssim=_float("QC_MIN_OUTSIDE_SSIM", 0.80),
        qc_min_edge_f1=_float("QC_MIN_EDGE_F1", 0.50),
        qc_min_inside_change=_float("QC_MIN_INSIDE_CHANGE", 0.04),
        qc_min_halo_ssim=_float("QC_MIN_HALO_SSIM", 0.80),
        qc_min_coverage=_float("QC_MIN_COVERAGE", 0.005),
        qc_max_coverage=_float("QC_MAX_COVERAGE", 0.45),
    )
