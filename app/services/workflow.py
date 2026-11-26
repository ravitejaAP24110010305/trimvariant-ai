"""End-to-end variant workflow: masks -> generate -> composite -> evaluate -> store."""
from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings
from app.database import Database, new_id, utcnow
from app.schemas import GenerateRequest
from app.services import evaluation, prompts, regions
from app.services.image_prep import build_control_image, mask_to_image
from app.services.pipeline import Engine, GenerationParams

logger = logging.getLogger(__name__)


class NotFoundError(LookupError):
    pass


class VariantWorkflow:
    def __init__(self, settings: Settings, db: Database, engine: Engine):
        self.settings = settings
        self.db = db
        self.engine = engine
        # Single worker: diffusion models are GPU/RAM bound, so jobs run one after another.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="variant-worker")

    # ------------------------------------------------------------- creation
    def create_variant(self, req: GenerateRequest, mode: str | None = None, seed: int | None = None,
                       compare_group: str | None = None) -> str:
        """Validate the request, build + store the mask, and enqueue (status=queued) a DB row."""
        s = self.settings
        image = self.db.get_image(req.image_id)
        if not image:
            raise NotFoundError(f"Image '{req.image_id}' not found")
        mode = mode or req.mode
        prompt = prompts.build_prompt(req.feature, req.option, req.custom_prompt)  # ValueError if invalid
        negative = prompts.build_negative(req.negative_prompt)

        src = Image.open(image["prepared_path"]).convert("RGB")
        dilate = s.mask_dilate_px if req.mask_dilate_px is None else req.mask_dilate_px
        feather = s.mask_feather_px if req.mask_feather_px is None else req.mask_feather_px
        mask = regions.refine_mask(regions.build_mask(src.size, req.regions, src), dilate)
        coverage = regions.mask_coverage(mask)
        if coverage < 0.001:
            raise regions.RegionError("The mask is empty or far too small - mark the area you want to change.")
        if coverage > 0.9:
            raise regions.RegionError("The mask covers almost the entire image; select just the feature.")

        params = {
            "steps": req.steps or s.default_steps,
            "guidance_scale": req.guidance_scale or s.default_guidance,
            "strength": req.strength or s.default_strength,
            "controlnet_scale": s.default_controlnet_scale if req.controlnet_scale is None else req.controlnet_scale,
            "control_guidance_start": req.control_guidance_start,
            "control_guidance_end": req.control_guidance_end,
            "keep_inner_edges": req.keep_inner_edges,
            "mask_dilate_px": dilate,
            "mask_feather_px": feather,
            "mask_coverage": round(coverage, 4),
        }
        if seed is None:
            seed = req.seed
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        variant_id = self.db.insert_variant(
            image["id"], req.feature, req.option, mode, prompt, negative, params, int(seed), compare_group)

        out_dir = s.variants_dir / variant_id
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / "mask.png"
        mask_to_image(mask).save(mask_path)
        fields = {"mask_path": str(mask_path)}
        if mode == "controlnet":
            control = build_control_image(src, mask, req.keep_inner_edges, s.canny_low, s.canny_high)
            control_path = out_dir / "control.png"
            control.save(control_path)
            fields["control_path"] = str(control_path)
        self.db.update_variant(variant_id, **fields)
        return variant_id

    def submit(self, req: GenerateRequest) -> str:
        variant_id = self.create_variant(req)
        self._executor.submit(self.execute, variant_id)
        return variant_id

    def create_compare(self, req: GenerateRequest) -> list[str]:
        """Same image, mask, prompt and seed through both pipelines for a fair comparison."""
        group = new_id()
        seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
        return [self.create_variant(req, mode=m, seed=seed, compare_group=group) for m in ("inpaint", "controlnet")]

    def submit_compare(self, req: GenerateRequest) -> list[str]:
        ids = self.create_compare(req)
        for variant_id in ids:
            self._executor.submit(self.execute, variant_id)
        return ids

    # ------------------------------------------------------------ execution
    def execute(self, variant_id: str) -> None:
        """Run one variant to completion. Never raises: failures are stored on the row."""
        row = self.db.get_variant(variant_id)
        if not row or row["status"] != "queued":
            return
        started = time.time()
        self.db.update_variant(variant_id, status="running", started_at=utcnow())
        try:
            s = self.settings
            image = self.db.get_image(row["image_id"])
            if not image:
                raise NotFoundError("Source image was deleted")
            original = Image.open(image["prepared_path"]).convert("RGB")
            mask = np.asarray(Image.open(row["mask_path"]).convert("L"))
            control = Image.open(row["control_path"]).convert("RGB") if row["mode"] == "controlnet" else None
            p = row["params"]
            params = GenerationParams(
                prompt=row["prompt"], negative_prompt=row["negative_prompt"], steps=p["steps"],
                guidance_scale=p["guidance_scale"], strength=p["strength"],
                controlnet_scale=p["controlnet_scale"], control_guidance_start=p["control_guidance_start"],
                control_guidance_end=p["control_guidance_end"], seed=row["seed"],
            )
            raw = self.engine.generate(row["mode"], original, mask_to_image(mask), control, params).convert("RGB")
            if raw.size != original.size:
                raw = raw.resize(original.size, Image.LANCZOS)

            # Paste the generated feature back so everything outside the mask stays pixel-identical.
            alpha = regions.feather_alpha(mask, p["mask_feather_px"])[..., None]
            blended = np.asarray(original, np.float32) * (1 - alpha) + np.asarray(raw, np.float32) * alpha
            final = Image.fromarray(np.clip(blended + 0.5, 0, 255).astype(np.uint8))

            out_dir = Path(row["mask_path"]).parent
            raw.save(out_dir / "raw.png")
            final.save(out_dir / "output.png")

            metrics = evaluation.evaluate(original, raw, final, mask, s, prompt=row["prompt"],
                                          feather_px=p["mask_feather_px"])
            qc = evaluation.run_qc(metrics, s)
            self.db.update_variant(
                variant_id, status="done", output_path=str(out_dir / "output.png"),
                raw_path=str(out_dir / "raw.png"), metrics=metrics, qc=qc,
                finished_at=utcnow(), duration_s=round(time.time() - started, 2),
            )
        except Exception as exc:
            logger.exception("Variant %s failed", variant_id)
            self.db.update_variant(variant_id, status="failed", error=f"{type(exc).__name__}: {exc}",
                                   finished_at=utcnow(), duration_s=round(time.time() - started, 2))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
