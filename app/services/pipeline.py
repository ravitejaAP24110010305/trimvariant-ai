"""Stable Diffusion Inpainting and ControlNet-guided Inpainting (Milestones 2 & 3).

Heavy libraries (torch / diffusers) are imported lazily so the web app, database and tests
work without them; a clear EngineUnavailable error is raised when generation is requested.
"""
from __future__ import annotations

import gc
import importlib.util
import logging
import threading
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from app.config import Settings

logger = logging.getLogger(__name__)


class EngineUnavailable(RuntimeError):
    """torch/diffusers are missing or the models could not be loaded."""


@dataclass
class GenerationParams:
    prompt: str
    negative_prompt: str
    steps: int = 30
    guidance_scale: float = 7.5
    strength: float = 1.0
    controlnet_scale: float = 1.0
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    seed: int = 0


class Engine(Protocol):
    def generate(self, mode: str, image: Image.Image, mask: Image.Image,
                 control_image: Image.Image | None, params: GenerationParams) -> Image.Image: ...

    def info(self) -> dict: ...


class DiffusersEngine:
    """Lazy-loading wrapper around diffusers inpainting pipelines.

    mode="inpaint"    -> StableDiffusionInpaintPipeline (baseline)
    mode="controlnet" -> StableDiffusionControlNetInpaintPipeline + canny ControlNet
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()  # one GPU job at a time
        self._pipes: dict[str, object] = {}
        self._controlnet = None

    # ------------------------------------------------------------------ info
    @staticmethod
    def dependencies_installed() -> bool:
        return all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers", "transformers"))

    def info(self) -> dict:
        info = {
            "dependencies_installed": self.dependencies_installed(),
            "loaded_modes": sorted(self._pipes),
            "inpaint_model": self.settings.inpaint_model,
            "controlnet_model": self.settings.controlnet_model,
            "device": "unknown",
        }
        if info["dependencies_installed"]:
            try:
                info["device"] = self._device()
            except Exception:  # pragma: no cover - defensive
                pass
        return info

    # ---------------------------------------------------------------- device
    def _device(self) -> str:
        import torch

        want = self.settings.device
        if want in ("cuda", "mps", "cpu"):
            return want
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _dtype(self, device: str):
        import torch

        return torch.float16 if device in ("cuda", "mps") else torch.float32

    # --------------------------------------------------------------- loading
    def _common_kwargs(self, dtype) -> dict:
        kwargs = {"torch_dtype": dtype, "safety_checker": None, "requires_safety_checker": False,
                  "local_files_only": self.settings.local_files_only}
        if self.settings.hf_token:
            kwargs["token"] = self.settings.hf_token
        return kwargs

    def _finalize(self, pipe, device: str):
        from diffusers import UniPCMultistepScheduler

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if device == "cuda" and self.settings.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        if device != "cuda" or self.settings.cpu_offload:
            pipe.enable_attention_slicing()
        return pipe

    def _release(self, keep: str) -> None:
        if self.settings.keep_both_models:
            return
        for mode in [m for m in self._pipes if m != keep]:
            del self._pipes[mode]
        self._controlnet = None if keep != "controlnet" else self._controlnet
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    def _load(self, mode: str):
        if mode in self._pipes:
            return self._pipes[mode]
        if not self.dependencies_installed():
            raise EngineUnavailable(
                "torch / diffusers / transformers are not installed. Run: pip install -r requirements.txt "
                "(see README for the CUDA build of PyTorch)."
            )
        try:
            from diffusers import (ControlNetModel, StableDiffusionControlNetInpaintPipeline,
                                   StableDiffusionInpaintPipeline)

            device = self._device()
            dtype = self._dtype(device)
            self._release(keep=mode)
            logger.info("Loading '%s' pipeline on %s (%s)", mode, device, dtype)
            if mode == "inpaint":
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    self.settings.inpaint_model, **self._common_kwargs(dtype))
            else:
                token = {"token": self.settings.hf_token} if self.settings.hf_token else {}
                self._controlnet = ControlNetModel.from_pretrained(
                    self.settings.controlnet_model, torch_dtype=dtype,
                    local_files_only=self.settings.local_files_only, **token)
                pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    self.settings.inpaint_model, controlnet=self._controlnet, **self._common_kwargs(dtype))
            self._pipes[mode] = self._finalize(pipe, device)
            return self._pipes[mode]
        except EngineUnavailable:
            raise
        except Exception as exc:
            raise EngineUnavailable(
                f"Could not load the '{mode}' pipeline ({type(exc).__name__}: {exc}). Check your internet "
                f"connection / HF_TOKEN / model names in .env."
            ) from exc

    # ------------------------------------------------------------ generation
    def generate(self, mode: str, image: Image.Image, mask: Image.Image,
                 control_image: Image.Image | None, params: GenerationParams) -> Image.Image:
        if mode not in ("inpaint", "controlnet"):
            raise ValueError(f"Unknown mode '{mode}'")
        if mode == "controlnet" and control_image is None:
            raise ValueError("controlnet mode needs a control image")
        with self._lock:
            import torch

            pipe = self._load(mode)
            generator = torch.Generator(device="cpu").manual_seed(int(params.seed))
            kwargs = dict(
                prompt=params.prompt, negative_prompt=params.negative_prompt,
                image=image.convert("RGB"), mask_image=mask.convert("L"),
                num_inference_steps=params.steps, guidance_scale=params.guidance_scale,
                strength=params.strength, generator=generator,
            )
            if mode == "controlnet":
                kwargs.update(
                    control_image=control_image.convert("RGB"),  # type: ignore[union-attr]
                    controlnet_conditioning_scale=params.controlnet_scale,
                    control_guidance_start=params.control_guidance_start,
                    control_guidance_end=params.control_guidance_end,
                )
            with torch.inference_mode():
                result = pipe(**kwargs).images[0]
            return result
