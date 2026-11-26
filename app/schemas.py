"""Request/response models (pydantic v2)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

RegionType = Literal["box", "polygon", "ellipse", "auto_wheels", "mask_png"]


def _in_unit(values: list[float], what: str) -> None:
    if any(v < 0.0 or v > 1.0 for v in values):
        raise ValueError(f"{what} must use normalized coordinates in the range 0..1")


class RegionSpec(BaseModel):
    """An editable region, in normalized (0..1) image coordinates.

    - box:         box=[x0, y0, x1, y1]
    - polygon:     points=[[x, y], ...] (>= 3 points)
    - ellipse:     ellipse=[cx, cy, rx, ry]
    - auto_wheels: detect wheels automatically (side-view images)
    - mask_png:    data = PNG (data-URL or base64) drawn in the UI; white = edit
    """

    type: RegionType
    box: list[float] | None = None
    points: list[list[float]] | None = None
    ellipse: list[float] | None = None
    wheel_scale: float = Field(0.85, ge=0.3, le=1.3)
    data: str | None = Field(None, max_length=30_000_000)

    @model_validator(mode="after")
    def _check(self) -> "RegionSpec":
        if self.type == "box":
            if not self.box or len(self.box) != 4:
                raise ValueError("box region needs box=[x0,y0,x1,y1]")
            _in_unit(self.box, "box")
        elif self.type == "polygon":
            if not self.points or len(self.points) < 3 or any(len(p) != 2 for p in self.points):
                raise ValueError("polygon region needs at least 3 [x,y] points")
            _in_unit([c for p in self.points for c in p], "polygon")
        elif self.type == "ellipse":
            if not self.ellipse or len(self.ellipse) != 4:
                raise ValueError("ellipse region needs ellipse=[cx,cy,rx,ry]")
            _in_unit(self.ellipse, "ellipse")
        elif self.type == "mask_png":
            if not self.data:
                raise ValueError("mask_png region needs data")
        return self


class GenerateRequest(BaseModel):
    image_id: str = Field(min_length=1, max_length=64)
    feature: str = Field(min_length=1, max_length=40)
    option: str | None = Field(None, max_length=120)
    custom_prompt: str | None = Field(None, max_length=500)
    negative_prompt: str | None = Field(None, max_length=500)
    mode: Literal["inpaint", "controlnet"] = "controlnet"
    regions: list[RegionSpec] = Field(min_length=1, max_length=20)
    steps: int | None = Field(None, ge=5, le=100)
    guidance_scale: float | None = Field(None, ge=1.0, le=20.0)
    strength: float | None = Field(None, ge=0.1, le=1.0)
    controlnet_scale: float | None = Field(None, ge=0.0, le=2.0)
    control_guidance_start: float = Field(0.0, ge=0.0, le=1.0)
    control_guidance_end: float = Field(1.0, ge=0.0, le=1.0)
    keep_inner_edges: bool = False
    mask_dilate_px: int | None = Field(None, ge=0, le=64)
    mask_feather_px: int | None = Field(None, ge=0, le=64)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def _check(self) -> "GenerateRequest":
        if self.control_guidance_end < self.control_guidance_start:
            raise ValueError("control_guidance_end must be >= control_guidance_start")
        return self


class ReviewRequest(BaseModel):
    decision: Literal["approved", "rejected", "pending"]
    note: str | None = Field(None, max_length=1000)
