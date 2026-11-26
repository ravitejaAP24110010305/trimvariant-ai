"""FastAPI application: REST API + static web UI."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import ValidationError

from app import __version__
from app.config import Settings, get_settings
from app.database import Database
from app.schemas import GenerateRequest, ReviewRequest
from app.services import library, prompts, regions, report
from app.services.image_prep import ImageValidationError
from app.services.pipeline import DiffusersEngine, Engine
from app.services.workflow import NotFoundError, VariantWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("trimvariant")

STATIC_DIR = Path(__file__).parent / "static"
FILE_KINDS = {"output": "output_path", "raw": "raw_path", "mask": "mask_path", "control": "control_path"}


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.init()
    engine = engine or DiffusersEngine(settings)
    workflow = VariantWorkflow(settings, db, engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        interrupted = db.fail_interrupted_jobs()
        if interrupted:
            logger.warning("Marked %d interrupted job(s) as failed", interrupted)
        yield
        workflow.shutdown()

    app = FastAPI(title="TrimVariant AI", version=__version__, lifespan=lifespan)
    app.state.settings, app.state.db, app.state.workflow = settings, db, workflow

    # -------------------------------------------------------------- helpers
    def image_or_404(image_id: str) -> dict:
        image = db.get_image(image_id)
        if not image:
            raise HTTPException(404, f"Image '{image_id}' not found")
        return image

    def variant_or_404(variant_id: str) -> dict:
        variant = db.get_variant(variant_id)
        if not variant:
            raise HTTPException(404, f"Variant '{variant_id}' not found")
        return variant

    def public_variant(v: dict) -> dict:
        data = {k: v[k] for k in ("id", "image_id", "feature", "option", "mode", "prompt", "negative_prompt",
                                  "params", "seed", "compare_group", "status", "error", "metrics", "qc",
                                  "review_decision", "review_note", "created_at", "finished_at", "duration_s")}
        data["files"] = {kind: f"/api/variants/{v['id']}/files/{kind}"
                         for kind, col in FILE_KINDS.items() if v.get(col)}
        return data

    def public_image(i: dict) -> dict:
        return {k: i[k] for k in ("id", "filename", "view", "width", "height", "orig_width", "orig_height",
                                  "created_at")} | {"url": f"/api/images/{i['id']}/file"}

    @app.exception_handler(regions.RegionError)
    async def _region_error(_, exc: regions.RegionError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(ImageValidationError)
    async def _image_error(_, exc: ImageValidationError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(NotFoundError)
    async def _not_found(_, exc: NotFoundError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def _value_error(_, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    # ----------------------------------------------------------------- meta
    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "engine": engine.info()}

    @app.get("/api/features")
    def features():
        return {"features": prompts.FEATURE_CATALOG, "region_presets": regions.REGION_PRESETS,
                "defaults": {"steps": settings.default_steps, "guidance_scale": settings.default_guidance,
                             "strength": settings.default_strength,
                             "controlnet_scale": settings.default_controlnet_scale,
                             "mask_dilate_px": settings.mask_dilate_px,
                             "mask_feather_px": settings.mask_feather_px}}

    # --------------------------------------------------------------- images
    @app.post("/api/images", status_code=201)
    def upload_image(file: UploadFile = File(...), view: str = Form("side")):
        data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        image = library.register_image(settings, db, file.filename or "upload", data, view)
        return public_image(image)

    @app.get("/api/images")
    def list_images():
        return [public_image(i) for i in db.list_images()]

    @app.get("/api/images/{image_id}/file")
    def image_file(image_id: str):
        return FileResponse(image_or_404(image_id)["prepared_path"], media_type="image/png")

    @app.delete("/api/images/{image_id}", status_code=204)
    def delete_image(image_id: str):
        image_or_404(image_id)
        library.delete_image(settings, db, image_id)
        return Response(status_code=204)

    @app.post("/api/images/{image_id}/detect-wheels")
    def detect_wheels(image_id: str, scale: float = Query(0.85, ge=0.3, le=1.3)):
        image = image_or_404(image_id)
        w, h = image["width"], image["height"]
        circles = regions.detect_wheels(Image.open(image["prepared_path"]))
        return {"count": len(circles), "scale": scale,
                "circles": [{"cx": cx / w, "cy": cy / h, "r": r / w} for cx, cy, r in circles]}

    # ------------------------------------------------------------- variants
    def _validated(payload: dict) -> GenerateRequest:
        try:
            return GenerateRequest(**payload)
        except ValidationError as exc:
            raise HTTPException(422, detail=[{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()])

    @app.post("/api/variants", status_code=202)
    def create_variant(payload: dict):
        req = _validated(payload)
        return {"id": workflow.submit(req)}

    @app.post("/api/compare", status_code=202)
    def compare(payload: dict):
        req = _validated(payload)
        return {"ids": workflow.submit_compare(req)}

    @app.get("/api/variants")
    def list_variants(image_id: str | None = None):
        return [public_variant(v) for v in db.list_variants(image_id)]

    @app.get("/api/variants/{variant_id}")
    def get_variant(variant_id: str):
        return public_variant(variant_or_404(variant_id))

    @app.get("/api/variants/{variant_id}/files/{kind}")
    def variant_file(variant_id: str, kind: str):
        variant = variant_or_404(variant_id)
        if kind not in FILE_KINDS:
            raise HTTPException(404, "Unknown file kind")
        path = variant.get(FILE_KINDS[kind])
        if not path or not Path(path).exists():
            raise HTTPException(404, "File not available")
        return FileResponse(path, media_type="image/png",
                            filename=f"{variant_id}_{kind}.png" if kind == "output" else None)

    @app.post("/api/variants/{variant_id}/review")
    def review_variant(variant_id: str, body: ReviewRequest):
        variant_or_404(variant_id)
        db.update_variant(variant_id, review_decision=body.decision, review_note=body.note)
        return public_variant(variant_or_404(variant_id))

    @app.delete("/api/variants/{variant_id}", status_code=204)
    def delete_variant(variant_id: str):
        variant_or_404(variant_id)
        library.delete_variant(settings, db, variant_id)
        return Response(status_code=204)

    # --------------------------------------------------------------- report
    @app.get("/api/report.md")
    def report_md(image_id: str | None = None):
        return PlainTextResponse(report.to_markdown(db, image_id), media_type="text/markdown")

    @app.get("/api/report.csv")
    def report_csv(image_id: str | None = None):
        return PlainTextResponse(report.to_csv(db, image_id), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=trimvariant_variants.csv"})

    # ------------------------------------------------------------------- UI
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
