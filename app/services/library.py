"""Image library: store uploads and prepared (model-ready) copies."""
from __future__ import annotations

import shutil
from pathlib import Path

from app.config import Settings
from app.database import Database, new_id
from app.services.image_prep import prepare_for_model, validate_upload

VALID_VIEWS = {"side", "front", "rear", "three_quarter"}


def register_image(settings: Settings, db: Database, filename: str, data: bytes, view: str = "side") -> dict:
    if view not in VALID_VIEWS:
        raise ValueError(f"view must be one of {sorted(VALID_VIEWS)}")
    img = validate_upload(filename, data, settings)  # raises ImageValidationError
    prepared = prepare_for_model(img, settings.max_image_side)

    settings.ensure_dirs()
    stem = new_id()
    ext = Path(filename).suffix.lower()
    original_path = settings.uploads_dir / f"{stem}{ext}"
    prepared_path = settings.prepared_dir / f"{stem}.png"
    original_path.write_bytes(data)
    prepared.save(prepared_path)
    return db.insert_image(
        filename=Path(filename).name, view=view, original_path=original_path, prepared_path=prepared_path,
        width=prepared.size[0], height=prepared.size[1], orig_width=img.size[0], orig_height=img.size[1],
    )


def delete_image(settings: Settings, db: Database, image_id: str) -> bool:
    image = db.get_image(image_id)
    if not image:
        return False
    for variant in db.list_variants(image_id):
        shutil.rmtree(settings.variants_dir / variant["id"], ignore_errors=True)
    for key in ("original_path", "prepared_path"):
        Path(image[key]).unlink(missing_ok=True)
    return db.delete_image(image_id)


def delete_variant(settings: Settings, db: Database, variant_id: str) -> bool:
    shutil.rmtree(settings.variants_dir / variant_id, ignore_errors=True)
    return db.delete_variant(variant_id)
