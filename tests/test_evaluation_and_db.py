import numpy as np
import pytest
from PIL import Image

from app.services import evaluation, prompts


def _mask(shape, box):
    m = np.zeros(shape, np.uint8)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = 255
    return m


def test_identical_outside_scores_high(settings, car):
    mask = _mask((car.size[1], car.size[0]), (250, 380, 120, 280))
    arr = np.asarray(car).copy()
    arr[mask > 0] = (0, 255, 0)
    edited = Image.fromarray(arr)
    m = evaluation.evaluate(car, edited, edited, mask, settings)
    assert m["raw_outside_ssim"] > 0.99 and m["raw_edge_f1"] > 0.95 and m["inside_change"] > 0.1
    assert m["raw_outside_mae"] < 0.01


def test_background_damage_lowers_scores(settings, car):
    mask = _mask((car.size[1], car.size[0]), (250, 380, 120, 280))
    rng = np.random.default_rng(0)
    noisy = np.clip(np.asarray(car).astype(int) + rng.integers(-60, 60, (car.size[1], car.size[0], 3)), 0, 255)
    noisy_img = Image.fromarray(noisy.astype(np.uint8))
    good = evaluation.evaluate(car, car, car, mask, settings)
    bad = evaluation.evaluate(car, noisy_img, noisy_img, mask, settings)
    assert bad["raw_outside_ssim"] < good["raw_outside_ssim"] - 0.3
    assert bad["raw_edge_f1"] < good["raw_edge_f1"]


def test_qc_pass_and_fail(settings, car):
    mask = _mask((car.size[1], car.size[0]), (250, 380, 120, 280))
    arr = np.asarray(car).copy()
    arr[mask > 0] = (0, 255, 0)
    edited = Image.fromarray(arr)
    ok = evaluation.run_qc(evaluation.evaluate(car, edited, edited, mask, settings), settings)
    assert ok["passed"], ok["failed"]
    unchanged = evaluation.run_qc(evaluation.evaluate(car, car, car, mask, settings), settings)
    assert not unchanged["passed"] and "feature_changed" in unchanged["failed"]


def test_prompts():
    assert "gloss black" in prompts.build_prompt("wheels", "gloss black alloy", None)
    assert prompts.build_prompt("grille", None, "extra").count("extra") == 1
    assert "matte black splitter" in prompts.build_prompt("custom", None, "matte black splitter")
    with pytest.raises(ValueError):
        prompts.build_prompt("custom", None, "")
    with pytest.raises(ValueError):
        prompts.get_feature("nope")


def test_database_roundtrip(db, settings, car_png_bytes):
    from app.services import library

    img = library.register_image(settings, db, "car.png", car_png_bytes, "side")
    assert db.get_image(img["id"])["width"] % 8 == 0
    vid = db.insert_variant(img["id"], "wheels", "x", "inpaint", "p", "n", {"steps": 5}, 1)
    db.update_variant(vid, status="done", metrics={"a": 1}, qc={"passed": True})
    v = db.get_variant(vid)
    assert v["params"] == {"steps": 5} and v["metrics"] == {"a": 1} and v["review_decision"] == "pending"
    with pytest.raises(ValueError):
        db.update_variant(vid, bogus=1)
    db.insert_variant(img["id"], "wheels", None, "inpaint", "p", "n", {}, 2)
    assert db.fail_interrupted_jobs() == 1
    assert library.delete_image(settings, db, img["id"])
    assert db.list_variants() == []  # cascade


def test_register_rejects_invalid_view(db, settings, car_png_bytes):
    from app.services import library

    with pytest.raises(ValueError):
        library.register_image(settings, db, "car.png", car_png_bytes, "top")
