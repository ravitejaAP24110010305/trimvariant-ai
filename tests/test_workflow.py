import numpy as np
import pytest
from PIL import Image

from app.schemas import GenerateRequest, RegionSpec
from app.services import library, report
from app.services.regions import RegionError
from app.services.workflow import NotFoundError, VariantWorkflow
from tests.conftest import FakeEngine


@pytest.fixture()
def setup(settings, db, car_png_bytes):
    image = library.register_image(settings, db, "car.png", car_png_bytes, "side")
    return image, VariantWorkflow(settings, db, FakeEngine())


def request(image_id, **kw):
    base = dict(image_id=image_id, feature="wheels", option="gloss black split 10-spoke sport alloy wheel",
                regions=[RegionSpec(type="auto_wheels")], steps=10, seed=7)
    base.update(kw)
    return GenerateRequest(**base)


def test_full_variant_run(setup, db):
    image, wf = setup
    vid = wf.create_variant(request(image["id"], mode="controlnet"))
    wf.execute(vid)
    v = db.get_variant(vid)
    assert v["status"] == "done", v["error"]
    assert v["seed"] == 7 and v["control_path"] and v["mask_path"]
    out = np.asarray(Image.open(v["output_path"]))
    orig = np.asarray(Image.open(image["prepared_path"]).convert("RGB"))
    mask = np.asarray(Image.open(v["mask_path"])) > 127
    far = ~(__import__("cv2").dilate(mask.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)
    assert (out[far] == orig[far]).all()  # background pixel-identical
    assert (np.abs(out[mask].astype(int) - orig[mask].astype(int)).mean()) > 20  # feature changed
    assert v["metrics"]["raw_outside_ssim"] > 0.99 and v["qc"]["passed"]


def test_inpaint_mode_has_no_control_image(setup, db):
    image, wf = setup
    vid = wf.create_variant(request(image["id"], mode="inpaint"))
    wf.execute(vid)
    v = db.get_variant(vid)
    assert v["status"] == "done" and not v["control_path"]


def test_compare_uses_same_seed_and_group(setup, db):
    image, wf = setup
    ids = wf.create_compare(request(image["id"], seed=None))
    rows = [db.get_variant(i) for i in ids]
    assert {r["mode"] for r in rows} == {"inpaint", "controlnet"}
    assert rows[0]["seed"] == rows[1]["seed"] and rows[0]["compare_group"] == rows[1]["compare_group"]
    for i in ids:
        wf.execute(i)
    md = report.to_markdown(db, image["id"])
    assert "Plain inpainting vs ControlNet" in md and "controlnet" in md
    csv_text = report.to_csv(db, image["id"])
    assert csv_text.count("\n") == 3  # header + 2 rows


def test_failure_is_recorded_not_raised(settings, db, car_png_bytes):
    image = library.register_image(settings, db, "car.png", car_png_bytes, "side")
    wf = VariantWorkflow(settings, db, FakeEngine(fail=True))
    vid = wf.create_variant(request(image["id"]))
    wf.execute(vid)
    v = db.get_variant(vid)
    assert v["status"] == "failed" and "engine exploded" in v["error"]


def test_invalid_requests(setup):
    image, wf = setup
    with pytest.raises(NotFoundError):
        wf.create_variant(request("missing"))
    with pytest.raises(RegionError):
        wf.create_variant(request(image["id"], regions=[RegionSpec(type="box", box=[0.5, 0.5, 0.5, 0.5])],
                                  mask_dilate_px=0))
    with pytest.raises(ValueError):
        wf.create_variant(request(image["id"], feature="custom"))  # no description
    with pytest.raises(ValueError):
        wf.create_variant(request(image["id"], feature="unknown"))
    with pytest.raises(RegionError):
        wf.create_variant(request(image["id"], regions=[RegionSpec(type="box", box=[0, 0, 1, 1])]))


def test_execute_ignores_non_queued(setup, db):
    image, wf = setup
    vid = wf.create_variant(request(image["id"]))
    wf.execute(vid)
    wf.execute(vid)  # second call is a no-op
    assert len(wf.engine.calls) == 1


def test_schema_rejects_bad_params(setup):
    image, _ = setup
    with pytest.raises(ValueError):
        request(image["id"], steps=1)
    with pytest.raises(ValueError):
        request(image["id"], control_guidance_start=0.8, control_guidance_end=0.2)
    with pytest.raises(ValueError):
        GenerateRequest(image_id="x", feature="wheels", regions=[])
