import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.schemas import RegionSpec
from app.services import regions
from app.services.image_prep import (ImageValidationError, build_control_image, canny_edges, prepare_for_model,
                                     validate_upload)


def png_bytes(size=(512, 320), color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_validate_accepts_good_image(settings):
    img = validate_upload("car.png", png_bytes(), settings)
    assert img.size == (512, 320)


@pytest.mark.parametrize("name,data,msg", [
    ("car.gif", b"x", "Unsupported"),
    ("car.png", b"", "empty"),
    ("car.png", b"not an image", "not a valid image"),
])
def test_validate_rejects_bad_files(settings, name, data, msg):
    with pytest.raises(ImageValidationError, match=msg):
        validate_upload(name, data, settings)


def test_validate_rejects_small_image(settings):
    with pytest.raises(ImageValidationError, match="too small"):
        validate_upload("tiny.png", png_bytes((100, 100)), settings)


def test_prepare_snaps_to_multiple_of_8_and_limits_size():
    out = prepare_for_model(Image.new("RGB", (2001, 1003)), 768)
    assert max(out.size) <= 768 and out.size[0] % 8 == 0 and out.size[1] % 8 == 0
    small = prepare_for_model(Image.new("RGB", (500, 301)), 768)
    assert small.size[0] % 8 == 0 and small.size[1] % 8 == 0


def test_control_image_drops_inner_edges(car):
    mask = np.zeros((car.size[1], car.size[0]), np.uint8)
    mask[200:400, 100:300] = 255
    full = np.asarray(build_control_image(car, mask, keep_inner_edges=True))
    outer = np.asarray(build_control_image(car, mask, keep_inner_edges=False))
    assert full[..., 0][mask > 0].sum() > 0
    assert outer[..., 0][mask > 0].sum() == 0
    assert outer.shape == full.shape == (car.size[1], car.size[0], 3)
    assert canny_edges(car).shape == (car.size[1], car.size[0])


def test_box_polygon_ellipse_masks(car):
    size = car.size
    box = regions.build_mask(size, [RegionSpec(type="box", box=[0.1, 0.1, 0.3, 0.3])])
    assert 0.03 < regions.mask_coverage(box) < 0.05
    poly = regions.build_mask(size, [RegionSpec(type="polygon", points=[[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]])])
    assert regions.mask_coverage(poly) > 0.05
    ell = regions.build_mask(size, [RegionSpec(type="ellipse", ellipse=[0.5, 0.5, 0.1, 0.1])])
    assert ell[size[1] // 2, size[0] // 2] == 255


def test_region_validation():
    with pytest.raises(ValueError):
        RegionSpec(type="box", box=[0.1, 0.1])
    with pytest.raises(ValueError):
        RegionSpec(type="box", box=[0.1, 0.1, 1.5, 0.3])
    with pytest.raises(ValueError):
        RegionSpec(type="polygon", points=[[0, 0], [1, 1]])


def test_mask_png_roundtrip(car):
    m = np.zeros((car.size[1], car.size[0]), np.uint8)
    m[50:100, 60:160] = 255
    buf = io.BytesIO()
    Image.fromarray(m).convert("RGB").save(buf, format="PNG")
    data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    out = regions.build_mask(car.size, [RegionSpec(type="mask_png", data=data)])
    assert (out == m).all()


def test_bad_mask_png_raises(car):
    with pytest.raises(regions.RegionError):
        regions.build_mask(car.size, [RegionSpec(type="mask_png", data="!!!notbase64!!!")])


def test_detect_wheels_on_synthetic_car(car):
    circles = regions.detect_wheels(car)
    assert len(circles) == 2
    w, h = car.size
    for cx, cy, r in circles:
        assert cy > h * 0.55 and r > w * 0.04


def test_auto_wheels_mask_is_in_lower_half(car):
    mask = regions.build_mask(car.size, [RegionSpec(type="auto_wheels")], car)
    ys, _ = np.where(mask > 0)
    assert ys.min() > car.size[1] * 0.45


def test_detect_wheels_none_found():
    with pytest.raises(regions.RegionError):
        regions.detect_wheels(Image.new("RGB", (512, 320), (120, 120, 120)))


def test_refine_and_feather():
    mask = np.zeros((64, 64), np.uint8)
    mask[28:36, 28:36] = 255
    assert regions.refine_mask(mask, 4).sum() > mask.sum()
    assert regions.refine_mask(mask, 0) is mask
    alpha = regions.feather_alpha(mask, 6)
    assert alpha.max() == 1.0 and alpha[0, 0] == 0.0 and alpha[27, 32] > 0.0
    assert (regions.feather_alpha(mask, 0) == (mask > 127)).all()
