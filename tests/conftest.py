import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.sample_data import make_synthetic_car  # noqa: E402


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIMVARIANT_DATA_DIR", str(tmp_path / "data"))
    from app.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    s.ensure_dirs()
    yield s
    get_settings.cache_clear()


@pytest.fixture()
def car() -> Image.Image:
    return make_synthetic_car()


@pytest.fixture()
def car_png_bytes(car):
    import io

    buf = io.BytesIO()
    car.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def db(settings):
    from app.database import Database

    database = Database(settings.db_path)
    database.init()
    return database


class FakeEngine:
    """Test double: repaints the masked area a solid colour (no GPU/model needed)."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []

    def generate(self, mode, image, mask, control_image, params):
        self.calls.append((mode, params.seed))
        if self.fail:
            raise RuntimeError("engine exploded")
        import numpy as np

        arr = np.asarray(image).copy()
        arr[np.asarray(mask) > 127] = (240, 200, 20)
        return Image.fromarray(arr)

    def info(self):
        return {"dependencies_installed": False, "device": "fake"}


@pytest.fixture()
def fake_engine():
    return FakeEngine()
