"""API tests - skipped automatically when fastapi/httpx are not installed."""
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from tests.conftest import FakeEngine  # noqa: E402


@pytest.fixture()
def client(settings):
    app = create_app(settings, FakeEngine())
    with TestClient(app) as c:
        yield c


def upload(client, data):
    return client.post("/api/images", files={"file": ("car.png", data, "image/png")}, data={"view": "side"})


def wait_done(client, vid, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        v = client.get(f"/api/variants/{vid}").json()
        if v["status"] in ("done", "failed"):
            return v
        time.sleep(0.1)
    raise AssertionError("timeout")


def test_health_and_features(client):
    assert client.get("/api/health").json()["status"] == "ok"
    feats = client.get("/api/features").json()
    assert any(f["id"] == "wheels" for f in feats["features"])
    assert client.get("/").status_code == 200


def test_upload_validation(client):
    assert upload(client, b"junk").status_code == 400


def test_end_to_end(client, car_png_bytes):
    img = upload(client, car_png_bytes)
    assert img.status_code == 201
    image_id = img.json()["id"]
    assert client.get(f"/api/images/{image_id}/file").status_code == 200
    wheels = client.post(f"/api/images/{image_id}/detect-wheels").json()
    assert wheels["count"] >= 1

    body = {"image_id": image_id, "feature": "wheels", "mode": "controlnet", "steps": 10, "seed": 3,
            "regions": [{"type": "auto_wheels"}]}
    resp = client.post("/api/variants", json=body)
    assert resp.status_code == 202, resp.text
    v = wait_done(client, resp.json()["id"])
    assert v["status"] == "done" and v["qc"]["passed"]
    assert client.get(v["files"]["output"]).status_code == 200
    assert client.get(f"/api/variants/{v['id']}/files/bogus").status_code == 404

    rv = client.post(f"/api/variants/{v['id']}/review", json={"decision": "approved", "note": "ok"})
    assert rv.json()["review_decision"] == "approved"
    assert client.post(f"/api/variants/{v['id']}/review", json={"decision": "maybe"}).status_code == 422

    cmp = client.post("/api/compare", json=body).json()
    assert len(cmp["ids"]) == 2
    for i in cmp["ids"]:
        wait_done(client, i)
    assert "controlnet" in client.get("/api/report.md").text
    assert client.get("/api/report.csv").text.startswith("variant_id")

    assert client.delete(f"/api/variants/{v['id']}").status_code == 204
    assert client.delete(f"/api/images/{image_id}").status_code == 204
    assert client.get(f"/api/images/{image_id}/file").status_code == 404


def test_error_responses(client, car_png_bytes):
    image_id = upload(client, car_png_bytes).json()["id"]
    bad = {"image_id": image_id, "feature": "wheels", "regions": [{"type": "box", "box": [0.1, 0.1]}]}
    assert client.post("/api/variants", json=bad).status_code == 422
    assert client.post("/api/variants", json={"image_id": "nope", "feature": "wheels",
                                              "regions": [{"type": "auto_wheels"}]}).status_code == 404
