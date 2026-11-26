"""Write a synthetic side-view car drawing to data/samples/synthetic_car.png (pipeline smoke test only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.sample_data import make_synthetic_car  # noqa: E402

out = Path(__file__).resolve().parent.parent / "data" / "samples" / "synthetic_car.png"
out.parent.mkdir(parents=True, exist_ok=True)
make_synthetic_car().save(out)
print(f"Wrote {out}")
