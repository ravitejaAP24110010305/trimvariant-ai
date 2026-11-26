"""Command-line interface for batch work (no web server needed).

Examples
  python scripts/cli.py add data/samples/car.jpg --view side
  python scripts/cli.py generate IMAGE_ID --feature wheels --option "gloss black split 10-spoke sport alloy wheel" --auto-wheels
  python scripts/cli.py compare IMAGE_ID --feature grille --box 0.3 0.5 0.7 0.68 --seed 42
  python scripts/cli.py report --out report.md --csv variants.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.database import Database  # noqa: E402
from app.schemas import GenerateRequest, RegionSpec  # noqa: E402
from app.services import library, report  # noqa: E402
from app.services.pipeline import DiffusersEngine  # noqa: E402
from app.services.workflow import VariantWorkflow  # noqa: E402


def build_regions(args: argparse.Namespace) -> list[RegionSpec]:
    regions: list[RegionSpec] = []
    if args.auto_wheels:
        regions.append(RegionSpec(type="auto_wheels"))
    if args.box:
        regions.append(RegionSpec(type="box", box=list(args.box)))
    if args.mask:
        import base64

        regions.append(RegionSpec(type="mask_png", data=base64.b64encode(Path(args.mask).read_bytes()).decode()))
    if not regions:
        sys.exit("Provide a region: --auto-wheels, --box X0 Y0 X1 Y1 (0..1) or --mask mask.png")
    return regions


def request_from(args: argparse.Namespace, mode: str) -> GenerateRequest:
    return GenerateRequest(
        image_id=args.image_id, feature=args.feature, option=args.option, custom_prompt=args.prompt,
        mode=mode, regions=build_regions(args), steps=args.steps, guidance_scale=args.guidance,
        controlnet_scale=args.cn_scale, keep_inner_edges=args.keep_inner_edges, seed=args.seed,
    )


def show(db: Database, variant_id: str) -> None:
    v = db.get_variant(variant_id)
    assert v is not None
    if v["status"] != "done":
        print(f"[{v['mode']}] {variant_id}: {v['status']} - {v['error']}")
        return
    m = v["metrics"]
    verdict = "PASS" if v["qc"]["passed"] else "FAIL (" + ", ".join(v["qc"]["failed"]) + ")"
    print(f"[{v['mode']}] {variant_id}: QC {verdict} | bgSSIM={m['raw_outside_ssim']} edgeF1={m['raw_edge_f1']} "
          f"change={m['inside_change']} -> {v['output_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TrimVariant AI command line")
    sub = parser.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add", help="register a vehicle image")
    add.add_argument("path")
    add.add_argument("--view", default="side", choices=sorted(library.VALID_VIEWS))

    sub.add_parser("list", help="list registered images")

    for name in ("generate", "compare"):
        g = sub.add_parser(name, help="generate one variant" if name == "generate" else "run inpaint AND controlnet")
        g.add_argument("image_id")
        g.add_argument("--feature", required=True)
        g.add_argument("--option")
        g.add_argument("--prompt", help="extra details (or full text for feature=custom)")
        g.add_argument("--auto-wheels", action="store_true")
        g.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
        g.add_argument("--mask", help="PNG mask file, white = edit")
        g.add_argument("--steps", type=int)
        g.add_argument("--guidance", type=float)
        g.add_argument("--cn-scale", type=float)
        g.add_argument("--keep-inner-edges", action="store_true")
        g.add_argument("--seed", type=int)
        if name == "generate":
            g.add_argument("--mode", choices=["inpaint", "controlnet"], default="controlnet")

    rep = sub.add_parser("report", help="write evaluation report")
    rep.add_argument("--image-id")
    rep.add_argument("--out", default="report.md")
    rep.add_argument("--csv")

    args = parser.parse_args()
    settings = get_settings()
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.init()

    if args.cmd == "add":
        path = Path(args.path)
        image = library.register_image(settings, db, path.name, path.read_bytes(), args.view)
        print(f"Registered {image['id']} ({image['width']}x{image['height']})")
    elif args.cmd == "list":
        for i in db.list_images():
            print(f"{i['id']}  {i['filename']}  {i['width']}x{i['height']}  {i['view']}")
    elif args.cmd in ("generate", "compare"):
        workflow = VariantWorkflow(settings, db, DiffusersEngine(settings))
        if args.cmd == "generate":
            ids = [workflow.create_variant(request_from(args, args.mode))]
        else:
            ids = workflow.create_compare(request_from(args, "controlnet"))
        for variant_id in ids:
            workflow.execute(variant_id)
            show(db, variant_id)
        workflow.shutdown()
    elif args.cmd == "report":
        Path(args.out).write_text(report.to_markdown(db, args.image_id), encoding="utf-8")
        print(f"Wrote {args.out}")
        if args.csv:
            Path(args.csv).write_text(report.to_csv(db, args.image_id), encoding="utf-8")
            print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
