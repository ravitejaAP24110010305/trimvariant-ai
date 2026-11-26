"""Evaluation report (Markdown) and CSV export of all generated variants."""
from __future__ import annotations

import csv
import io

from app.database import Database
from app.services import evaluation

CSV_FIELDS = [
    "variant_id", "image_id", "feature", "option", "mode", "seed", "status", "review", "qc_passed",
    "coverage", "raw_outside_ssim", "raw_outside_mae", "raw_edge_f1", "final_background_ssim",
    "inside_change", "halo_ssim", "clip_score", "duration_s", "prompt",
]


def _rows(variants: list[dict]) -> list[dict]:
    rows = []
    for v in variants:
        m = v.get("metrics") or {}
        rows.append({
            "variant_id": v["id"], "image_id": v["image_id"], "feature": v["feature"], "option": v["option"] or "",
            "mode": v["mode"], "seed": v["seed"], "status": v["status"], "review": v["review_decision"],
            "qc_passed": (v.get("qc") or {}).get("passed", ""), "duration_s": v.get("duration_s") or "",
            "prompt": v["prompt"], **{k: m.get(k, "") if m.get(k) is not None else "" for k in CSV_FIELDS[9:17]},
        })
    return rows


def to_csv(db: Database, image_id: str | None = None) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(_rows(db.list_variants(image_id)))
    return buf.getvalue()


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def to_markdown(db: Database, image_id: str | None = None) -> str:
    variants = db.list_variants(image_id)
    done = [v for v in variants if v["status"] == "done"]
    lines = ["# TrimVariant AI - Evaluation Report", ""]
    lines += [f"- Variants generated: **{len(variants)}** (completed: {len(done)}, "
              f"failed: {sum(v['status'] == 'failed' for v in variants)})",
              f"- Passed automatic QC: **{sum(1 for v in done if v['qc']['passed'])}** / {len(done)}",
              f"- Human-approved: **{sum(1 for v in done if v['review_decision'] == 'approved')}**, "
              f"rejected: **{sum(1 for v in done if v['review_decision'] == 'rejected')}**", ""]

    summary = evaluation.summarize_by_mode(variants)
    lines += ["## Plain inpainting vs ControlNet-guided inpainting", "",
              "| Metric | Inpaint | ControlNet |", "|---|---|---|",
              f"| Variants | {summary['inpaint']['count']} | {summary['controlnet']['count']} |"]
    for key, label in [("raw_outside_ssim", "Background SSIM (higher = better)"),
                       ("raw_outside_mae", "Background MAE (lower = better)"),
                       ("raw_edge_f1", "Structure edge F1 (higher = better)"),
                       ("inside_change", "Feature change magnitude"),
                       ("halo_ssim", "Halo SSIM (higher = better)"),
                       ("qc_pass_rate", "QC pass rate")]:
        lines.append(f"| {label} | {_fmt(summary['inpaint'][key])} | {_fmt(summary['controlnet'][key])} |")

    lines += ["", "## All variants", "",
              "| ID | Feature | Mode | Seed | BG SSIM | Edge F1 | Change | Halo | QC | Review |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for v in variants:
        m = v.get("metrics") or {}
        qc = "-" if not v.get("qc") else ("PASS" if v["qc"]["passed"] else "FAIL: " + ", ".join(v["qc"]["failed"]))
        lines.append(f"| {v['id']} | {v['feature']}{' / ' + v['option'] if v['option'] else ''} | {v['mode']} | "
                     f"{v['seed']} | {_fmt(m.get('raw_outside_ssim'))} | {_fmt(m.get('raw_edge_f1'))} | "
                     f"{_fmt(m.get('inside_change'))} | {_fmt(m.get('halo_ssim'))} | {qc} | "
                     f"{v['review_decision']} |")

    failures: dict[str, int] = {}
    for v in done:
        for name in v["qc"]["failed"]:
            failures[name] = failures.get(name, 0) + 1
    lines += ["", "## Common limitations observed (QC failures)", ""]
    lines += [f"- `{k}`: {n} variant(s)" for k, n in sorted(failures.items(), key=lambda kv: -kv[1])] or [
        "- No automatic QC failures recorded."]
    lines += ["", "_Metrics are automatic heuristics; every variant still needs human review before marketing use._", ""]
    return "\n".join(lines)
