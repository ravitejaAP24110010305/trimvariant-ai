# TrimVariant AI – Vehicle Feature Variant Editor

AI-assisted image editing for automotive marketing. Pick a feature on a vehicle photo (wheels, grille, roof, spoiler, paint…), describe the trim variant, and generate it with **Stable Diffusion Inpainting**. **ControlNet (canny)** keeps the vehicle's structure and geometry intact. Every result is scored automatically and goes through quality-control checkpoints plus human review.

## What it does

| Project requirement | Where it lives |
|---|---|
| Understand inpainting + structural conditioning | This README ("How it works"), `docs/REPORT_TEMPLATE.md` |
| **Prepare vehicle images**, identify editable regions | `app/services/image_prep.py`, `app/services/regions.py` (upload validation, resize to SD-safe size, brush/box/polygon/ellipse masks, automatic wheel detection) |
| **Implement inpainting** preserving surroundings | `app/services/pipeline.py` (`StableDiffusionInpaintPipeline`) + feathered paste-back in `workflow.py` |
| **Apply ControlNet guidance** | `pipeline.py` (`StableDiffusionControlNetInpaintPipeline` + canny ControlNet), `image_prep.build_control_image` |
| **Analyze generated variants** (accuracy, quality, structure) | `app/services/evaluation.py`, side-by-side "Compare both pipelines" (same seed + mask) |
| **Evaluate the workflow**, define QC checkpoints | `evaluation.run_qc`, human approve/reject, `report.py` (Markdown + CSV) |
| Teamwork / version control / code quality | Git-friendly layout, tests in `tests/`, typed config |

### How it works
```
 photo ──► validate + resize (multiple of 8) ──► mask (brush / box / auto-wheels) ──► dilate
                                                      │
                          canny edges (edges inside the mask removed) ──► ControlNet
                                                      ▼
            SD-Inpaint (baseline)   or   SD-Inpaint + ControlNet ──► raw output
                                                      ▼
        feathered paste-back (pixels outside the mask stay identical) ──► final
                                                      ▼
           metrics + QC checkpoints ──► SQLite ──► gallery, report, CSV
```
*Inpainting* regenerates only the masked pixels from a text prompt. *ControlNet* adds an edge map as an extra condition, so body lines, wheel arches and window frames around the edit stay where they were.

## Project structure
```
trimvariant-ai/
├── app/
│   ├── main.py                FastAPI app + REST API
│   ├── config.py              settings (.env)
│   ├── database.py            SQLite (images, variants, metrics, reviews)
│   ├── schemas.py             request validation
│   ├── services/
│   │   ├── image_prep.py      validation, resizing, canny / control image
│   │   ├── regions.py         masks, wheel detection, feathering
│   │   ├── prompts.py         feature catalogue + prompt building
│   │   ├── pipeline.py        diffusers engine (inpaint / controlnet)
│   │   ├── evaluation.py      metrics + QC checkpoints
│   │   ├── workflow.py        job orchestration (background worker)
│   │   ├── library.py         image storage
│   │   ├── report.py          Markdown / CSV report
│   │   └── sample_data.py     synthetic test car
│   └── static/                index.html · style.css · app.js (web UI)
├── scripts/                   cli.py · make_sample_image.py
├── tests/                     pytest suite
├── docs/REPORT_TEMPLATE.md    outline for the final report / presentation
├── data/samples/              put your car photos here
├── requirements.txt · .env.example · pytest.ini · .vscode/
```

## Setup in VS Code

**Requirements:** Python 3.10–3.13, Git. An NVIDIA GPU with ≥ 6 GB VRAM is strongly recommended (≈ 5–15 s per image). CPU works but takes several minutes per image.

1. Open the folder: `File ▸ Open Folder… ▸ trimvariant-ai`. Install the recommended extensions when prompted.
2. Create a virtual environment (`Ctrl+Shift+` ` → terminal):
   ```bash
   python -m venv .venv
   # Windows PowerShell:  .venv\Scripts\Activate.ps1      macOS/Linux:  source .venv/bin/activate
   ```
   Then `Ctrl+Shift+P ▸ Python: Select Interpreter ▸ .venv`.
3. **(NVIDIA GPU only)** install the CUDA build of PyTorch first – pick your command at <https://pytorch.org/get-started/locally/>, e.g.
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
4. Install the rest:
   ```bash
   pip install -r requirements.txt
   ```
5. Configure (optional): `copy .env.example .env` (Windows) / `cp .env.example .env`. Defaults work out of the box.
6. Run the tests (no GPU or model download needed):
   ```bash
   python -m pytest
   ```
7. Start the app – press **F5** and choose *"TrimVariant: run server (reload)"*, or:
   ```bash
   python -m uvicorn app.main:app --reload --port 8000
   ```
   Open <http://localhost:8000>. API docs: <http://localhost:8000/docs>.

The **first generation downloads the models** (~5.5 GB, cached afterwards). A top-right chip in the UI shows whether the engine is ready.

## Using the web UI
1. **Upload** a car photo (side view works best for auto-wheel detection). `python scripts/make_sample_image.py` creates a synthetic drawing only for a smoke test – use real photos for real results.
2. Choose a **feature** and **option** (or *Custom feature* and describe it).
3. **Mark the region**: *Auto-detect wheels*, a *Quick region*, or paint with the brush/rectangle. Keep the mask tight around the feature.
4. Press **Generate variant**, or **Compare both pipelines** to run plain inpainting and ControlNet with the same seed and mask.
5. Click a card to inspect: before/after slider, raw output, mask, ControlNet edge map, metrics, QC checks. **Approve / Reject** records the human decision.
6. Download the **Report (.md)** or **CSV** from the top bar for your project report.

### Tuning tips
- Wheels/grilles: ControlNet scale 0.8–1.2; steps 25–35; guidance 7–9.
- Feature outline drifting? Raise ControlNet scale or tick *Keep original feature outline*.
- Visible edge around the edit? Raise *Edge feather* and *Mask grow*.
- Out of VRAM? Set `CPU_OFFLOAD=true`, `KEEP_BOTH_MODELS=false`, lower `MAX_IMAGE_SIDE` to 512.

## Command line
```bash
python scripts/cli.py add data/samples/car.jpg --view side          # prints IMAGE_ID
python scripts/cli.py generate IMAGE_ID --feature wheels --option "bronze turbine-style alloy wheel" --auto-wheels
python scripts/cli.py compare  IMAGE_ID --feature grille --box 0.3 0.5 0.7 0.68 --seed 42
python scripts/cli.py report --out report.md --csv variants.csv
```

## REST API (summary)
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/features` | engine status, feature catalogue, defaults |
| POST/GET/DELETE | `/api/images` | upload, list, delete images |
| POST | `/api/images/{id}/detect-wheels` | auto wheel detection |
| POST | `/api/variants` | queue one variant (`mode`: `inpaint` \| `controlnet`) |
| POST | `/api/compare` | queue both pipelines, same seed + mask |
| GET | `/api/variants[?image_id=]`, `/api/variants/{id}` | status, metrics, QC |
| GET | `/api/variants/{id}/files/{output\|raw\|mask\|control}` | result images |
| POST | `/api/variants/{id}/review` | human approve / reject |
| GET | `/api/report.md`, `/api/report.csv` | evaluation outputs |

Regions are JSON: `{"type":"box","box":[x0,y0,x1,y1]}`, `polygon`, `ellipse`, `auto_wheels`, or `mask_png` – all coordinates normalized 0–1.

## Metrics and QC checkpoints
Scored on the **raw** model output (fair comparison of inpaint vs ControlNet) and the **final** composite:

| Metric | Meaning | Default QC rule |
|---|---|---|
| Background SSIM / MAE | did the model leave everything outside the mask alone? | SSIM ≥ 0.80 |
| Structure edge F1 | canny edges outside the mask match the original (geometry preserved) | ≥ 0.50 |
| Halo SSIM | no colour bleed in a ring just outside the edit | ≥ 0.80 |
| Feature change | the edit visibly differs from the original | ≥ 0.04 |
| Coverage | mask is neither trivial nor excessive | 0.5 % – 45 % |
| CLIP similarity (optional) | edited crop vs prompt (`ENABLE_CLIP_SCORE=true`) | informational |

Thresholds are **starting values** – calibrate them on your own images in `.env`. Automatic checks never replace human review: no variant should reach marketing use without it.

## Limitations (worth stating in your report)
- Stable Diffusion 1.5 can invent wheel/grille details; it does not know real OEM trim designs or logos. Always review for brand accuracy.
- Auto wheel detection is a Hough-circle heuristic for side views; use the brush on other angles.
- Canny guidance preserves geometry, not colour or reflections; large masks (> ~25 % of the image) drift more.
- Output is limited to `MAX_IMAGE_SIDE` (default 768 px) – upscale separately if print resolution is needed.
- The safety checker is disabled (car images trigger false positives); keep the app private or re-enable it before exposing it publicly.

## Note on the assignment text
The "Deliverables" and "Evaluation Criteria" screenshots mention a text-generation model and marketing copy, which does not match the image-editing problem statement, objectives and milestones. This project follows the problem statement. The equivalents are: preprocessed dataset → prepared images + masks; model checkpoint → configured pretrained SD/ControlNet (fine-tuning is out of scope for a 1-week project); text-generation script → `scripts/cli.py` + sample variants; report → `/api/report.md`. Confirm with your instructor if the deliverables list is authoritative.
