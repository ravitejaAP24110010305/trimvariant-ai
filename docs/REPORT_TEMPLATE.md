# TrimVariant AI – Project Report (template)

Team: ______  ·  Date: ______

## 1. Problem and motivation
Automakers need imagery for many trim/feature/market combinations; photographing every one is expensive. Goal: an AI-assisted workflow that edits selected features while preserving the vehicle and scene.

## 2. Background (Objective 1 – Understand)
- Latent diffusion and text-conditioned inpainting (mask, prompt, VAE latent blending).
- ControlNet: extra conditioning (canny edges) that constrains structure.

## 3. Week 1 – Image preparation and feature identification
- Source images (count, views, licence): ______
- Preprocessing: validation, resize to ≤ 768 px, multiples of 8.
- Editable regions identified (wheels, grille, roof, spoiler…): ______  Masking method: ______

## 4. Week 2 – Inpainting pipeline
- Model: `stable-diffusion-v1-5/stable-diffusion-inpainting`; steps / guidance / strength used: ______
- Example results (insert original, mask, output): ______

## 5. Week 3 – ControlNet-guided generation
- ControlNet: `lllyasviel/control_v11p_sd15_canny`; conditioning scale ______; inner edges kept? ______
- Inpaint vs ControlNet (use the table from `GET /api/report.md`, same seed and mask): ______

## 6. Week 4 – Evaluation and quality control
- Metrics definitions (see README) and observed values: ______
- QC checkpoints and thresholds chosen (with justification): ______
- Human review outcomes (approved / rejected counts): ______
- Failure cases and limitations (brand accuracy, geometry drift, seams): ______

## 7. Conclusions and future work
ControlNet-inpaint (canny) vs depth/segmentation conditioning, SDXL inpainting, LoRA fine-tuning on OEM wheel catalogues, automatic segmentation (SAM) for masks, batch generation per market.

## 8. Contributions (4 members) and Git history
______
