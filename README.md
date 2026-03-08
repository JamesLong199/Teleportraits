# Teleportraits (Unofficial Reproduction)

This repository implements an end-to-end, training-free pipeline inspired by:

- Jialu Gao, K J Joseph, Fernando De La Torre.
- "Teleportraits: Training-Free People Insertion into Any Scene" (ICCV 2025 / arXiv:2510.05660).

The implementation is built on top of `diffusers` SDXL and includes the key components described in the paper:

- DDIM inversion with fixed-point refinement.
- Two-pass affordance-aware insertion with latent blending.
- Mask-guided self-attention via reference K/V injection in SDXL up-block self-attention.

## Notes

- This is a faithful engineering reproduction, not official code from the authors.
- Some constants hidden in HTML-rendered equations are exposed as CLI/config knobs.
- You can tune these values to match paper figures once you compare outputs.

## Install

```bash
# Conda (recommended on server)
conda env create -f environment.yml
conda activate teleportraits

# Then install this package
pip install -e .
```

```bash
# venv (local alternative)
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
python -m teleportraits.cli \
  --input-json input_config.example.json
```

All runtime/model/pass settings are now read from JSON and organized by pass type:

- `passes.inversion`
- `passes.affordance`
- `passes.affordance_refine` (OpenPose second affordance pass)
- `passes.final`

Each of `passes.affordance`, `passes.affordance_refine`, and `passes.final` includes both:

- `controlnet_depth`
- `controlnet_pose`

The input JSON is copied into each run folder as `input_config.json` for debugging/reproducibility.

Final generation prompt is explicitly provided by `passes.final.prompt` (no automatic scene/reference prompt composition).

Key outputs:

- `final.png`: final insertion result.
- `initial_pass.png`: initial human generation pass used for mask extraction.
- `affordance_pass.png`: affordance result used for downstream masking/final pass.
- `affordance_pose.png`: OpenPose map extracted from `initial_pass.png` when openpose refinement is enabled.
- `scene_reconstruction.png`: reconstructed scene from inversion trajectory.
- `foreground_mask.png`: foreground mask used for latent blending.
- `reference_mask.png`: reference subject mask used for K/V masking.

Mask-related runtime settings:

- `runtime.mask_threshold`: legacy threshold for a pixel-difference foreground mask path. Kept for backward compatibility; current default foreground extraction uses SAM3 and does not rely on this threshold.
- `runtime.mask_min_area_ratio`: minimum connected-component area ratio kept after binary-mask cleanup (used by SAM3 foreground mask post-processing).
- `runtime.use_transformers_reference_mask`: when `true`, use a transformers person-segmentation model for reference mask extraction; when `false`, use the heuristic reference mask extractor.

Final masks are configured at:

- `passes.final.masks.scene_foreground_mask` (`source: "affordance_pass_sam3"` placeholder, or `source: "file"` + `path`)
- `passes.final.masks.reference_foreground_mask`

## Recommended Defaults

- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- Scheduler: `DDIMScheduler`
- Inference steps: `50`
- Per-pass guidance scales (`passes.affordance.guidance_scale`, `passes.affordance_refine.guidance_scale`, `passes.final.guidance_scale`): `7.5` as a starting point
- Blend window (step indices): `10` to `20` (paper-like starting point)

## Progress Output

- Stage logs are controlled via `runtime.verbose`.
- Diffusion/inversion progress bars are controlled via `runtime.show_progress_bar`.

## Resume Behavior

- Each run is saved under a time-based subdirectory:
  - `--output-dir /tmp/teleportraits_out` creates `/tmp/teleportraits_out/exp_YYYYmmdd_HHMMSS/`
- Re-run with the same `--output-dir`:
  - always creates a new `exp_*` directory
- To resume a specific run directly, pass that subdirectory as `--output-dir`, e.g.:
  - `--output-dir /tmp/teleportraits_out/exp_20260304_101530`

## Disclaimer

This repository may require tuning for your prompts/images and GPU memory budget.

## Troubleshooting

- Black outputs or `invalid value encountered in cast`:
  - Set `runtime.torch_dtype` to `"float32"`.
  - Reduce `passes.inversion.fixed_point_iters`.
- Poor reference mask:
  - Default behavior uses a background-color heuristic (good for white/clean background portraits).
  - Optional model-based mask: set `runtime.use_transformers_reference_mask=true` (downloads extra weights).
  - For best results, use a clean reference portrait with strong subject/background contrast.
