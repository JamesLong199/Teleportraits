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
teleportraits \
  --scene-image /path/to/scene.jpg \
  --reference-image /path/to/person.jpg \
  --reference-mask-image /path/to/person_mask.png \
  --scene-prompt "a wide-angle city street at sunset with a person near the crosswalk" \
  --reference-prompt "a full-body photo of a woman in a red coat" \
  --output-dir /tmp/teleportraits_out
```

By default, the insertion prompt is composed in paper style by replacing `a person` in the scene prompt with the reference prompt text. You can override with `--edit-prompt` if needed.

Key outputs:

- `final.png`: final insertion result.
- `affordance_pass.png`: first-pass generation used for mask extraction.
- `scene_reconstruction.png`: reconstructed scene from inversion trajectory.
- `foreground_mask.png`: foreground mask used for latent blending.
- `reference_mask.png`: reference subject mask used for K/V masking.

If `--reference-mask-image` is provided, it is used directly and segmentation is skipped.

## Recommended Defaults

- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- Scheduler: `DDIMScheduler`
- Inference steps: `50`
- Edit guidance scale: start around `7.5` to `12.0`
- Blend window (step indices): start around `15`, end around `40`

## Disclaimer

This repository may require tuning for your prompts/images and GPU memory budget.

## Troubleshooting

- Black outputs or `invalid value encountered in cast`:
  - Run with `--torch-dtype float32`.
  - Reduce inversion iterations, e.g. `--inversion-fixed-point-iters 3`.
- Poor reference mask:
  - Default behavior uses a background-color heuristic (good for white/clean background portraits).
  - Optional model-based mask: add `--use-transformers-reference-mask` (downloads extra weights).
  - For best results, use a clean reference portrait with strong subject/background contrast.
