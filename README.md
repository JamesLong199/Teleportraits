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
  --scene-prompt "a wide-angle city street at sunset" \
  --reference-prompt "a full-body photo of a woman in a red coat" \
  --edit-prompt "a woman in a red coat standing near the crosswalk" \
  --output-dir /tmp/teleportraits_out
```

Key outputs:

- `final.png`: final insertion result.
- `affordance_pass.png`: first-pass generation used for mask extraction.
- `scene_reconstruction.png`: reconstructed scene from inversion trajectory.
- `foreground_mask.png`: foreground mask used for latent blending.
- `reference_mask.png`: reference subject mask used for K/V masking.

## Recommended Defaults

- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- Scheduler: `DDIMScheduler`
- Inference steps: `50`
- Edit guidance scale: start around `7.5` to `12.0`
- Blend window (step indices): start around `15`, end around `40`

## Disclaimer

This repository may require tuning for your prompts/images and GPU memory budget.
