import numpy as np
import torch

from teleportraits.blending import BlendWindow, LatentBlender, build_latent_masks


def test_build_latent_masks_shape_and_range() -> None:
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0

    latent_shape = torch.Size([1, 4, 4, 4])
    fg = build_latent_masks(mask, latent_shape, device=torch.device("cpu"), dtype=torch.float32)

    assert fg.shape == (1, 1, 4, 4)
    assert float(fg.min()) >= 0.0
    assert float(fg.max()) <= 1.0


def test_latent_blender_applies_only_inside_window() -> None:
    fg_mask = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    fg_mask[:, :, 0, 0] = 0.0

    scene_latent = torch.full((1, 4, 2, 2), -1.0)
    generated = torch.full((1, 4, 2, 2), 2.0)

    blender = LatentBlender(
        scene_trajectory={10: scene_latent},
        fg_mask_latent=fg_mask,
        window=BlendWindow(start_step=1, end_step=3),
    )

    unchanged = blender(step_index=0, _timestep=20, next_timestep=10, latents=generated)
    assert torch.allclose(unchanged, generated)

    blended = blender(step_index=2, _timestep=20, next_timestep=10, latents=generated)
    assert blended[0, 0, 0, 0].item() == -1.0
    assert blended[0, 0, 1, 1].item() == 2.0
