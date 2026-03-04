from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BlendWindow:
    start_step: int
    end_step: int

    def contains(self, step_index: int) -> bool:
        return self.start_step <= step_index <= self.end_step


def build_latent_masks(mask_np: np.ndarray, latent_shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if mask_np.ndim != 2:
        raise ValueError("Foreground mask must be HxW")

    fg = torch.from_numpy(mask_np.astype(np.float32))[None, None, :, :].to(device=device, dtype=dtype)
    fg = F.interpolate(fg, size=(latent_shape[-2], latent_shape[-1]), mode="nearest")
    fg = fg.clamp(0.0, 1.0)
    return fg


class LatentBlender:
    def __init__(
        self,
        scene_trajectory: Dict[int, torch.Tensor],
        fg_mask_latent: torch.Tensor,
        window: BlendWindow,
    ) -> None:
        self.scene_trajectory = scene_trajectory
        self.fg_mask_latent = fg_mask_latent
        self.bg_mask_latent = 1.0 - fg_mask_latent
        self.window = window

    def __call__(self, step_index: int, _timestep: int, next_timestep: int, latents: torch.Tensor) -> torch.Tensor:
        if not self.window.contains(step_index):
            return latents

        scene_latent = self.scene_trajectory.get(next_timestep)
        if scene_latent is None:
            return latents

        scene_latent = scene_latent.to(device=latents.device, dtype=latents.dtype)
        return self.fg_mask_latent * latents + self.bg_mask_latent * scene_latent
