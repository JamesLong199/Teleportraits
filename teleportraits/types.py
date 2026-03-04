from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class PromptEmbeds:
    prompt_embeds: torch.Tensor
    add_text_embeds: torch.Tensor
    add_time_ids: torch.Tensor


@dataclass
class TrajectoryResult:
    final_latents: torch.Tensor
    latents_by_timestep: Dict[int, torch.Tensor]
    timesteps: List[int]
