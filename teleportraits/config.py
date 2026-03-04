from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TeleportraitConfig:
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 50

    inversion_guidance_scale: float = 1.0
    inversion_fixed_point_iters: int = 10

    edit_guidance_scale: float = 9.0
    negative_prompt: str = ""

    blend_start_step: int = 15
    blend_end_step: int = 40

    attention_enabled: bool = True
    attention_inject_start_step: int = 0
    attention_inject_end_step: int = 49
    attention_target_prefixes: Tuple[str, ...] = ("up_blocks.1", "up_blocks.2")

    mask_threshold: float = 0.08
    mask_min_area_ratio: float = 0.001

    seed: int = 0
    torch_dtype: str = "float16"
    device: str = "cuda"
