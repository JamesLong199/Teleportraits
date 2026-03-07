from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TeleportraitConfig:
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 50
    image_size: int = 1024

    inversion_guidance_scale: float = 1.0
    inversion_fixed_point_iters: int = 2
    inversion_prompt: Optional[str] = None
    random_start_latent: bool = False

    edit_guidance_scale: float = 7.5
    affordance_use_controlnet_depth: bool = False
    affordance_controlnet_model_id: str = "diffusers/controlnet-depth-sdxl-1.0"
    affordance_controlnet_dir: str = "./pretrained/controlnet-depth-sdxl-1.0"
    affordance_controlnet_scale: float = 1.0
    affordance_controlnet_start_step: int = 0
    affordance_controlnet_end_step: int = 999
    affordance_use_controlnet_openpose: bool = False
    affordance_openpose_controlnet_model_id: str = "thibaud/controlnet-openpose-sdxl-1.0"
    affordance_openpose_controlnet_dir: str = "./pretrained/controlnet-openpose-sdxl-1.0"
    affordance_openpose_controlnet_scale: float = 1.0
    openpose_detector_model_id: str = "lllyasviel/Annotators"
    openpose_detector_dir: str = ""
    affordance_controlnet_mask_image: Optional[str] = None
    affordance_controlnet_mask_invert: bool = False
    affordance_controlnet_mask_start_step: int = 0
    affordance_controlnet_mask_end_step: int = 999
    moge_pretrained_model: str = "Ruicheng/moge-2-vitl-normal"
    moge_checkpoint_dir: str = "./pretrained/moge"
    moge_model_version: str = "v2"
    moge_conda_env: str = ""
    moge_use_fp16: bool = True

    negative_prompt: str = (
        "low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, "
        "extra limbs, missing fingers, malformed limbs, mutated, deformed, disfigured, "
        "poorly drawn, jpeg artifacts, watermark, text, logo, signature, cropped, out of frame"
    )

    blend_start_step: int = 10
    blend_end_step: int = 20

    attention_enabled: bool = True
    attention_inject_start_step: int = 0
    attention_inject_end_step: int = 49
    attention_target_prefixes: Tuple[str, ...] = ("up_blocks.1", "up_blocks.2")
    affordance_only: bool = False

    mask_threshold: float = 0.08
    mask_min_area_ratio: float = 0.001
    foreground_mask_prompt: str = "person"
    foreground_mask_confidence_threshold: float = 0.5
    sam3_checkpoint_dir: Optional[str] = None
    sam3_conda_env: str = "sam3"
    use_transformers_reference_mask: bool = False

    verbose: bool = True
    show_progress_bar: bool = True

    seed: int = 0
    torch_dtype: str = "float16"
    device: str = "cuda"
