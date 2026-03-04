from __future__ import annotations

import inspect
from typing import Tuple

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from teleportraits.types import PromptEmbeds


def parse_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def encode_prompt_sdxl(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str,
    image_size: Tuple[int, int],
    device: torch.device,
    do_cfg: bool,
) -> PromptEmbeds:
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        do_classifier_free_guidance=do_cfg,
        device=device,
        num_images_per_prompt=1,
    )

    if do_cfg:
        text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    else:
        text_embeds = pooled_prompt_embeds

    width, height = image_size
    add_time_ids = _get_add_time_ids(pipe, height=height, width=width, dtype=prompt_embeds.dtype, device=device)

    if do_cfg:
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    return PromptEmbeds(
        prompt_embeds=prompt_embeds,
        add_text_embeds=text_embeds,
        add_time_ids=add_time_ids,
    )


def _get_add_time_ids(
    pipe: StableDiffusionXLPipeline,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)

    fn = pipe._get_add_time_ids
    signature = inspect.signature(fn)
    kwargs = {
        "original_size": original_size,
        "crops_coords_top_left": crops_coords_top_left,
        "target_size": target_size,
        "dtype": dtype,
    }

    if "text_encoder_projection_dim" in signature.parameters:
        kwargs["text_encoder_projection_dim"] = pipe.text_encoder_2.config.projection_dim

    add_time_ids = fn(**kwargs)
    add_time_ids = add_time_ids.to(device=device, dtype=dtype)
    return add_time_ids


def image_to_latents(
    pipe: StableDiffusionXLPipeline,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    image_tensor = pipe.image_processor.preprocess(image).to(device=device, dtype=dtype)
    needs_upcast = _vae_needs_upcast(pipe)
    original_vae_dtype = pipe.vae.dtype

    if needs_upcast:
        pipe.upcast_vae()
        vae_dtype = next(iter(pipe.vae.post_quant_conv.parameters())).dtype
        image_tensor = image_tensor.to(dtype=vae_dtype)

    latent_dist = pipe.vae.encode(image_tensor).latent_dist
    if hasattr(latent_dist, "mode"):
        latents = latent_dist.mode()
    else:
        latents = latent_dist.mean

    latents = latents * pipe.vae.config.scaling_factor
    latents = latents.to(device=device, dtype=dtype)

    if needs_upcast:
        pipe.vae.to(dtype=original_vae_dtype)

    return latents


def latents_to_image(pipe: StableDiffusionXLPipeline, latents: torch.Tensor) -> Image.Image:
    needs_upcast = _vae_needs_upcast(pipe)
    original_vae_dtype = pipe.vae.dtype

    latents = latents / pipe.vae.config.scaling_factor
    if needs_upcast:
        pipe.upcast_vae()
        vae_dtype = next(iter(pipe.vae.post_quant_conv.parameters())).dtype
        latents = latents.to(dtype=vae_dtype)

    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)

    if needs_upcast:
        pipe.vae.to(dtype=original_vae_dtype)

    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image[0]


def _vae_needs_upcast(pipe: StableDiffusionXLPipeline) -> bool:
    force_upcast = bool(getattr(pipe.vae.config, "force_upcast", False))
    return force_upcast and pipe.vae.dtype in {torch.float16, torch.bfloat16}
