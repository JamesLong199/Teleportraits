from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from diffusers import StableDiffusionXLPipeline

from teleportraits.types import PromptEmbeds


@dataclass
class InversionResult:
    start_latents: torch.Tensor
    latents_by_timestep: Dict[int, torch.Tensor]
    timesteps_ascending: List[int]


def ddim_fixed_point_invert(
    pipe: StableDiffusionXLPipeline,
    clean_latents: torch.Tensor,
    prompt_embeds: PromptEmbeds,
    guidance_scale: float,
    num_inference_steps: int,
    fixed_point_iters: int,
) -> InversionResult:
    device = clean_latents.device
    dtype = clean_latents.dtype

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps_desc = [int(t.item()) for t in pipe.scheduler.timesteps]
    timesteps_asc = list(reversed(timesteps_desc))

    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device=device, dtype=dtype)

    latents_by_timestep: Dict[int, torch.Tensor] = {}
    x_prev = clean_latents
    latents_by_timestep[timesteps_asc[0]] = x_prev.detach().clone()

    for prev_t, curr_t in zip(timesteps_asc[:-1], timesteps_asc[1:]):
        alpha_prev = alphas_cumprod[prev_t]
        alpha_curr = alphas_cumprod[curr_t]

        coeff_x = torch.sqrt(alpha_curr / alpha_prev)
        coeff_eps = torch.sqrt(1.0 - alpha_curr) - coeff_x * torch.sqrt(1.0 - alpha_prev)

        x_t = coeff_x * x_prev

        for _ in range(fixed_point_iters):
            eps = _predict_noise_cfg(
                pipe=pipe,
                latents=x_t,
                timestep=curr_t,
                prompt_embeds=prompt_embeds,
                guidance_scale=guidance_scale,
            )
            x_t = coeff_x * x_prev + coeff_eps * eps

        x_prev = x_t
        latents_by_timestep[curr_t] = x_prev.detach().clone()

    return InversionResult(
        start_latents=x_prev,
        latents_by_timestep=latents_by_timestep,
        timesteps_ascending=timesteps_asc,
    )


def _predict_noise_cfg(
    pipe: StableDiffusionXLPipeline,
    latents: torch.Tensor,
    timestep: int,
    prompt_embeds: PromptEmbeds,
    guidance_scale: float,
) -> torch.Tensor:
    t = torch.tensor(timestep, device=latents.device, dtype=torch.long)

    model_input = torch.cat([latents, latents], dim=0)
    model_input = pipe.scheduler.scale_model_input(model_input, t)

    added_cond_kwargs = {
        "text_embeds": prompt_embeds.add_text_embeds,
        "time_ids": prompt_embeds.add_time_ids,
    }

    noise_pred = pipe.unet(
        model_input,
        t,
        encoder_hidden_states=prompt_embeds.prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
