from __future__ import annotations

from typing import Callable, Optional

import torch
from diffusers import StableDiffusionXLPipeline
from tqdm.auto import tqdm

from teleportraits.types import PromptEmbeds, TrajectoryResult

PostStepHook = Callable[[int, int, int, torch.Tensor], torch.Tensor]


def run_denoise_trajectory(
    pipe: StableDiffusionXLPipeline,
    start_latents: torch.Tensor,
    prompt_embeds: PromptEmbeds,
    guidance_scale: float,
    num_inference_steps: int,
    attn_controller: Optional[object] = None,
    post_step_hook: Optional[PostStepHook] = None,
    stage_name: str = "Denoising",
    show_progress_bar: bool = True,
) -> TrajectoryResult:
    device = start_latents.device
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    latents = start_latents
    latents_by_timestep = {}

    step_iter = enumerate(timesteps)
    if show_progress_bar:
        step_iter = enumerate(
            tqdm(
                timesteps,
                total=len(timesteps),
                desc=stage_name,
                leave=False,
                dynamic_ncols=True,
            )
        )

    for i, t in step_iter:
        t_int = int(t.item())
        latents_by_timestep[t_int] = latents.detach().clone()

        if attn_controller is not None:
            attn_controller.set_step(step_index=i, timestep=t_int)

        if prompt_embeds.do_cfg:
            model_input = torch.cat([latents, latents], dim=0)
        else:
            model_input = latents
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

        if prompt_embeds.do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if post_step_hook is not None:
            next_t_int = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else 0
            latents = post_step_hook(i, t_int, next_t_int, latents)

    latents_by_timestep[0] = latents.detach().clone()
    return TrajectoryResult(
        final_latents=latents,
        latents_by_timestep=latents_by_timestep,
        timesteps=[int(t.item()) for t in timesteps],
    )
