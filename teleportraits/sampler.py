from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from diffusers import ControlNetModel, StableDiffusionXLPipeline
from tqdm.auto import tqdm

from teleportraits.types import PromptEmbeds, TrajectoryResult

PostStepHook = Callable[[int, int, int, torch.Tensor], torch.Tensor]
ControlNetArg = Union[ControlNetModel, Sequence[ControlNetModel]]
ControlImageArg = Union[torch.Tensor, Sequence[torch.Tensor]]
ControlScaleArg = Union[float, Sequence[float]]


def run_denoise_trajectory(
    pipe: StableDiffusionXLPipeline,
    start_latents: torch.Tensor,
    prompt_embeds: PromptEmbeds,
    guidance_scale: float,
    num_inference_steps: int,
    attn_controller: Optional[object] = None,
    post_step_hook: Optional[PostStepHook] = None,
    controlnet: Optional[ControlNetArg] = None,
    control_image: Optional[ControlImageArg] = None,
    controlnet_conditioning_scale: ControlScaleArg = 1.0,
    controlnet_inject_start_step: int = 0,
    controlnet_inject_end_step: int = 999,
    controlnet_residual_suppress_mask: Optional[torch.Tensor] = None,
    controlnet_mask_inject_start_step: int = 0,
    controlnet_mask_inject_end_step: int = 999,
    stage_name: str = "Denoising",
    show_progress_bar: bool = True,
) -> TrajectoryResult:
    device = start_latents.device
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    controlnets, control_images, control_scales = _normalize_controlnet_inputs(
        controlnet=controlnet,
        control_image=control_image,
        conditioning_scale=controlnet_conditioning_scale,
    )

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

        down_block_res_samples = None
        mid_block_res_sample = None
        use_controlnet_this_step = (
            len(controlnets) > 0
            and controlnet_inject_start_step <= i <= controlnet_inject_end_step
        )
        if use_controlnet_this_step:
            for cn, cn_image, cn_scale in zip(controlnets, control_images, control_scales):
                controlnet_cond = cn_image
                if prompt_embeds.do_cfg and controlnet_cond.shape[0] == 1:
                    controlnet_cond = torch.cat([controlnet_cond, controlnet_cond], dim=0)

                down_i, mid_i = cn(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds.prompt_embeds,
                    controlnet_cond=controlnet_cond,
                    conditioning_scale=cn_scale,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
                if down_block_res_samples is None:
                    down_block_res_samples = tuple(down_i)
                    mid_block_res_sample = mid_i
                else:
                    down_block_res_samples = tuple(
                        prev + cur for prev, cur in zip(down_block_res_samples, down_i)
                    )
                    mid_block_res_sample = mid_block_res_sample + mid_i

            use_mask_this_step = (
                controlnet_residual_suppress_mask is not None
                and controlnet_mask_inject_start_step <= i <= controlnet_mask_inject_end_step
            )
            if use_mask_this_step:
                suppress_mask = _prepare_suppress_mask(
                    controlnet_residual_suppress_mask,
                    batch_size=down_block_res_samples[0].shape[0],
                    device=down_block_res_samples[0].device,
                    dtype=down_block_res_samples[0].dtype,
                )
                down_block_res_samples = tuple(
                    _apply_suppress_mask_to_feature(x, suppress_mask) for x in down_block_res_samples
                )
                mid_block_res_sample = _apply_suppress_mask_to_feature(mid_block_res_sample, suppress_mask)

        noise_pred = pipe.unet(
            model_input,
            t,
            encoder_hidden_states=prompt_embeds.prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
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


def _prepare_suppress_mask(
    suppress_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if suppress_mask.ndim != 4 or suppress_mask.shape[1] != 1:
        raise ValueError(
            "controlnet_residual_suppress_mask must have shape [B,1,H,W] or [1,1,H,W]."
        )
    out = suppress_mask.to(device=device, dtype=dtype)
    if out.shape[0] == 1 and batch_size > 1:
        out = out.repeat(batch_size, 1, 1, 1)
    elif out.shape[0] != batch_size:
        raise ValueError(
            f"controlnet_residual_suppress_mask batch {out.shape[0]} does not match model batch {batch_size}."
        )
    return out


def _apply_suppress_mask_to_feature(feature: torch.Tensor, suppress_mask: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(suppress_mask, size=feature.shape[-2:], mode="bilinear", align_corners=False)
    keep = 1.0 - resized
    return feature * keep


def _normalize_controlnet_inputs(
    controlnet: Optional[ControlNetArg],
    control_image: Optional[ControlImageArg],
    conditioning_scale: ControlScaleArg,
) -> tuple[list[ControlNetModel], list[torch.Tensor], list[float]]:
    if controlnet is None:
        return [], [], []

    if isinstance(controlnet, ControlNetModel):
        controlnets = [controlnet]
    else:
        controlnets = list(controlnet)
    if not controlnets:
        return [], [], []
    if not all(isinstance(x, ControlNetModel) for x in controlnets):
        raise TypeError("controlnet must be a ControlNetModel or a sequence of ControlNetModel.")

    if control_image is None:
        raise ValueError("control_image must be provided when controlnet is enabled")
    if isinstance(control_image, torch.Tensor):
        control_images = [control_image]
    else:
        control_images = list(control_image)
    if len(control_images) == 1 and len(controlnets) > 1:
        control_images = control_images * len(controlnets)
    if len(control_images) != len(controlnets):
        raise ValueError(
            "control_image count must match controlnet count. "
            f"Got {len(control_images)} images for {len(controlnets)} controlnets."
        )
    if not all(isinstance(x, torch.Tensor) for x in control_images):
        raise TypeError("control_image must be a tensor or a sequence of tensors.")

    if isinstance(conditioning_scale, (int, float)):
        scales = [float(conditioning_scale)] * len(controlnets)
    else:
        scales = [float(x) for x in conditioning_scale]
        if len(scales) == 1 and len(controlnets) > 1:
            scales = scales * len(controlnets)
        if len(scales) != len(controlnets):
            raise ValueError(
                "controlnet_conditioning_scale count must match controlnet count. "
                f"Got {len(scales)} scales for {len(controlnets)} controlnets."
            )

    return controlnets, control_images, scales
