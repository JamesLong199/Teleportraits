from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from PIL import Image

from teleportraits.attention import (
    AttentionWindow,
    MaskGuidedAttentionController,
    install_mask_guided_processors,
    restore_processors,
)
from teleportraits.blending import BlendWindow, LatentBlender, build_latent_masks
from teleportraits.config import TeleportraitConfig
from teleportraits.inversion import ddim_fixed_point_invert
from teleportraits.sampler import run_denoise_trajectory
from teleportraits.sdxl_utils import (
    encode_prompt_sdxl,
    image_to_latents,
    latents_to_image,
    load_image,
    parse_dtype,
)
from teleportraits.segmentation import (
    DifferenceMaskExtractor,
    TransformersPersonMaskExtractor,
    reference_person_mask,
)


class TeleportraitsPipeline:
    def __init__(self, config: TeleportraitConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = parse_dtype(config.torch_dtype)

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            config.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        self.diff_mask_extractor = DifferenceMaskExtractor(
            threshold=config.mask_threshold,
            min_area_ratio=config.mask_min_area_ratio,
        )

        self.reference_mask_extractor: Optional[TransformersPersonMaskExtractor]
        if config.attention_enabled:
            self.reference_mask_extractor = TransformersPersonMaskExtractor()
        else:
            self.reference_mask_extractor = None

    @torch.no_grad()
    def run(
        self,
        scene_image_path: str,
        reference_image_path: str,
        scene_prompt: str,
        reference_prompt: str,
        edit_prompt: str,
        output_dir: str,
    ) -> Dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        scene_image = _resize_to_multiple_of_8(load_image(scene_image_path))
        reference_image = _resize_to_multiple_of_8(load_image(reference_image_path))

        scene_prompt_embeds = encode_prompt_sdxl(
            self.pipe,
            prompt=scene_prompt,
            negative_prompt=self.config.negative_prompt,
            image_size=scene_image.size,
            device=self.device,
            do_cfg=True,
        )

        reference_prompt_embeds = encode_prompt_sdxl(
            self.pipe,
            prompt=reference_prompt,
            negative_prompt=self.config.negative_prompt,
            image_size=reference_image.size,
            device=self.device,
            do_cfg=True,
        )

        edit_prompt_embeds = encode_prompt_sdxl(
            self.pipe,
            prompt=edit_prompt,
            negative_prompt=self.config.negative_prompt,
            image_size=scene_image.size,
            device=self.device,
            do_cfg=True,
        )

        scene_latents = image_to_latents(self.pipe, scene_image, device=self.device, dtype=self.dtype)
        reference_latents = image_to_latents(self.pipe, reference_image, device=self.device, dtype=self.dtype)

        scene_inv = ddim_fixed_point_invert(
            pipe=self.pipe,
            clean_latents=scene_latents,
            prompt_embeds=scene_prompt_embeds,
            guidance_scale=self.config.inversion_guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            fixed_point_iters=self.config.inversion_fixed_point_iters,
        )

        reference_inv = ddim_fixed_point_invert(
            pipe=self.pipe,
            clean_latents=reference_latents,
            prompt_embeds=reference_prompt_embeds,
            guidance_scale=self.config.inversion_guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            fixed_point_iters=self.config.inversion_fixed_point_iters,
        )

        scene_reconstruction = run_denoise_trajectory(
            pipe=self.pipe,
            start_latents=scene_inv.start_latents,
            prompt_embeds=scene_prompt_embeds,
            guidance_scale=1.0,
            num_inference_steps=self.config.num_inference_steps,
        )
        scene_reconstruction_image = latents_to_image(self.pipe, scene_reconstruction.final_latents)

        affordance_pass = run_denoise_trajectory(
            pipe=self.pipe,
            start_latents=scene_inv.start_latents,
            prompt_embeds=edit_prompt_embeds,
            guidance_scale=self.config.edit_guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
        )
        affordance_image = latents_to_image(self.pipe, affordance_pass.final_latents)

        foreground_mask = self.diff_mask_extractor.extract(affordance_image, scene_reconstruction_image)

        reference_mask = reference_person_mask(reference_image, self.reference_mask_extractor)
        reference_mask_tensor = torch.from_numpy(reference_mask).to(device=self.device, dtype=self.dtype)

        blend_window = BlendWindow(
            start_step=self.config.blend_start_step,
            end_step=min(self.config.blend_end_step, self.config.num_inference_steps - 1),
        )
        fg_mask_latent = build_latent_masks(
            foreground_mask,
            latent_shape=scene_inv.start_latents.shape,
            device=self.device,
            dtype=self.dtype,
        )
        blender = LatentBlender(
            scene_trajectory=scene_reconstruction.latents_by_timestep,
            fg_mask_latent=fg_mask_latent,
            window=blend_window,
        )

        controller = MaskGuidedAttentionController(
            window=AttentionWindow(
                start_step=self.config.attention_inject_start_step,
                end_step=min(self.config.attention_inject_end_step, self.config.num_inference_steps - 1),
            )
        )
        controller.set_reference_mask(reference_mask_tensor)

        original_processors = None
        if self.config.attention_enabled:
            original_processors = install_mask_guided_processors(
                self.pipe,
                controller,
                target_prefixes=self.config.attention_target_prefixes,
            )

        try:
            if self.config.attention_enabled:
                controller.set_mode(controller.MODE_CAPTURE)
                run_denoise_trajectory(
                    pipe=self.pipe,
                    start_latents=reference_inv.start_latents,
                    prompt_embeds=reference_prompt_embeds,
                    guidance_scale=1.0,
                    num_inference_steps=self.config.num_inference_steps,
                    attn_controller=controller,
                )

                controller.set_mode(controller.MODE_INJECT)

            final_pass = run_denoise_trajectory(
                pipe=self.pipe,
                start_latents=scene_inv.start_latents,
                prompt_embeds=edit_prompt_embeds,
                guidance_scale=self.config.edit_guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                attn_controller=controller if self.config.attention_enabled else None,
                post_step_hook=blender,
            )
        finally:
            if original_processors is not None:
                restore_processors(self.pipe, original_processors)

        final_image = latents_to_image(self.pipe, final_pass.final_latents)

        final_path = output_path / "final.png"
        affordance_path = output_path / "affordance_pass.png"
        scene_recon_path = output_path / "scene_reconstruction.png"
        fg_mask_path = output_path / "foreground_mask.png"
        ref_mask_path = output_path / "reference_mask.png"

        final_image.save(final_path)
        affordance_image.save(affordance_path)
        scene_reconstruction_image.save(scene_recon_path)
        _mask_to_pil(foreground_mask).save(fg_mask_path)
        _mask_to_pil(reference_mask).save(ref_mask_path)

        return {
            "final": str(final_path),
            "affordance_pass": str(affordance_path),
            "scene_reconstruction": str(scene_recon_path),
            "foreground_mask": str(fg_mask_path),
            "reference_mask": str(ref_mask_path),
        }


def _resize_to_multiple_of_8(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_width = max(8, (width // 8) * 8)
    new_height = max(8, (height // 8) * 8)
    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _mask_to_pil(mask: np.ndarray) -> Image.Image:
    arr = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")
