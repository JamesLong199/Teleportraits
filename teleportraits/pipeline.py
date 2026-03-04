from __future__ import annotations

import inspect
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
from teleportraits.masks import load_binary_mask
from teleportraits.prompts import compose_edit_prompt
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

        load_kwargs = {"use_safetensors": True}
        signature = inspect.signature(StableDiffusionXLPipeline.from_pretrained)
        if "dtype" in signature.parameters:
            load_kwargs["dtype"] = self.dtype
        else:
            load_kwargs["torch_dtype"] = self.dtype

        self.pipe = StableDiffusionXLPipeline.from_pretrained(config.model_id, **load_kwargs)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        self.diff_mask_extractor = DifferenceMaskExtractor(
            threshold=config.mask_threshold,
            min_area_ratio=config.mask_min_area_ratio,
        )

        self.reference_mask_extractor: Optional[TransformersPersonMaskExtractor]
        if config.attention_enabled and config.use_transformers_reference_mask:
            self.reference_mask_extractor = TransformersPersonMaskExtractor()
        else:
            self.reference_mask_extractor = None

    @torch.no_grad()
    def run(
        self,
        scene_image_path: str,
        reference_image_path: str,
        reference_mask_path: Optional[str],
        reference_mask_invert: bool,
        scene_prompt: str,
        reference_prompt: str,
        edit_prompt: Optional[str],
        output_dir: str,
        person_placeholder: str = "a person",
    ) -> Dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cache_path = output_path / "_cache"
        cache_path.mkdir(parents=True, exist_ok=True)

        scene_input_path = output_path / "scene_input.png"
        reference_input_path = output_path / "reference_input.png"
        initial_path = output_path / "initial_pass.png"
        affordance_path = output_path / "affordance_pass.png"
        scene_recon_path = output_path / "scene_reconstruction.png"
        fg_mask_path = output_path / "foreground_mask.png"
        ref_mask_path = output_path / "reference_mask.png"
        final_path = output_path / "final.png"

        scene_inv_cache = cache_path / "scene_inv_start_latents.pt"
        reference_inv_cache = cache_path / "reference_inv_start_latents.pt"
        scene_recon_final_cache = cache_path / "scene_reconstruction_final_latents.pt"
        scene_recon_traj_cache = cache_path / "scene_reconstruction_trajectory.pt"
        initial_final_cache = cache_path / "initial_pass_final_latents.pt"
        fg_mask_cache = cache_path / "foreground_mask.npy"
        ref_mask_cache = cache_path / "reference_mask.npy"

        outputs: Dict[str, str] = {
            "scene_input": str(scene_input_path),
            "reference_input": str(reference_input_path),
            "initial_pass": str(initial_path),
            "affordance_pass": str(affordance_path),
            "scene_reconstruction": str(scene_recon_path),
            "foreground_mask": str(fg_mask_path),
            "reference_mask": str(ref_mask_path),
            "final": str(final_path),
        }

        _log_stage(self.config, "0/8 Loading inputs")
        scene_image = _resize_to_multiple_of_8(load_image(scene_image_path))
        reference_image = _resize_to_multiple_of_8(load_image(reference_image_path))
        scene_image.save(scene_input_path)
        reference_image.save(reference_input_path)

        _log_stage(self.config, "1/8 Encoding prompts")
        resolved_edit_prompt = compose_edit_prompt(
            scene_prompt=scene_prompt,
            reference_prompt=reference_prompt,
            explicit_edit_prompt=edit_prompt,
            placeholder=person_placeholder,
        )
        outputs["resolved_edit_prompt"] = resolved_edit_prompt

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
            prompt=resolved_edit_prompt,
            negative_prompt=self.config.negative_prompt,
            image_size=scene_image.size,
            device=self.device,
            do_cfg=True,
        )

        if scene_inv_cache.exists():
            _log_stage(self.config, "2/8 DDIM inversion: scene (resumed)")
            scene_inv_start = _load_tensor(scene_inv_cache, device=self.device, dtype=self.dtype)
        else:
            _log_stage(self.config, "2/8 DDIM inversion: scene")
            scene_latents = image_to_latents(self.pipe, scene_image, device=self.device, dtype=self.dtype)
            scene_inv = ddim_fixed_point_invert(
                pipe=self.pipe,
                clean_latents=scene_latents,
                prompt_embeds=scene_prompt_embeds,
                guidance_scale=self.config.inversion_guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                fixed_point_iters=self.config.inversion_fixed_point_iters,
                stage_name="Invert scene",
                show_progress_bar=self.config.show_progress_bar,
            )
            scene_inv_start = scene_inv.start_latents
            _save_tensor(scene_inv_cache, scene_inv_start)
        _ensure_finite(scene_inv_start, "scene_inv_start")

        if reference_inv_cache.exists():
            _log_stage(self.config, "3/8 DDIM inversion: reference (resumed)")
            reference_inv_start = _load_tensor(reference_inv_cache, device=self.device, dtype=self.dtype)
        else:
            _log_stage(self.config, "3/8 DDIM inversion: reference")
            reference_latents = image_to_latents(self.pipe, reference_image, device=self.device, dtype=self.dtype)
            reference_inv = ddim_fixed_point_invert(
                pipe=self.pipe,
                clean_latents=reference_latents,
                prompt_embeds=reference_prompt_embeds,
                guidance_scale=self.config.inversion_guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                fixed_point_iters=self.config.inversion_fixed_point_iters,
                stage_name="Invert reference",
                show_progress_bar=self.config.show_progress_bar,
            )
            reference_inv_start = reference_inv.start_latents
            _save_tensor(reference_inv_cache, reference_inv_start)
        _ensure_finite(reference_inv_start, "reference_inv_start")

        if scene_recon_path.exists() and scene_recon_final_cache.exists() and scene_recon_traj_cache.exists():
            _log_stage(self.config, "4/8 Scene reconstruction (resumed)")
            scene_reconstruction_image = load_image(str(scene_recon_path))
            scene_reconstruction_final = _load_tensor(scene_recon_final_cache, device=self.device, dtype=self.dtype)
            scene_reconstruction_traj = _load_tensor_dict(
                scene_recon_traj_cache, device=self.device, dtype=self.dtype
            )
        else:
            _log_stage(self.config, "4/8 Scene reconstruction")
            scene_reconstruction = run_denoise_trajectory(
                pipe=self.pipe,
                start_latents=scene_inv_start,
                prompt_embeds=scene_prompt_embeds,
                guidance_scale=1.0,
                num_inference_steps=self.config.num_inference_steps,
                stage_name="Reconstruct scene",
                show_progress_bar=self.config.show_progress_bar,
            )
            scene_reconstruction_final = scene_reconstruction.final_latents
            scene_reconstruction_traj = scene_reconstruction.latents_by_timestep
            scene_reconstruction_image = latents_to_image(self.pipe, scene_reconstruction_final)
            scene_reconstruction_image.save(scene_recon_path)
            _save_tensor(scene_recon_final_cache, scene_reconstruction_final)
            _save_tensor_dict(scene_recon_traj_cache, scene_reconstruction_traj)
        _ensure_finite(scene_reconstruction_final, "scene_reconstruction_final")

        if initial_path.exists() and initial_final_cache.exists():
            _log_stage(self.config, "5/8 Initial human generation pass (resumed)")
            initial_pass_image = load_image(str(initial_path))
            initial_final = _load_tensor(initial_final_cache, device=self.device, dtype=self.dtype)
        else:
            _log_stage(self.config, "5/8 Initial human generation pass")
            initial_pass = run_denoise_trajectory(
                pipe=self.pipe,
                start_latents=scene_inv_start,
                prompt_embeds=edit_prompt_embeds,
                guidance_scale=self.config.edit_guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                stage_name="Initial human pass",
                show_progress_bar=self.config.show_progress_bar,
            )
            initial_final = initial_pass.final_latents
            initial_pass_image = latents_to_image(self.pipe, initial_final)
            initial_pass_image.save(initial_path)
            # Backward-compatibility alias for previous filename.
            initial_pass_image.save(affordance_path)
            _save_tensor(initial_final_cache, initial_final)
        if not affordance_path.exists():
            initial_pass_image.save(affordance_path)
        _ensure_finite(initial_final, "initial_final")

        if fg_mask_path.exists() and fg_mask_cache.exists():
            _log_stage(self.config, "6/8 Foreground mask extraction (resumed)")
            foreground_mask = np.load(fg_mask_cache).astype(np.float32)
        else:
            _log_stage(self.config, "6/8 Foreground mask extraction from initial pass")
            foreground_mask = self.diff_mask_extractor.extract(initial_pass_image, scene_reconstruction_image)
            _mask_to_pil(foreground_mask).save(fg_mask_path)
            np.save(fg_mask_cache, foreground_mask.astype(np.float32))

        if reference_mask_path and Path(reference_mask_path).exists():
            _log_stage(self.config, "7/8 Loading reference mask override")
            reference_mask = load_binary_mask(
                path=reference_mask_path,
                target_size=reference_image.size,
                invert=reference_mask_invert,
            )
            _mask_to_pil(reference_mask).save(ref_mask_path)
            np.save(ref_mask_cache, reference_mask.astype(np.float32))
        elif ref_mask_path.exists() and ref_mask_cache.exists():
            _log_stage(self.config, "7/8 Reference mask extraction (resumed)")
            reference_mask = np.load(ref_mask_cache).astype(np.float32)
        else:
            _log_stage(self.config, "7/8 Reference mask extraction")
            reference_mask = reference_person_mask(reference_image, self.reference_mask_extractor)
            _mask_to_pil(reference_mask).save(ref_mask_path)
            np.save(ref_mask_cache, reference_mask.astype(np.float32))

        reference_mask_tensor = torch.from_numpy(reference_mask).to(device=self.device, dtype=self.dtype)
        fg_mask_latent = build_latent_masks(
            foreground_mask,
            latent_shape=scene_inv_start.shape,
            device=self.device,
            dtype=self.dtype,
        )

        if final_path.exists():
            _log_stage(self.config, "8/8 Final output already exists (resumed)")
            return outputs

        blend_window = BlendWindow(
            start_step=self.config.blend_start_step,
            end_step=min(self.config.blend_end_step, self.config.num_inference_steps - 1),
        )
        blender = LatentBlender(
            scene_trajectory=scene_reconstruction_traj,
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
                _log_stage(self.config, "8/8 Reference attention capture")
                controller.set_mode(controller.MODE_CAPTURE)
                run_denoise_trajectory(
                    pipe=self.pipe,
                    start_latents=reference_inv_start,
                    prompt_embeds=reference_prompt_embeds,
                    guidance_scale=1.0,
                    num_inference_steps=self.config.num_inference_steps,
                    attn_controller=controller,
                    stage_name="Capture reference K/V",
                    show_progress_bar=self.config.show_progress_bar,
                )
                controller.set_mode(controller.MODE_INJECT)

            _log_stage(self.config, "8/8 Final pass with latent blending")
            final_pass = run_denoise_trajectory(
                pipe=self.pipe,
                start_latents=scene_inv_start,
                prompt_embeds=edit_prompt_embeds,
                guidance_scale=self.config.edit_guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                attn_controller=controller if self.config.attention_enabled else None,
                post_step_hook=blender,
                stage_name="Final blended pass",
                show_progress_bar=self.config.show_progress_bar,
            )
            _ensure_finite(final_pass.final_latents, "final_pass.final_latents")
        finally:
            if original_processors is not None:
                restore_processors(self.pipe, original_processors)

        final_image = latents_to_image(self.pipe, final_pass.final_latents)
        final_image.save(final_path)
        return outputs


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


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    torch.save(tensor.detach().cpu(), path)


def _load_tensor(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected tensor in {path}, got {type(tensor)}")
    return tensor.to(device=device, dtype=dtype)


def _save_tensor_dict(path: Path, tensor_dict: Dict[int, torch.Tensor]) -> None:
    serializable = {int(k): v.detach().cpu() for k, v in tensor_dict.items()}
    torch.save(serializable, path)


def _load_tensor_dict(path: Path, device: torch.device, dtype: torch.dtype) -> Dict[int, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {path}, got {type(obj)}")
    out: Dict[int, torch.Tensor] = {}
    for k, v in obj.items():
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"Expected tensor values in {path}, got {type(v)} at key {k}")
        out[int(k)] = v.to(device=device, dtype=dtype)
    return out


def _log_stage(config: TeleportraitConfig, message: str) -> None:
    if config.verbose:
        print(f"[Teleportraits] {message}", flush=True)


def _ensure_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"Found non-finite values in {name}. Try --torch-dtype float32 and lower --inversion-fixed-point-iters."
        )
