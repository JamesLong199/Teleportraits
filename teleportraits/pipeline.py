from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
import inspect
from pathlib import Path
import shutil
from typing import Any, Dict, Optional

import numpy as np
import torch
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionXLPipeline
from PIL import Image

from teleportraits.attention import (
    AttentionWindow,
    MaskGuidedAttentionController,
    install_mask_guided_processors,
    restore_processors,
)
from teleportraits.blending import BlendWindow, LatentBlender, build_latent_masks
from teleportraits.config import TeleportraitConfig
from teleportraits.depth import MogeDepthMapExtractor, depth_to_control_image
from teleportraits.inversion import ddim_fixed_point_invert
from teleportraits.masks import load_binary_mask
from teleportraits.pose import OpenposeMapExtractor
from teleportraits.sampler import run_denoise_trajectory
from teleportraits.sdxl_utils import (
    encode_prompt_sdxl,
    image_to_latents,
    latents_to_image,
    load_image,
    parse_dtype,
)
from teleportraits.segmentation import (
    Sam3ForegroundMaskExtractor,
    TransformersPersonMaskExtractor,
    reference_person_mask,
)
from teleportraits.types import PromptEmbeds


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

        self.pipe = self._load_diffusers_component(
            StableDiffusionXLPipeline,
            config.model_id,
            load_kwargs,
            "base pipeline",
        )
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            prediction_type="epsilon",
            timestep_spacing="trailing",
            set_alpha_to_one=True,
            steps_offset=0,
        )
        self.pipe = self.pipe.to(self.device)

        self.affordance_controlnet: Optional[ControlNetModel] = None
        self.affordance_openpose_controlnet: Optional[ControlNetModel] = None
        if config.affordance_use_controlnet_depth:
            self.affordance_controlnet = self._load_controlnet(
                model_id=config.affordance_controlnet_model_id,
                model_dir=config.affordance_controlnet_dir,
                model_label="depth",
            )
        if (
            config.affordance_use_controlnet_pose
            or config.affordance_refine_use_controlnet_pose
            or config.final_use_controlnet_pose
        ):
            self.affordance_openpose_controlnet = self._load_controlnet(
                model_id=config.affordance_openpose_controlnet_model_id,
                model_dir=config.affordance_openpose_controlnet_dir,
                model_label="openpose",
            )

        self.depth_extractor = MogeDepthMapExtractor(
            pretrained_model_name_or_path=config.moge_pretrained_model,
            checkpoint_dir=config.moge_checkpoint_dir,
            model_version=config.moge_model_version,
            device=config.device,
            use_fp16=config.moge_use_fp16,
            conda_env=config.moge_conda_env,
        )
        self.openpose_extractor: Optional[OpenposeMapExtractor] = None
        if (
            config.affordance_use_controlnet_pose
            or config.affordance_refine_use_controlnet_pose
            or config.final_use_controlnet_pose
        ):
            self.openpose_extractor = OpenposeMapExtractor(
                pretrained_model_name_or_path=config.openpose_detector_model_id,
                model_dir=config.openpose_detector_dir,
                device=config.device,
            )

        self.foreground_mask_extractor = Sam3ForegroundMaskExtractor(
            prompt=config.foreground_mask_prompt,
            confidence_threshold=config.foreground_mask_confidence_threshold,
            min_area_ratio=config.mask_min_area_ratio,
            device=config.device,
            checkpoint_dir=config.sam3_checkpoint_dir,
            conda_env=config.sam3_conda_env,
        )

        self.reference_mask_extractor: Optional[TransformersPersonMaskExtractor]
        if config.attention_enabled and config.use_transformers_reference_mask:
            self.reference_mask_extractor = TransformersPersonMaskExtractor()
        else:
            self.reference_mask_extractor = None

    def _load_controlnet(self, model_id: str, model_dir: str, model_label: str) -> ControlNetModel:
        controlnet_source = model_id
        controlnet_dir = Path(model_dir).expanduser()
        if model_dir.strip() and controlnet_dir.exists():
            controlnet_source = str(controlnet_dir.resolve())
        elif model_dir.strip() and self.config.verbose:
            _log_stage(
                self.config,
                f"ControlNet {model_label} dir not found ({model_dir}); falling back to model id: {model_id}",
            )

        controlnet_kwargs = {"use_safetensors": True}
        controlnet_signature = inspect.signature(ControlNetModel.from_pretrained)
        if "dtype" in controlnet_signature.parameters:
            controlnet_kwargs["dtype"] = self.dtype
        else:
            controlnet_kwargs["torch_dtype"] = self.dtype
        return self._load_diffusers_component(
            ControlNetModel,
            controlnet_source,
            controlnet_kwargs,
            f"{model_label} controlnet",
        ).to(self.device)

    def _load_diffusers_component(
        self,
        loader_cls: Any,
        source: str,
        load_kwargs: Dict[str, Any],
        component_label: str,
    ) -> Any:
        try:
            return loader_cls.from_pretrained(source, **load_kwargs)
        except OSError as exc:
            if not self._should_retry_without_safetensors(source, load_kwargs, exc):
                raise

            retry_kwargs = dict(load_kwargs)
            retry_kwargs["use_safetensors"] = False
            if self.config.verbose:
                _log_stage(
                    self.config,
                    f"Falling back to .bin weights for {component_label}: {source}",
                )
            return loader_cls.from_pretrained(source, **retry_kwargs)

    @staticmethod
    def _should_retry_without_safetensors(
        source: str,
        load_kwargs: Dict[str, Any],
        error: OSError,
    ) -> bool:
        if not load_kwargs.get("use_safetensors", False):
            return False

        source_path = Path(source).expanduser()
        if not source_path.exists():
            return False

        message = str(error)
        return "safetensors" in message and "no file named" in message

    @torch.no_grad()
    def run(
        self,
        scene_image_path: str,
        reference_image_path: str,
        foreground_mask_path: Optional[str],
        reference_mask_path: Optional[str],
        reference_mask_invert: bool,
        affordance_prompt: str,
        final_scene_prompt: str,
        final_prompt: str,
        reference_prompt: str,
        output_dir: str,
        affordance_refine_prompt: Optional[str] = None,
        inversion_reference_prompt: Optional[str] = None,
        input_json_path: Optional[str] = None,
    ) -> Dict[str, str]:
        base_output_dir = Path(output_dir)
        foreground_mask_override = bool(foreground_mask_path and Path(foreground_mask_path).exists())
        output_path = _resolve_run_output_dir(
            base_output_dir,
            self.config,
            use_child_run=foreground_mask_override and base_output_dir.name.startswith("exp_"),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        cache_path = output_path / "_cache"
        cache_path.mkdir(parents=True, exist_ok=True)
        parent_cache_path: Optional[Path] = None
        if foreground_mask_override and base_output_dir.name.startswith("exp_"):
            candidate_parent_cache = base_output_dir / "_cache"
            if candidate_parent_cache.exists():
                parent_cache_path = candidate_parent_cache
                _log_stage(self.config, f"Using parent cache fallback from: {candidate_parent_cache}")

        scene_input_path = output_path / "scene_input.png"
        reference_input_path = output_path / "reference_input.png"
        initial_path = output_path / "initial_pass.png"
        affordance_path = output_path / "affordance_pass.png"
        affordance_refine_path = output_path / "affordance_refine_pass.png"
        affordance_pose_path = output_path / "affordance_pose.png"
        scene_depth_path = output_path / "scene_depth.png"
        affordance_controlnet_mask_path = output_path / "affordance_controlnet_mask.png"
        scene_recon_path = output_path / "scene_reconstruction.png"
        fg_mask_path = output_path / "foreground_mask.png"
        ref_mask_path = output_path / "reference_mask.png"
        final_path = output_path / "final.png"
        run_config_path = output_path / "run_config.json"

        scene_inv_cache = _resolve_cache_file(cache_path, parent_cache_path, "scene_inv_start_latents.pt")
        scene_random_cache = _resolve_cache_file(cache_path, parent_cache_path, "scene_random_start_latents.pt")
        reference_inv_cache = _resolve_cache_file(cache_path, parent_cache_path, "reference_inv_start_latents.pt")
        scene_recon_final_cache = _resolve_cache_file(cache_path, parent_cache_path, "scene_reconstruction_final_latents.pt")
        scene_recon_traj_cache = _resolve_cache_file(cache_path, parent_cache_path, "scene_reconstruction_trajectory.pt")
        initial_final_cache = _resolve_cache_file(cache_path, parent_cache_path, "initial_pass_final_latents.pt")
        affordance_final_cache = _resolve_cache_file(cache_path, parent_cache_path, "affordance_pass_final_latents.pt")
        fg_mask_cache = _resolve_cache_file(cache_path, parent_cache_path, "foreground_mask.npy")
        ref_mask_cache = _resolve_cache_file(cache_path, parent_cache_path, "reference_mask.npy")

        outputs: Dict[str, str] = {
            "run_dir": str(output_path),
            "scene_input": str(scene_input_path),
            "reference_input": str(reference_input_path),
            "initial_pass": str(initial_path),
            "affordance_pass": str(affordance_path),
            "affordance_refine_pass": str(affordance_refine_path),
            "affordance_pose": str(affordance_pose_path),
            "scene_depth": str(scene_depth_path),
            "affordance_controlnet_mask": str(affordance_controlnet_mask_path),
            "scene_reconstruction": str(scene_recon_path),
            "foreground_mask": str(fg_mask_path),
            "reference_mask": str(ref_mask_path),
            "final": str(final_path),
            "run_config": str(run_config_path),
        }
        outputs["foreground_mask_prompt"] = self.config.foreground_mask_prompt
        outputs["sam3_checkpoint_dir"] = self.config.sam3_checkpoint_dir or ""
        outputs["sam3_conda_env"] = self.config.sam3_conda_env
        if foreground_mask_override and foreground_mask_path is not None:
            outputs["user_foreground_mask_path"] = foreground_mask_path
        if input_json_path is not None:
            input_json_src = Path(input_json_path).expanduser()
            if not input_json_src.exists():
                raise FileNotFoundError(f"input_json_path does not exist: {input_json_path}")
            copied_input_path = output_path / "input_config.json"
            shutil.copy2(input_json_src, copied_input_path)
            outputs["input_config"] = str(copied_input_path)

        _log_stage(self.config, "0/8 Loading inputs")
        scene_image = _resize_for_model(load_image(scene_image_path), target_size=self.config.image_size)
        reference_image = _resize_for_model(load_image(reference_image_path), target_size=self.config.image_size)
        scene_image.save(scene_input_path)
        reference_image.save(reference_input_path)

        _log_stage(self.config, "1/8 Encoding prompts")
        resolved_final_prompt = final_prompt.strip()
        if not resolved_final_prompt:
            raise ValueError("final_prompt must not be empty.")
        resolved_scene_inversion_prompt = (
            self.config.inversion_prompt.strip()
            if self.config.inversion_prompt is not None and self.config.inversion_prompt.strip()
            else affordance_prompt.strip()
        )
        resolved_initial_prompt = affordance_prompt.strip()
        resolved_affordance_refine_prompt = (
            affordance_refine_prompt.strip()
            if affordance_refine_prompt is not None and affordance_refine_prompt.strip()
            else resolved_initial_prompt
        )
        resolved_reference_inversion_prompt = (
            inversion_reference_prompt.strip()
            if inversion_reference_prompt is not None and inversion_reference_prompt.strip()
            else reference_prompt.strip()
        )
        resolved_negative_prompt = self.config.negative_prompt.strip()
        if resolved_negative_prompt:
            _log_stage(self.config, f'Negative prompt enabled: "{resolved_negative_prompt}"')
        else:
            _log_stage(self.config, "Negative prompt disabled (empty).")
        inversion_do_cfg = self.config.inversion_guidance_scale != 1.0
        outputs["resolved_final_prompt"] = resolved_final_prompt
        outputs["resolved_edit_prompt"] = resolved_final_prompt
        outputs["resolved_initial_prompt"] = resolved_initial_prompt
        outputs["resolved_affordance_refine_prompt"] = resolved_affordance_refine_prompt
        outputs["resolved_scene_inversion_prompt"] = resolved_scene_inversion_prompt
        outputs["resolved_reference_inversion_prompt"] = resolved_reference_inversion_prompt
        outputs["resolved_negative_prompt"] = resolved_negative_prompt
        run_config: Dict[str, Any] = {
            "inputs": {
                "scene_image_path": scene_image_path,
                "reference_image_path": reference_image_path,
                "foreground_mask_path": foreground_mask_path,
                "reference_mask_path": reference_mask_path,
                "reference_mask_invert": reference_mask_invert,
                "affordance_prompt": affordance_prompt,
                "final_scene_prompt": final_scene_prompt,
                "final_prompt": final_prompt,
                "reference_prompt": reference_prompt,
                "affordance_refine_prompt": affordance_refine_prompt,
                "inversion_reference_prompt": inversion_reference_prompt,
                "output_dir": output_dir,
                "input_json_path": input_json_path,
            },
            "resolved_inputs": {
                "scene_inversion_prompt": resolved_scene_inversion_prompt,
                "reference_inversion_prompt": resolved_reference_inversion_prompt,
                "initial_prompt": resolved_initial_prompt,
                "affordance_refine_prompt": resolved_affordance_refine_prompt,
                "final_prompt": resolved_final_prompt,
                "negative_prompt": resolved_negative_prompt,
                "scene_image_size": list(scene_image.size),
                "reference_image_size": list(reference_image.size),
            },
            "hyperparameters": asdict(self.config),
        }
        _save_json(run_config_path, run_config)

        scene_prompt_embeds_inv = encode_prompt_sdxl(
            self.pipe,
            prompt=resolved_scene_inversion_prompt,
            negative_prompt=resolved_negative_prompt,
            image_size=scene_image.size,
            device=self.device,
            do_cfg=inversion_do_cfg,
        )
        reference_prompt_embeds_inv: Optional[PromptEmbeds] = None
        reference_prompt_embeds_attn: Optional[PromptEmbeds] = None
        if not self.config.affordance_only:
            reference_prompt_embeds_inv = encode_prompt_sdxl(
                self.pipe,
                prompt=resolved_reference_inversion_prompt,
                negative_prompt=resolved_negative_prompt,
                image_size=reference_image.size,
                device=self.device,
                do_cfg=inversion_do_cfg,
            )
            if self.config.attention_enabled:
                # Attention capture needs CFG batch split (uncond/cond) so conditional K/V can be stored.
                reference_prompt_embeds_attn = encode_prompt_sdxl(
                    self.pipe,
                    prompt=reference_prompt,
                    negative_prompt=resolved_negative_prompt,
                    image_size=reference_image.size,
                    device=self.device,
                    do_cfg=True,
                )
        initial_prompt_embeds = encode_prompt_sdxl(
            self.pipe,
            prompt=resolved_initial_prompt,
            negative_prompt=resolved_negative_prompt,
            image_size=scene_image.size,
            device=self.device,
            do_cfg=True,
        )
        final_prompt_embeds = encode_prompt_sdxl(
            self.pipe,
            prompt=resolved_final_prompt,
            negative_prompt=resolved_negative_prompt,
            image_size=scene_image.size,
            device=self.device,
            do_cfg=True,
        )
        affordance_refine_prompt_embeds = encode_prompt_sdxl(
            self.pipe,
            prompt=resolved_affordance_refine_prompt,
            negative_prompt=resolved_negative_prompt,
            image_size=scene_image.size,
            device=self.device,
            do_cfg=True,
        )
        _log_inversion_config(
            config=self.config,
            scheduler=self.pipe.scheduler,
            scene_image_size=scene_image.size,
            reference_image_size=reference_image.size,
            inversion_do_cfg=inversion_do_cfg,
            scene_inversion_prompt=resolved_scene_inversion_prompt,
            reference_inversion_prompt=resolved_reference_inversion_prompt,
        )

        if self.config.random_start_latent:
            if scene_random_cache.exists():
                _log_stage(self.config, "2/8 Scene inversion: skipped (random latent, resumed)")
                scene_inv_start = _load_tensor(scene_random_cache, device=self.device, dtype=self.dtype)
            else:
                _log_stage(self.config, "2/8 Scene inversion: skipped (random latent start)")
                latent_h = scene_image.size[1] // self.pipe.vae_scale_factor
                latent_w = scene_image.size[0] // self.pipe.vae_scale_factor
                latent_channels = int(getattr(self.pipe.unet.config, "in_channels", 4))
                generator = torch.Generator(device=self.device).manual_seed(int(self.config.seed))
                scene_inv_start = torch.randn(
                    (1, latent_channels, latent_h, latent_w),
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                )
                _save_tensor(scene_random_cache, scene_inv_start)
        else:
            if scene_inv_cache.exists():
                _log_stage(self.config, "2/8 DDIM inversion: scene (resumed)")
                scene_inv_start = _load_tensor(scene_inv_cache, device=self.device, dtype=self.dtype)
            else:
                _log_stage(self.config, "2/8 DDIM inversion: scene")
                scene_latents = image_to_latents(self.pipe, scene_image, device=self.device, dtype=self.dtype)
                scene_inv = ddim_fixed_point_invert(
                    pipe=self.pipe,
                    clean_latents=scene_latents,
                    prompt_embeds=scene_prompt_embeds_inv,
                    guidance_scale=self.config.inversion_guidance_scale,
                    num_inference_steps=self.config.num_inference_steps,
                    fixed_point_iters=self.config.inversion_fixed_point_iters,
                    stage_name="Invert scene",
                    show_progress_bar=self.config.show_progress_bar,
                )
                scene_inv_start = scene_inv.start_latents
                _save_tensor(scene_inv_cache, scene_inv_start)
        _ensure_finite(scene_inv_start, "scene_inv_start")

        reference_inv_start: Optional[torch.Tensor] = None
        if self.config.affordance_only:
            _log_stage(self.config, "3/8 DDIM inversion: reference (skipped: affordance-only mode)")
        else:
            if reference_inv_cache.exists():
                _log_stage(self.config, "3/8 DDIM inversion: reference (resumed)")
                reference_inv_start = _load_tensor(reference_inv_cache, device=self.device, dtype=self.dtype)
            else:
                _log_stage(self.config, "3/8 DDIM inversion: reference")
                reference_latents = image_to_latents(self.pipe, reference_image, device=self.device, dtype=self.dtype)
                reference_inv = ddim_fixed_point_invert(
                    pipe=self.pipe,
                    clean_latents=reference_latents,
                    prompt_embeds=reference_prompt_embeds_inv,
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
                prompt_embeds=scene_prompt_embeds_inv,
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

        affordance_controlnet_start_step = self.config.affordance_controlnet_start_step
        affordance_controlnet_end_step = min(
            self.config.affordance_controlnet_end_step,
            self.config.num_inference_steps - 1,
        )
        affordance_controlnet_mask_start_step = self.config.affordance_controlnet_mask_start_step
        affordance_controlnet_mask_end_step = min(
            self.config.affordance_controlnet_mask_end_step,
            self.config.num_inference_steps - 1,
        )
        affordance_refine_controlnet_start_step = self.config.affordance_refine_controlnet_start_step
        affordance_refine_controlnet_end_step = min(
            self.config.affordance_refine_controlnet_end_step,
            self.config.num_inference_steps - 1,
        )
        final_controlnet_start_step = self.config.final_controlnet_start_step
        final_controlnet_end_step = min(
            self.config.final_controlnet_end_step,
            self.config.num_inference_steps - 1,
        )
        use_affordance_depth_controlnet = self.config.affordance_use_controlnet_depth
        use_affordance_pose_controlnet = self.config.affordance_use_controlnet_pose
        use_affordance_refine_depth_controlnet = self.config.affordance_refine_use_controlnet_depth
        use_affordance_refine_pose_controlnet = self.config.affordance_refine_use_controlnet_pose
        use_final_depth_controlnet = self.config.final_use_controlnet_depth
        use_final_pose_controlnet = self.config.final_use_controlnet_pose

        initial_pass_image: Optional[Image.Image] = None
        initial_final: Optional[torch.Tensor] = None
        affordance_pass_image: Optional[Image.Image] = None
        affordance_final: Optional[torch.Tensor] = None
        depth_control_image_tensor: Optional[torch.Tensor] = None
        affordance_pose_control_image_tensor: Optional[torch.Tensor] = None
        openpose_control_image_tensor: Optional[torch.Tensor] = None
        affordance_controlnet_residual_suppress_mask: Optional[torch.Tensor] = None
        if foreground_mask_override and (use_affordance_refine_pose_controlnet or use_final_pose_controlnet):
            raise ValueError(
                "Pose-guided refine/final requires generating an initial affordance pass; "
                "it cannot be combined with --foreground-mask-image override mode."
            )

        require_scene_depth_control = (
            use_affordance_depth_controlnet or use_affordance_refine_depth_controlnet or use_final_depth_controlnet
        )
        if require_scene_depth_control:
            if self.affordance_controlnet is None:
                raise RuntimeError("ControlNet Depth is enabled but model was not initialized.")
            if scene_depth_path.exists():
                _log_stage(self.config, "5/8 Loading scene depth map for affordance control (resumed)")
                scene_depth_image = load_image(str(scene_depth_path)).convert("RGB")
            else:
                _log_stage(self.config, "5/8 Generating scene depth map with MoGe for affordance control")
                scene_depth_norm = self.depth_extractor.extract(scene_image)
                scene_depth_image = depth_to_control_image(scene_depth_norm)
                scene_depth_image.save(scene_depth_path)
            if scene_depth_image.size != scene_image.size:
                scene_depth_image = scene_depth_image.resize(scene_image.size, Image.Resampling.BICUBIC)
            depth_control_image_tensor = _control_image_to_tensor(
                scene_depth_image,
                device=self.device,
                dtype=self.dtype,
            )

            if self.config.affordance_controlnet_mask_image is not None:
                mask_path = Path(self.config.affordance_controlnet_mask_image)
                if not mask_path.exists():
                    raise FileNotFoundError(
                        "affordance_controlnet_mask_image does not exist: "
                        f"{self.config.affordance_controlnet_mask_image}"
                    )
                _log_stage(self.config, "5/8 Loading affordance ControlNet residual suppression mask")
                suppress_mask_np = load_binary_mask(
                    path=str(mask_path),
                    target_size=scene_image.size,
                    invert=self.config.affordance_controlnet_mask_invert,
                )
                _mask_to_pil(suppress_mask_np).save(affordance_controlnet_mask_path)
                affordance_controlnet_residual_suppress_mask = _mask_to_tensor(
                    suppress_mask_np,
                    device=self.device,
                    dtype=self.dtype,
                )

            if affordance_controlnet_residual_suppress_mask is not None and not (
                affordance_controlnet_start_step
                <= affordance_controlnet_mask_start_step
                <= affordance_controlnet_mask_end_step
                <= affordance_controlnet_end_step
            ):
                raise ValueError(
                    "affordance_controlnet_mask_range must stay within affordance_controlnet_range. "
                    f"Got mask range {affordance_controlnet_mask_start_step}:{affordance_controlnet_mask_end_step} and "
                    f"controlnet range {affordance_controlnet_start_step}:{affordance_controlnet_end_step}."
                )

        if use_affordance_pose_controlnet:
            if self.affordance_openpose_controlnet is None:
                raise RuntimeError("ControlNet OpenPose is enabled but model was not initialized.")
            if not self.config.affordance_pose_image_path:
                raise ValueError(
                    "affordance_pose_image_path must be provided when affordance pose controlnet is enabled."
                )
            pose_path = Path(self.config.affordance_pose_image_path).expanduser()
            if not pose_path.exists():
                raise FileNotFoundError(
                    f"affordance pose control image does not exist: {self.config.affordance_pose_image_path}"
                )
            affordance_pose_img = load_image(str(pose_path)).convert("RGB")
            if affordance_pose_img.size != scene_image.size:
                affordance_pose_img = affordance_pose_img.resize(scene_image.size, Image.Resampling.BICUBIC)
            affordance_pose_control_image_tensor = _control_image_to_tensor(
                affordance_pose_img,
                device=self.device,
                dtype=self.dtype,
            )

        if foreground_mask_override:
            _log_stage(self.config, "5/8 Initial human generation pass (skipped: user foreground mask provided)")
            outputs["pipeline_mode"] = "user_foreground_mask"
            outputs["initial_pass_skipped"] = "true"
            outputs["affordance_pass_skipped"] = "true"
            # Remove stale artifacts in case this run dir previously contained a full run.
            if initial_path.exists():
                initial_path.unlink()
            if affordance_path.exists():
                affordance_path.unlink()
            if affordance_refine_path.exists():
                affordance_refine_path.unlink()
            if affordance_pose_path.exists():
                affordance_pose_path.unlink()
        elif initial_path.exists() and initial_final_cache.exists():
            _log_stage(self.config, "5/8 Initial human generation pass (resumed)")
            initial_pass_image = load_image(str(initial_path))
            initial_final = _load_tensor(initial_final_cache, device=self.device, dtype=self.dtype)
        else:
            _log_stage(self.config, "5/8 Initial human generation pass")
            affordance_controlnet, affordance_control_image, affordance_control_scale = _compose_controlnet_inputs(
                depth_controlnet=self.affordance_controlnet if use_affordance_depth_controlnet else None,
                depth_control_image=depth_control_image_tensor,
                depth_scale=self.config.affordance_controlnet_scale,
                openpose_controlnet=self.affordance_openpose_controlnet if use_affordance_pose_controlnet else None,
                openpose_control_image=affordance_pose_control_image_tensor,
                openpose_scale=self.config.affordance_pose_controlnet_scale,
            )
            initial_pass = run_denoise_trajectory(
                pipe=self.pipe,
                start_latents=scene_inv_start,
                prompt_embeds=initial_prompt_embeds,
                guidance_scale=self.config.affordance_guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                controlnet=affordance_controlnet,
                control_image=affordance_control_image,
                controlnet_conditioning_scale=affordance_control_scale,
                controlnet_inject_start_step=affordance_controlnet_start_step,
                controlnet_inject_end_step=affordance_controlnet_end_step,
                controlnet_residual_suppress_mask=affordance_controlnet_residual_suppress_mask,
                controlnet_mask_inject_start_step=affordance_controlnet_mask_start_step,
                controlnet_mask_inject_end_step=affordance_controlnet_mask_end_step,
                stage_name="Initial human pass",
                show_progress_bar=self.config.show_progress_bar,
            )
            initial_final = initial_pass.final_latents
            initial_pass_image = latents_to_image(self.pipe, initial_final)
            initial_pass_image.save(initial_path)
            _save_tensor(initial_final_cache, initial_final)
        if initial_final is not None:
            _ensure_finite(initial_final, "initial_final")
        affordance_pass_image = initial_pass_image
        affordance_final = initial_final
        if initial_pass_image is not None:
            # Keep initial affordance pass artifact stable for debugging, even if a refine pass runs later.
            initial_pass_image.save(affordance_path)

        need_generated_pose_map = use_affordance_refine_pose_controlnet or use_final_pose_controlnet
        if need_generated_pose_map:
            if self.affordance_openpose_controlnet is None or self.openpose_extractor is None:
                raise RuntimeError("OpenPose ControlNet is enabled but model/extractor was not initialized.")
            if affordance_pose_path.exists():
                pose_image = load_image(str(affordance_pose_path)).convert("RGB")
                if pose_image.size != scene_image.size:
                    pose_image = pose_image.resize(scene_image.size, Image.Resampling.BICUBIC)
                    pose_image.save(affordance_pose_path)
                openpose_control_image_tensor = _control_image_to_tensor(
                    pose_image,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                if initial_pass_image is None:
                    raise RuntimeError("Initial affordance image is required for OpenPose extraction.")
                _log_stage(self.config, "5/8 Extracting pose from initial affordance image")
                pose_image = self.openpose_extractor.extract(initial_pass_image, target_size=scene_image.size)
                pose_image.save(affordance_pose_path)
                openpose_control_image_tensor = _control_image_to_tensor(
                    pose_image,
                    device=self.device,
                    dtype=self.dtype,
                )

        run_affordance_refine = use_affordance_refine_depth_controlnet or use_affordance_refine_pose_controlnet
        if run_affordance_refine:
            if affordance_refine_path.exists() and affordance_final_cache.exists():
                _log_stage(self.config, "5/8 Second affordance generation pass (refine, resumed)")
                affordance_pass_image = load_image(str(affordance_refine_path))
                affordance_final = _load_tensor(affordance_final_cache, device=self.device, dtype=self.dtype)
            else:
                second_controlnet, second_control_image, second_control_scale = _compose_controlnet_inputs(
                    depth_controlnet=self.affordance_controlnet if use_affordance_refine_depth_controlnet else None,
                    depth_control_image=depth_control_image_tensor,
                    depth_scale=self.config.affordance_refine_depth_controlnet_scale,
                    openpose_controlnet=self.affordance_openpose_controlnet if use_affordance_refine_pose_controlnet else None,
                    openpose_control_image=openpose_control_image_tensor,
                    openpose_scale=self.config.affordance_refine_pose_controlnet_scale,
                )
                _log_stage(self.config, "5/8 Second affordance generation pass (refine)")
                second_affordance = run_denoise_trajectory(
                    pipe=self.pipe,
                    start_latents=scene_inv_start,
                    prompt_embeds=affordance_refine_prompt_embeds,
                    guidance_scale=self.config.affordance_refine_guidance_scale,
                    num_inference_steps=self.config.num_inference_steps,
                    controlnet=second_controlnet,
                    control_image=second_control_image,
                    controlnet_conditioning_scale=second_control_scale,
                    controlnet_inject_start_step=affordance_refine_controlnet_start_step,
                    controlnet_inject_end_step=affordance_refine_controlnet_end_step,
                    controlnet_residual_suppress_mask=affordance_controlnet_residual_suppress_mask,
                    controlnet_mask_inject_start_step=affordance_controlnet_mask_start_step,
                    controlnet_mask_inject_end_step=affordance_controlnet_mask_end_step,
                    stage_name="Affordance refine pass",
                    show_progress_bar=self.config.show_progress_bar,
                )
                affordance_final = second_affordance.final_latents
                _ensure_finite(affordance_final, "affordance_final")
                affordance_pass_image = latents_to_image(self.pipe, affordance_final)
                affordance_pass_image.save(affordance_refine_path)
                _save_tensor(affordance_final_cache, affordance_final)
        elif affordance_refine_path.exists():
            affordance_refine_path.unlink()
        if affordance_final is not None:
            _ensure_finite(affordance_final, "affordance_final")

        if self.config.affordance_only:
            _log_stage(
                self.config,
                "6/8-8/8 Skipped: affordance-only mode (stopped after scene inversion/reconstruction/initial pass)",
            )
            outputs["pipeline_mode"] = "affordance_only"
            return outputs

        if foreground_mask_override and foreground_mask_path is not None:
            _log_stage(self.config, "6/8 Loading foreground mask override")
            foreground_mask = load_binary_mask(
                path=foreground_mask_path,
                target_size=scene_image.size,
                invert=False,
            )
            _mask_to_pil(foreground_mask).save(fg_mask_path)
            np.save(fg_mask_cache, foreground_mask.astype(np.float32))
        elif fg_mask_path.exists() and fg_mask_cache.exists():
            _log_stage(self.config, "6/8 Foreground mask extraction (resumed)")
            foreground_mask = np.load(fg_mask_cache).astype(np.float32)
        else:
            _log_stage(self.config, "6/8 Foreground mask extraction from initial pass (SAM3)")
            if affordance_pass_image is None:
                raise RuntimeError("Affordance image is required for foreground mask extraction.")
            foreground_mask = self.foreground_mask_extractor.extract(affordance_pass_image, scene_reconstruction_image)
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
        final_controlnet, final_control_image, final_control_scale = _compose_controlnet_inputs(
            depth_controlnet=self.affordance_controlnet if use_final_depth_controlnet else None,
            depth_control_image=depth_control_image_tensor,
            depth_scale=self.config.final_depth_controlnet_scale,
            openpose_controlnet=self.affordance_openpose_controlnet if use_final_pose_controlnet else None,
            openpose_control_image=openpose_control_image_tensor,
            openpose_scale=self.config.final_pose_controlnet_scale,
        )

        original_processors = None
        if self.config.attention_enabled:
            original_processors = install_mask_guided_processors(
                self.pipe,
                controller,
                target_prefixes=self.config.attention_target_prefixes,
            )

        try:
            if self.config.attention_enabled:
                if reference_inv_start is None or reference_prompt_embeds_attn is None:
                    raise RuntimeError("Reference inversion artifacts are required for attention capture.")
                _log_stage(self.config, "8/8 Reference attention capture")
                controller.set_mode(controller.MODE_CAPTURE)
                run_denoise_trajectory(
                    pipe=self.pipe,
                    start_latents=reference_inv_start,
                    prompt_embeds=reference_prompt_embeds_attn,
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
                prompt_embeds=final_prompt_embeds,
                guidance_scale=self.config.final_guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                attn_controller=controller if self.config.attention_enabled else None,
                post_step_hook=blender,
                controlnet=final_controlnet,
                control_image=final_control_image,
                controlnet_conditioning_scale=final_control_scale,
                controlnet_inject_start_step=final_controlnet_start_step,
                controlnet_inject_end_step=final_controlnet_end_step,
                controlnet_residual_suppress_mask=affordance_controlnet_residual_suppress_mask,
                controlnet_mask_inject_start_step=affordance_controlnet_mask_start_step,
                controlnet_mask_inject_end_step=affordance_controlnet_mask_end_step,
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


def _resize_for_model(image: Image.Image, target_size: int) -> Image.Image:
    if target_size > 0:
        return image.resize((target_size, target_size), Image.Resampling.LANCZOS)

    width, height = image.size
    new_width = max(8, (width // 8) * 8)
    new_height = max(8, (height // 8) * 8)
    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _mask_to_pil(mask: np.ndarray) -> Image.Image:
    arr = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _control_image_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def _mask_to_tensor(mask: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def _compose_controlnet_inputs(
    depth_controlnet: Optional[ControlNetModel],
    depth_control_image: Optional[torch.Tensor],
    depth_scale: float,
    openpose_controlnet: Optional[ControlNetModel],
    openpose_control_image: Optional[torch.Tensor],
    openpose_scale: float,
) -> tuple[Optional[Any], Optional[Any], Any]:
    controlnets = []
    control_images = []
    control_scales = []

    if depth_controlnet is not None:
        if depth_control_image is None:
            raise ValueError("Depth ControlNet is enabled but depth control image is missing.")
        controlnets.append(depth_controlnet)
        control_images.append(depth_control_image)
        control_scales.append(float(depth_scale))

    if openpose_controlnet is not None:
        if openpose_control_image is None:
            raise ValueError("OpenPose ControlNet is enabled but openpose control image is missing.")
        controlnets.append(openpose_controlnet)
        control_images.append(openpose_control_image)
        control_scales.append(float(openpose_scale))

    if not controlnets:
        return None, None, 1.0
    if len(controlnets) == 1:
        return controlnets[0], control_images[0], control_scales[0]
    return tuple(controlnets), tuple(control_images), tuple(control_scales)


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


def _log_inversion_config(
    config: TeleportraitConfig,
    scheduler: DDIMScheduler,
    scene_image_size: tuple[int, int],
    reference_image_size: tuple[int, int],
    inversion_do_cfg: bool,
    scene_inversion_prompt: str,
    reference_inversion_prompt: str,
) -> None:
    if not config.verbose:
        return

    scheduler_config = getattr(scheduler, "config", None)
    prediction_type = getattr(scheduler_config, "prediction_type", "unknown")
    beta_schedule = getattr(scheduler_config, "beta_schedule", "unknown")
    timestep_spacing = getattr(scheduler_config, "timestep_spacing", "unknown")
    set_alpha_to_one = getattr(scheduler_config, "set_alpha_to_one", "unknown")
    steps_offset = getattr(scheduler_config, "steps_offset", "unknown")

    print("[Teleportraits] Inversion debug:", flush=True)
    print(f"[Teleportraits]   scheduler={scheduler.__class__.__name__}", flush=True)
    print(f"[Teleportraits]   prediction_type={prediction_type}", flush=True)
    print(f"[Teleportraits]   beta_schedule={beta_schedule}", flush=True)
    print(f"[Teleportraits]   timestep_spacing={timestep_spacing}", flush=True)
    print(f"[Teleportraits]   set_alpha_to_one={set_alpha_to_one}", flush=True)
    print(f"[Teleportraits]   steps_offset={steps_offset}", flush=True)
    print(f"[Teleportraits]   num_inference_steps={config.num_inference_steps}", flush=True)
    print(f"[Teleportraits]   inversion_guidance_scale={config.inversion_guidance_scale}", flush=True)
    print(f"[Teleportraits]   inversion_fixed_point_iters={config.inversion_fixed_point_iters}", flush=True)
    print(f"[Teleportraits]   inversion_do_cfg={inversion_do_cfg}", flush=True)
    print(f"[Teleportraits]   image_size_cfg={config.image_size}", flush=True)
    print(f"[Teleportraits]   scene_image_size={scene_image_size[0]}x{scene_image_size[1]}", flush=True)
    print(f"[Teleportraits]   reference_image_size={reference_image_size[0]}x{reference_image_size[1]}", flush=True)
    print(f"[Teleportraits]   device={config.device}", flush=True)
    print(f"[Teleportraits]   torch_dtype={config.torch_dtype}", flush=True)
    print(f"[Teleportraits]   scene_inversion_prompt={scene_inversion_prompt}", flush=True)
    print(f"[Teleportraits]   reference_inversion_prompt={reference_inversion_prompt}", flush=True)


def _resolve_run_output_dir(base_output_dir: Path, config: TeleportraitConfig, use_child_run: bool = False) -> Path:
    # If user already points to a concrete experiment folder, use it directly.
    if base_output_dir.name.startswith("exp_"):
        if use_child_run:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = base_output_dir / f"rerun_{timestamp}"
            suffix = 1
            while run_dir.exists():
                run_dir = base_output_dir / f"rerun_{timestamp}_{suffix:02d}"
                suffix += 1
            _log_stage(config, f"Resuming explicit run dir via child run: {run_dir.name}")
            return run_dir
        _log_stage(config, f"Resuming explicit run dir: {base_output_dir.name}")
        return base_output_dir

    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Root output dir always creates a fresh run folder.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"exp_{timestamp}"
    # Guard rare same-second collisions.
    if run_dir.exists():
        suffix = 1
        while (base_output_dir / f"exp_{timestamp}_{suffix:02d}").exists():
            suffix += 1
        run_dir = base_output_dir / f"exp_{timestamp}_{suffix:02d}"
    _log_stage(config, f"Creating new run: {run_dir.name}")
    return run_dir


def _ensure_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"Found non-finite values in {name}. Try --torch-dtype float32 and lower --inversion-fixed-point-iters."
        )


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_cache_file(cache_path: Path, parent_cache_path: Optional[Path], filename: str) -> Path:
    local_file = cache_path / filename
    if local_file.exists():
        return local_file
    if parent_cache_path is not None:
        parent_file = parent_cache_path / filename
        if parent_file.exists():
            return parent_file
    return local_file
