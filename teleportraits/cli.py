from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from teleportraits.config import TeleportraitConfig
from teleportraits.pipeline import TeleportraitsPipeline

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, "
    "extra limbs, missing fingers, malformed limbs, mutated, deformed, disfigured, "
    "poorly drawn, jpeg artifacts, watermark, text, logo, signature, cropped, out of frame"
)


def _as_dict(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object/dict, got: {type(value).__name__}")
    return value


def _as_bool(value: Any, name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be boolean-like, got: {value!r}")


def _as_float(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a float, got: {value!r}") from exc


def _as_int(value: Any, name: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an int, got: {value!r}") from exc


def _as_str(
    value: Any,
    name: str,
    default: str | None = None,
    required: bool = False,
    allow_empty: bool = False,
) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return default
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got: {type(value).__name__}")
    text = value.strip()
    if required and not text:
        raise ValueError(f"{name} cannot be empty")
    if text == "" and not allow_empty and default is not None:
        return default
    return text


def _as_str_tuple(value: Any, name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of strings")
    out = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name}[{idx}] must be a non-empty string")
        out.append(item.strip())
    return tuple(out)


def _parse_step_range(value: Any, name: str, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if ":" not in text:
            raise ValueError(f"{name} must be 'start:end', got: {value!r}")
        start_text, end_text = text.split(":", 1)
        return int(start_text.strip()), int(end_text.strip())
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, dict):
        if "start" not in value or "end" not in value:
            raise ValueError(f"{name} dict must contain start/end")
        return int(value["start"]), int(value["end"])
    raise ValueError(f"{name} must be a range string/list/dict, got: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teleportraits pipeline (JSON input mode)")
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to JSON input config describing models, pass-specific settings, and I/O.",
    )
    return parser


def _load_json(path: str) -> Dict[str, Any]:
    input_path = Path(path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON does not exist: {path}")
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Input JSON root must be an object/dict.")
    return payload


def _build_from_json(payload: Dict[str, Any]) -> Tuple[TeleportraitConfig, Dict[str, Any]]:
    io_cfg = _as_dict(payload.get("io"), "io")
    models_cfg = _as_dict(payload.get("models"), "models")
    passes_cfg = _as_dict(payload.get("passes"), "passes")
    runtime_cfg = _as_dict(payload.get("runtime"), "runtime")

    inversion_cfg = _as_dict(passes_cfg.get("inversion"), "passes.inversion")
    affordance_cfg = _as_dict(passes_cfg.get("affordance"), "passes.affordance")
    affordance_refine_cfg = _as_dict(passes_cfg.get("affordance_refine"), "passes.affordance_refine")
    final_cfg = _as_dict(passes_cfg.get("final"), "passes.final")

    affordance_depth_cfg = _as_dict(
        affordance_cfg.get("controlnet_depth", affordance_cfg.get("depth_controlnet")),
        "passes.affordance.controlnet_depth",
    )
    affordance_pose_cfg = _as_dict(
        affordance_cfg.get("controlnet_pose", affordance_cfg.get("pose_controlnet")),
        "passes.affordance.controlnet_pose",
    )
    affordance_refine_depth_cfg = _as_dict(
        affordance_refine_cfg.get("controlnet_depth", affordance_refine_cfg.get("depth_controlnet")),
        "passes.affordance_refine.controlnet_depth",
    )
    affordance_refine_pose_cfg = _as_dict(
        affordance_refine_cfg.get("controlnet_pose", affordance_refine_cfg.get("pose_controlnet")),
        "passes.affordance_refine.controlnet_pose",
    )
    final_depth_cfg = _as_dict(
        final_cfg.get("controlnet_depth", final_cfg.get("depth_controlnet")),
        "passes.final.controlnet_depth",
    )
    final_pose_cfg = _as_dict(
        final_cfg.get("controlnet_pose", final_cfg.get("pose_controlnet")),
        "passes.final.controlnet_pose",
    )
    affordance_fg_mask_cfg = _as_dict(affordance_cfg.get("foreground_mask"), "passes.affordance.foreground_mask")
    final_masks_cfg = _as_dict(final_cfg.get("masks"), "passes.final.masks")
    final_scene_fg_mask_cfg = _as_dict(
        final_masks_cfg.get("scene_foreground_mask", final_cfg.get("scene_foreground_mask")),
        "passes.final.masks.scene_foreground_mask",
    )
    final_ref_mask_cfg = _as_dict(
        final_masks_cfg.get("reference_foreground_mask", final_cfg.get("reference_foreground_mask")),
        "passes.final.masks.reference_foreground_mask",
    )
    final_attention_cfg = _as_dict(final_cfg.get("attention"), "passes.final.attention")

    model_depth_cfg = _as_dict(models_cfg.get("controlnet_depth"), "models.controlnet_depth")
    model_openpose_cfg = _as_dict(models_cfg.get("controlnet_openpose"), "models.controlnet_openpose")
    model_moge_cfg = _as_dict(models_cfg.get("moge"), "models.moge")
    model_openpose_detector_cfg = _as_dict(models_cfg.get("openpose_detector"), "models.openpose_detector")
    model_sam3_cfg = _as_dict(models_cfg.get("sam3"), "models.sam3")

    scene_image = _as_str(io_cfg.get("scene_image"), "io.scene_image", required=True)
    reference_image = _as_str(io_cfg.get("reference_image"), "io.reference_image", required=True)
    output_dir = _as_str(io_cfg.get("output_dir"), "io.output_dir", required=True)

    affordance_prompt = _as_str(
        affordance_cfg.get("prompt"),
        "passes.affordance.prompt",
        required=True,
    )
    final_scene_prompt = _as_str(
        final_cfg.get("scene_prompt"),
        "passes.final.scene_prompt",
        default=affordance_prompt,
    )
    final_prompt = _as_str(
        final_cfg.get("prompt"),
        "passes.final.prompt",
        required=True,
    )
    reference_prompt = _as_str(
        final_cfg.get("reference_prompt"),
        "passes.final.reference_prompt",
        default=_as_str(inversion_cfg.get("reference_prompt"), "passes.inversion.reference_prompt"),
        required=True,
    )
    affordance_refine_prompt = _as_str(
        affordance_refine_cfg.get("prompt"),
        "passes.affordance_refine.prompt",
        default=None,
    )

    image_size = _as_int(runtime_cfg.get("image_size"), "runtime.image_size", 1024)
    num_inference_steps = _as_int(runtime_cfg.get("num_inference_steps"), "runtime.num_inference_steps", 50)

    inversion_prompt_override = _as_str(
        inversion_cfg.get("scene_prompt"),
        "passes.inversion.scene_prompt",
        default=affordance_prompt,
    )
    inversion_reference_prompt = _as_str(
        inversion_cfg.get("reference_prompt"),
        "passes.inversion.reference_prompt",
        default=reference_prompt,
    )
    inversion_guidance_scale = _as_float(
        inversion_cfg.get("guidance_scale"),
        "passes.inversion.guidance_scale",
        1.0,
    )
    inversion_fixed_point_iters = _as_int(
        inversion_cfg.get("fixed_point_iters"),
        "passes.inversion.fixed_point_iters",
        2,
    )
    random_start_latent = _as_bool(
        inversion_cfg.get("random_start_latent"),
        "passes.inversion.random_start_latent",
        False,
    )

    negative_prompt = _as_str(
        affordance_cfg.get("negative_prompt"),
        "passes.affordance.negative_prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        allow_empty=True,
    )
    affordance_guidance_scale = _as_float(
        affordance_cfg.get("guidance_scale"),
        "passes.affordance.guidance_scale",
        7.5,
    )
    affordance_refine_guidance_scale = _as_float(
        affordance_refine_cfg.get("guidance_scale"),
        "passes.affordance_refine.guidance_scale",
        affordance_guidance_scale,
    )
    final_guidance_scale = _as_float(
        final_cfg.get("guidance_scale"),
        "passes.final.guidance_scale",
        7.5,
    )

    blend_start_step, blend_end_step = _parse_step_range(
        final_cfg.get("latent_blend_range"),
        "passes.final.latent_blend_range",
        (10, 20),
    )
    attention_start_step, attention_end_step = _parse_step_range(
        final_attention_cfg.get("range"),
        "passes.final.attention.range",
        (0, 49),
    )
    affordance_controlnet_start_step, affordance_controlnet_end_step = _parse_step_range(
        affordance_cfg.get("controlnet_range", affordance_depth_cfg.get("range")),
        "passes.affordance.controlnet_range",
        (0, 999),
    )
    affordance_controlnet_mask_start_step, affordance_controlnet_mask_end_step = _parse_step_range(
        affordance_depth_cfg.get("mask_range"),
        "passes.affordance.controlnet_depth.mask_range",
        (0, 999),
    )
    affordance_refine_controlnet_start_step, affordance_refine_controlnet_end_step = _parse_step_range(
        affordance_refine_cfg.get(
            "controlnet_range",
            affordance_refine_depth_cfg.get("range", affordance_refine_pose_cfg.get("range")),
        ),
        "passes.affordance_refine.controlnet_range",
        (0, 999),
    )
    final_controlnet_start_step, final_controlnet_end_step = _parse_step_range(
        final_cfg.get("controlnet_range", final_depth_cfg.get("range", final_pose_cfg.get("range"))),
        "passes.final.controlnet_range",
        (0, 999),
    )

    use_affordance_depth_controlnet = _as_bool(
        affordance_depth_cfg.get("enabled"),
        "passes.affordance.controlnet_depth.enabled",
        False,
    )
    affordance_depth_control_scale = _as_float(
        affordance_depth_cfg.get("scale"),
        "passes.affordance.controlnet_depth.scale",
        1.0,
    )
    depth_control_mask_image = _as_str(
        affordance_depth_cfg.get("mask_image"),
        "passes.affordance.controlnet_depth.mask_image",
        default=None,
    )
    depth_control_mask_invert = _as_bool(
        affordance_depth_cfg.get("mask_invert"),
        "passes.affordance.controlnet_depth.mask_invert",
        False,
    )
    use_affordance_pose_controlnet = _as_bool(
        affordance_pose_cfg.get("enabled"),
        "passes.affordance.controlnet_pose.enabled",
        False,
    )
    affordance_pose_control_scale = _as_float(
        affordance_pose_cfg.get("scale"),
        "passes.affordance.controlnet_pose.scale",
        1.0,
    )
    affordance_pose_image_path = _as_str(
        affordance_pose_cfg.get("path"),
        "passes.affordance.controlnet_pose.path",
        default=None,
    )

    use_affordance_refine_depth_controlnet = _as_bool(
        affordance_refine_depth_cfg.get("enabled"),
        "passes.affordance_refine.controlnet_depth.enabled",
        False,
    )
    affordance_refine_depth_control_scale = _as_float(
        affordance_refine_depth_cfg.get("scale"),
        "passes.affordance_refine.controlnet_depth.scale",
        1.0,
    )
    use_affordance_refine_pose_controlnet = _as_bool(
        affordance_refine_pose_cfg.get("enabled", affordance_refine_cfg.get("enabled")),
        "passes.affordance_refine.controlnet_pose.enabled",
        False,
    )
    affordance_refine_pose_control_scale = _as_float(
        affordance_refine_pose_cfg.get("scale"),
        "passes.affordance_refine.controlnet_pose.scale",
        1.0,
    )

    use_final_depth_controlnet = _as_bool(
        final_depth_cfg.get("enabled"),
        "passes.final.controlnet_depth.enabled",
        False,
    )
    final_depth_control_scale = _as_float(
        final_depth_cfg.get("scale"),
        "passes.final.controlnet_depth.scale",
        1.0,
    )
    use_final_pose_controlnet = _as_bool(
        final_pose_cfg.get("enabled"),
        "passes.final.controlnet_pose.enabled",
        False,
    )
    final_pose_control_scale = _as_float(
        final_pose_cfg.get("scale"),
        "passes.final.controlnet_pose.scale",
        1.0,
    )

    attention_enabled = _as_bool(
        final_attention_cfg.get("enabled"),
        "passes.final.attention.enabled",
        True,
    )
    attention_target_prefixes = _as_str_tuple(
        final_attention_cfg.get("target_prefixes"),
        "passes.final.attention.target_prefixes",
        ("up_blocks.1", "up_blocks.2"),
    )

    affordance_only = _as_bool(final_cfg.get("affordance_only"), "passes.final.affordance_only", False)
    foreground_mask_source = _as_str(
        final_scene_fg_mask_cfg.get("source"),
        "passes.final.masks.scene_foreground_mask.source",
        default="affordance_pass_sam3",
    )
    foreground_mask_path = _as_str(
        final_scene_fg_mask_cfg.get("path", affordance_fg_mask_cfg.get("path", affordance_fg_mask_cfg.get("override_image"))),
        "passes.final.masks.scene_foreground_mask.path",
        default=None,
    )
    if foreground_mask_path:
        foreground_mask_source = "file"
    if foreground_mask_source == "file" and not foreground_mask_path:
        raise ValueError("passes.final.masks.scene_foreground_mask.path is required when source='file'.")
    foreground_mask_override = foreground_mask_path if foreground_mask_source == "file" else None
    foreground_mask_prompt = _as_str(
        affordance_fg_mask_cfg.get("prompt"),
        "passes.affordance.foreground_mask.prompt",
        default="person",
    )
    foreground_mask_confidence_threshold = _as_float(
        affordance_fg_mask_cfg.get("confidence_threshold"),
        "passes.affordance.foreground_mask.confidence_threshold",
        0.5,
    )
    reference_mask_path = _as_str(
        final_ref_mask_cfg.get("path"),
        "passes.final.masks.reference_foreground_mask.path",
        default=None,
    )
    reference_mask_invert = _as_bool(
        final_ref_mask_cfg.get("invert"),
        "passes.final.masks.reference_foreground_mask.invert",
        False,
    )

    config = TeleportraitConfig(
        model_id=_as_str(models_cfg.get("sdxl"), "models.sdxl", default="stabilityai/stable-diffusion-xl-base-1.0"),
        num_inference_steps=num_inference_steps,
        image_size=image_size,
        inversion_guidance_scale=inversion_guidance_scale,
        inversion_fixed_point_iters=inversion_fixed_point_iters,
        inversion_prompt=inversion_prompt_override,
        random_start_latent=random_start_latent,
        edit_guidance_scale=final_guidance_scale,
        affordance_guidance_scale=affordance_guidance_scale,
        affordance_refine_guidance_scale=affordance_refine_guidance_scale,
        final_guidance_scale=final_guidance_scale,
        affordance_use_controlnet_depth=use_affordance_depth_controlnet,
        affordance_use_controlnet_pose=use_affordance_pose_controlnet,
        affordance_pose_image_path=affordance_pose_image_path,
        affordance_controlnet_model_id=_as_str(
            model_depth_cfg.get("model_id"),
            "models.controlnet_depth.model_id",
            default="diffusers/controlnet-depth-sdxl-1.0",
        ),
        affordance_controlnet_dir=_as_str(
            model_depth_cfg.get("model_dir"),
            "models.controlnet_depth.model_dir",
            default="./pretrained/controlnet-depth-sdxl-1.0",
        ),
        affordance_controlnet_scale=affordance_depth_control_scale,
        affordance_pose_controlnet_scale=affordance_pose_control_scale,
        affordance_controlnet_start_step=affordance_controlnet_start_step,
        affordance_controlnet_end_step=affordance_controlnet_end_step,
        affordance_refine_use_controlnet_depth=use_affordance_refine_depth_controlnet,
        affordance_refine_use_controlnet_pose=use_affordance_refine_pose_controlnet,
        affordance_refine_controlnet_start_step=affordance_refine_controlnet_start_step,
        affordance_refine_controlnet_end_step=affordance_refine_controlnet_end_step,
        affordance_refine_depth_controlnet_scale=affordance_refine_depth_control_scale,
        affordance_refine_pose_controlnet_scale=affordance_refine_pose_control_scale,
        final_use_controlnet_depth=use_final_depth_controlnet,
        final_use_controlnet_pose=use_final_pose_controlnet,
        final_controlnet_start_step=final_controlnet_start_step,
        final_controlnet_end_step=final_controlnet_end_step,
        final_depth_controlnet_scale=final_depth_control_scale,
        final_pose_controlnet_scale=final_pose_control_scale,
        affordance_openpose_controlnet_model_id=_as_str(
            model_openpose_cfg.get("model_id"),
            "models.controlnet_openpose.model_id",
            default="thibaud/controlnet-openpose-sdxl-1.0",
        ),
        affordance_openpose_controlnet_dir=_as_str(
            model_openpose_cfg.get("model_dir"),
            "models.controlnet_openpose.model_dir",
            default="./pretrained/controlnet-openpose-sdxl-1.0",
        ),
        affordance_openpose_controlnet_scale=affordance_refine_pose_control_scale,
        openpose_detector_model_id=_as_str(
            model_openpose_detector_cfg.get("model_id"),
            "models.openpose_detector.model_id",
            default="lllyasviel/Annotators",
        ),
        openpose_detector_dir=_as_str(
            model_openpose_detector_cfg.get("model_dir"),
            "models.openpose_detector.model_dir",
            default="",
        ),
        affordance_controlnet_mask_image=depth_control_mask_image,
        affordance_controlnet_mask_invert=depth_control_mask_invert,
        affordance_controlnet_mask_start_step=affordance_controlnet_mask_start_step,
        affordance_controlnet_mask_end_step=affordance_controlnet_mask_end_step,
        moge_pretrained_model=_as_str(
            model_moge_cfg.get("pretrained_model"),
            "models.moge.pretrained_model",
            default="Ruicheng/moge-2-vitl-normal",
        ),
        moge_checkpoint_dir=_as_str(
            model_moge_cfg.get("checkpoint_dir"),
            "models.moge.checkpoint_dir",
            default="./pretrained/moge",
        ),
        moge_model_version=_as_str(model_moge_cfg.get("model_version"), "models.moge.model_version", default="v2"),
        moge_conda_env=_as_str(model_moge_cfg.get("conda_env"), "models.moge.conda_env", default=""),
        moge_use_fp16=_as_bool(model_moge_cfg.get("use_fp16"), "models.moge.use_fp16", True),
        negative_prompt=negative_prompt,
        blend_start_step=blend_start_step,
        blend_end_step=blend_end_step,
        attention_enabled=attention_enabled,
        attention_inject_start_step=attention_start_step,
        attention_inject_end_step=attention_end_step,
        attention_target_prefixes=attention_target_prefixes,
        affordance_only=affordance_only,
        mask_threshold=_as_float(runtime_cfg.get("mask_threshold"), "runtime.mask_threshold", 0.08),
        mask_min_area_ratio=_as_float(runtime_cfg.get("mask_min_area_ratio"), "runtime.mask_min_area_ratio", 0.001),
        foreground_mask_prompt=foreground_mask_prompt,
        foreground_mask_confidence_threshold=foreground_mask_confidence_threshold,
        sam3_checkpoint_dir=_as_str(
            model_sam3_cfg.get("checkpoint_dir"),
            "models.sam3.checkpoint_dir",
            default="/mnt/whuscs/ljz/sam3",
        ),
        sam3_conda_env=_as_str(
            model_sam3_cfg.get("conda_env"),
            "models.sam3.conda_env",
            default="sam3",
        ),
        use_transformers_reference_mask=_as_bool(
            runtime_cfg.get("use_transformers_reference_mask"),
            "runtime.use_transformers_reference_mask",
            False,
        ),
        verbose=_as_bool(runtime_cfg.get("verbose"), "runtime.verbose", True),
        show_progress_bar=_as_bool(runtime_cfg.get("show_progress_bar"), "runtime.show_progress_bar", True),
        seed=_as_int(runtime_cfg.get("seed"), "runtime.seed", 0),
        torch_dtype=_as_str(runtime_cfg.get("torch_dtype"), "runtime.torch_dtype", default="float16"),
        device=_as_str(runtime_cfg.get("device"), "runtime.device", default="cuda"),
    )

    run_kwargs = {
        "scene_image_path": scene_image,
        "reference_image_path": reference_image,
        "foreground_mask_path": foreground_mask_override,
        "reference_mask_path": reference_mask_path,
        "reference_mask_invert": reference_mask_invert,
        "affordance_prompt": affordance_prompt,
        "final_scene_prompt": final_scene_prompt,
        "final_prompt": final_prompt,
        "reference_prompt": reference_prompt,
        "output_dir": output_dir,
        "affordance_refine_prompt": affordance_refine_prompt,
        "inversion_reference_prompt": inversion_reference_prompt,
    }
    return config, run_kwargs


def main() -> None:
    args = build_parser().parse_args()
    payload = _load_json(args.input_json)
    config, run_kwargs = _build_from_json(payload)

    pipeline = TeleportraitsPipeline(config)
    outputs = pipeline.run(
        scene_image_path=run_kwargs["scene_image_path"],
        reference_image_path=run_kwargs["reference_image_path"],
        foreground_mask_path=run_kwargs["foreground_mask_path"],
        reference_mask_path=run_kwargs["reference_mask_path"],
        reference_mask_invert=run_kwargs["reference_mask_invert"],
        affordance_prompt=run_kwargs["affordance_prompt"],
        final_scene_prompt=run_kwargs["final_scene_prompt"],
        final_prompt=run_kwargs["final_prompt"],
        reference_prompt=run_kwargs["reference_prompt"],
        output_dir=run_kwargs["output_dir"],
        affordance_refine_prompt=run_kwargs["affordance_refine_prompt"],
        inversion_reference_prompt=run_kwargs["inversion_reference_prompt"],
        input_json_path=args.input_json,
    )

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
