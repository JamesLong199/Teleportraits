from __future__ import annotations

import argparse
import json

from teleportraits.config import TeleportraitConfig
from teleportraits.pipeline import TeleportraitsPipeline

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, "
    "extra limbs, missing fingers, malformed limbs, mutated, deformed, disfigured, "
    "poorly drawn, jpeg artifacts, watermark, text, logo, signature, cropped, out of frame"
)


def _parse_step_range(value: str, arg_name: str) -> tuple[int, int]:
    text = value.strip()
    if ":" not in text:
        raise ValueError(f"{arg_name} must be in 'start:end' format, got: {value!r}")
    start_str, end_str = text.split(":", 1)
    try:
        start = int(start_str.strip())
        end = int(end_str.strip())
    except ValueError as exc:
        raise ValueError(f"{arg_name} must contain integer start/end, got: {value!r}") from exc
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teleportraits reproduction pipeline")

    parser.add_argument("--scene-image", required=True, help="Path to scene image")
    parser.add_argument("--reference-image", required=True, help="Path to reference person image")
    parser.add_argument(
        "--foreground-mask-image",
        default=None,
        help="Optional binary/gray mask for final-pass foreground region (white=subject, black=background).",
    )
    parser.add_argument(
        "--reference-mask-image",
        default=None,
        help="Optional binary/gray mask for the reference image (white=subject, black=background).",
    )
    parser.add_argument(
        "--reference-mask-invert",
        action="store_true",
        help="Invert the provided reference mask (useful if your mask uses black=subject).",
    )
    parser.add_argument("--scene-prompt", required=True, help="Prompt describing the scene")
    parser.add_argument("--reference-prompt", required=True, help="Prompt describing the reference subject")
    parser.add_argument(
        "--edit-prompt",
        default=None,
        help="Optional override for insertion prompt; if omitted, composed from scene/reference prompts.",
    )
    parser.add_argument(
        "--person-placeholder",
        default="a person",
        help="Placeholder phrase in scene prompt to be replaced by reference prompt.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")

    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=1024, help="Resize scene/reference to square size (paper: 1024).")

    parser.add_argument("--inversion-guidance-scale", type=float, default=1.0)
    parser.add_argument("--inversion-fixed-point-iters", type=int, default=2)
    parser.add_argument(
        "--inversion-prompt",
        default=None,
        help="Optional prompt override for scene inversion only. If omitted, uses --scene-prompt.",
    )

    parser.add_argument("--edit-guidance-scale", type=float, default=7.5)
    parser.add_argument(
        "--affordance-controlnet-depth",
        action="store_true",
        help="Enable ControlNet Depth conditioning in the affordance (initial human) pass.",
    )
    parser.add_argument(
        "--affordance-controlnet-model-id",
        type=str,
        default="diffusers/controlnet-depth-sdxl-1.0",
        help="ControlNet model id/path for affordance depth conditioning.",
    )
    parser.add_argument(
        "--affordance-controlnet-dir",
        type=str,
        default="./pretrained/controlnet-depth-sdxl-1.0",
        help=(
            "Local directory for ControlNet Depth weights. If it does not exist, "
            "--affordance-controlnet-model-id is used."
        ),
    )
    parser.add_argument(
        "--affordance-controlnet-scale",
        type=float,
        default=1.0,
        help="Conditioning scale for affordance depth ControlNet.",
    )
    parser.add_argument(
        "--moge-pretrained-model",
        type=str,
        default="Ruicheng/moge-2-vitl-normal",
        help="MoGe pretrained model name/path.",
    )
    parser.add_argument(
        "--moge-checkpoint-dir",
        type=str,
        default="./pretrained/moge",
        help=(
            "Local MoGe checkpoint directory or file. Accepted layouts include "
            "<dir>/model.pt or <dir>/<version>/model.pt."
        ),
    )
    parser.add_argument(
        "--moge-model-version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="MoGe model version.",
    )
    parser.add_argument(
        "--moge-conda-env",
        type=str,
        default="",
        help="Optional conda env for running MoGe depth extraction. Empty uses current Python env.",
    )
    parser.add_argument(
        "--moge-no-fp16",
        action="store_true",
        help="Disable fp16 in MoGe depth inference.",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)

    parser.add_argument("--blend-start-step", type=int, default=10)
    parser.add_argument("--blend-end-step", type=int, default=20)
    parser.add_argument(
        "--latent-blend-range",
        type=str,
        default=None,
        help="Latent blending step range as 'start:end' (inclusive), e.g. 10:20.",
    )

    parser.add_argument("--attention-enabled", action="store_true", dest="attention_enabled")
    parser.add_argument("--attention-disabled", action="store_false", dest="attention_enabled")
    parser.set_defaults(attention_enabled=True)
    parser.add_argument("--attention-inject-start-step", type=int, default=0)
    parser.add_argument("--attention-inject-end-step", type=int, default=49)
    parser.add_argument(
        "--mask-guided-attn-range",
        type=str,
        default=None,
        help=(
            "Mask-guided self-attention step range as 'start:end' (inclusive), e.g. 0:49."
        ),
    )
    parser.add_argument(
        "--affordance-only",
        action="store_true",
        help="Run only scene inversion, scene reconstruction, and initial/affordance pass, then stop.",
    )

    parser.add_argument("--mask-threshold", type=float, default=0.08)
    parser.add_argument("--mask-min-area-ratio", type=float, default=0.001)
    parser.add_argument(
        "--foreground-mask-prompt",
        type=str,
        default="person",
        help="Text prompt used by SAM3 to extract the foreground mask from affordance pass.",
    )
    parser.add_argument(
        "--foreground-mask-confidence-threshold",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold for foreground mask extraction.",
    )
    parser.add_argument(
        "--sam3-checkpoint-dir",
        type=str,
        default="/mnt/whuscs/ljz/sam3",
        help="Local SAM3 checkpoint directory containing sam3.pt (and related files).",
    )
    parser.add_argument(
        "--sam3-conda-env",
        type=str,
        default="sam3",
        help="Conda environment name used for SAM3 foreground mask extraction.",
    )
    parser.add_argument(
        "--use-transformers-reference-mask",
        action="store_true",
        help="Use transformers person segmentation model for reference masking (downloads extra weights).",
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--quiet", action="store_true", help="Reduce stage logging.")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable diffusion progress bars.")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    blend_start_step = args.blend_start_step
    blend_end_step = args.blend_end_step
    if args.latent_blend_range is not None:
        blend_start_step, blend_end_step = _parse_step_range(
            args.latent_blend_range, "--latent-blend-range"
        )

    attention_inject_start_step = args.attention_inject_start_step
    attention_inject_end_step = args.attention_inject_end_step
    if args.mask_guided_attn_range is not None:
        attention_inject_start_step, attention_inject_end_step = _parse_step_range(
            args.mask_guided_attn_range, "--mask-guided-attn-range"
        )

    config = TeleportraitConfig(
        model_id=args.model_id,
        num_inference_steps=args.num_inference_steps,
        image_size=args.image_size,
        inversion_guidance_scale=args.inversion_guidance_scale,
        inversion_fixed_point_iters=args.inversion_fixed_point_iters,
        inversion_prompt=args.inversion_prompt,
        edit_guidance_scale=args.edit_guidance_scale,
        affordance_use_controlnet_depth=args.affordance_controlnet_depth,
        affordance_controlnet_model_id=args.affordance_controlnet_model_id,
        affordance_controlnet_dir=args.affordance_controlnet_dir,
        affordance_controlnet_scale=args.affordance_controlnet_scale,
        moge_pretrained_model=args.moge_pretrained_model,
        moge_checkpoint_dir=args.moge_checkpoint_dir,
        moge_model_version=args.moge_model_version,
        moge_conda_env=args.moge_conda_env,
        moge_use_fp16=not args.moge_no_fp16,
        negative_prompt=args.negative_prompt,
        blend_start_step=blend_start_step,
        blend_end_step=blend_end_step,
        attention_enabled=args.attention_enabled,
        attention_inject_start_step=attention_inject_start_step,
        attention_inject_end_step=attention_inject_end_step,
        affordance_only=args.affordance_only,
        mask_threshold=args.mask_threshold,
        mask_min_area_ratio=args.mask_min_area_ratio,
        foreground_mask_prompt=args.foreground_mask_prompt,
        foreground_mask_confidence_threshold=args.foreground_mask_confidence_threshold,
        sam3_checkpoint_dir=args.sam3_checkpoint_dir,
        sam3_conda_env=args.sam3_conda_env,
        use_transformers_reference_mask=args.use_transformers_reference_mask,
        verbose=not args.quiet,
        show_progress_bar=not args.no_progress_bar,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    pipeline = TeleportraitsPipeline(config)
    outputs = pipeline.run(
        scene_image_path=args.scene_image,
        reference_image_path=args.reference_image,
        foreground_mask_path=args.foreground_mask_image,
        reference_mask_path=args.reference_mask_image,
        reference_mask_invert=args.reference_mask_invert,
        scene_prompt=args.scene_prompt,
        reference_prompt=args.reference_prompt,
        edit_prompt=args.edit_prompt,
        output_dir=args.output_dir,
        person_placeholder=args.person_placeholder,
    )

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
