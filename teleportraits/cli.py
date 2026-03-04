from __future__ import annotations

import argparse
import json

from teleportraits.config import TeleportraitConfig
from teleportraits.pipeline import TeleportraitsPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teleportraits reproduction pipeline")

    parser.add_argument("--scene-image", required=True, help="Path to scene image")
    parser.add_argument("--reference-image", required=True, help="Path to reference person image")
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

    parser.add_argument("--inversion-guidance-scale", type=float, default=1.0)
    parser.add_argument("--inversion-fixed-point-iters", type=int, default=5)

    parser.add_argument("--edit-guidance-scale", type=float, default=9.0)
    parser.add_argument("--negative-prompt", default="")

    parser.add_argument("--blend-start-step", type=int, default=15)
    parser.add_argument("--blend-end-step", type=int, default=40)

    parser.add_argument("--attention-enabled", action="store_true", dest="attention_enabled")
    parser.add_argument("--attention-disabled", action="store_false", dest="attention_enabled")
    parser.set_defaults(attention_enabled=True)
    parser.add_argument("--attention-inject-start-step", type=int, default=0)
    parser.add_argument("--attention-inject-end-step", type=int, default=49)

    parser.add_argument("--mask-threshold", type=float, default=0.08)
    parser.add_argument("--mask-min-area-ratio", type=float, default=0.001)
    parser.add_argument(
        "--use-transformers-reference-mask",
        action="store_true",
        help="Use transformers person segmentation model for reference masking (downloads extra weights).",
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = TeleportraitConfig(
        model_id=args.model_id,
        num_inference_steps=args.num_inference_steps,
        inversion_guidance_scale=args.inversion_guidance_scale,
        inversion_fixed_point_iters=args.inversion_fixed_point_iters,
        edit_guidance_scale=args.edit_guidance_scale,
        negative_prompt=args.negative_prompt,
        blend_start_step=args.blend_start_step,
        blend_end_step=args.blend_end_step,
        attention_enabled=args.attention_enabled,
        attention_inject_start_step=args.attention_inject_start_step,
        attention_inject_end_step=args.attention_inject_end_step,
        mask_threshold=args.mask_threshold,
        mask_min_area_ratio=args.mask_min_area_ratio,
        use_transformers_reference_mask=args.use_transformers_reference_mask,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    pipeline = TeleportraitsPipeline(config)
    outputs = pipeline.run(
        scene_image_path=args.scene_image,
        reference_image_path=args.reference_image,
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
