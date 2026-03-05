from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage

from teleportraits.sdxl_utils import pil_to_np


def _largest_component(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    max_idx = int(np.argmax(sizes)) + 1
    output = labeled == max_idx

    min_pixels = int(mask.shape[0] * mask.shape[1] * min_area_ratio)
    if output.sum() < min_pixels:
        return mask
    return output


def _heuristic_foreground_mask(image: Image.Image) -> np.ndarray:
    arr = pil_to_np(image)
    h, w, _ = arr.shape
    border = max(2, min(h, w) // 20)

    top = arr[:border, :, :]
    bottom = arr[-border:, :, :]
    left = arr[:, :border, :]
    right = arr[:, -border:, :]
    bg_pixels = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )

    bg_color = np.median(bg_pixels, axis=0)
    dist = np.linalg.norm(arr - bg_color[None, None, :], axis=2)
    mask = dist > 0.08
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = _largest_component(mask, min_area_ratio=0.001)

    area = float(mask.mean())
    if area < 0.01 or area > 0.95:
        return np.ones((h, w), dtype=np.float32)
    return mask.astype(np.float32)


@dataclass
class DifferenceMaskExtractor:
    threshold: float = 0.08
    min_area_ratio: float = 0.001

    def extract(self, generated: Image.Image, scene_reconstruction: Image.Image) -> np.ndarray:
        gen = pil_to_np(generated)
        rec = pil_to_np(scene_reconstruction.resize(generated.size, Image.Resampling.BICUBIC))

        diff = np.abs(gen - rec).mean(axis=2)
        mask = diff > self.threshold
        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_closing(mask, iterations=2)
        mask = _largest_component(mask, self.min_area_ratio)
        return mask.astype(np.float32)


class Sam3ForegroundMaskExtractor:
    def __init__(
        self,
        prompt: str,
        confidence_threshold: float = 0.5,
        min_area_ratio: float = 0.001,
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None,
        conda_env: str = "sam3",
    ) -> None:
        self.prompt = prompt.strip()
        self.confidence_threshold = confidence_threshold
        self.min_area_ratio = min_area_ratio
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.conda_env = conda_env

    def extract(self, generated: Image.Image, scene_reconstruction: Image.Image) -> np.ndarray:
        del scene_reconstruction
        if not self.prompt:
            raise ValueError("foreground_mask_prompt must not be empty")

        checkpoint_path = None
        load_from_hf = True
        if self.checkpoint_dir:
            checkpoint_path = Path(self.checkpoint_dir) / "sam3.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"SAM3 checkpoint not found at {checkpoint_path}. "
                    "Expected a directory containing sam3.pt."
                )
            load_from_hf = False

        script = """
import sys
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

in_path = sys.argv[1]
out_path = sys.argv[2]
prompt = sys.argv[3]
threshold = float(sys.argv[4])
device = sys.argv[5]
checkpoint_path = sys.argv[6]
load_from_hf = sys.argv[7] == "1"

model = build_sam3_image_model(
    device=device,
    checkpoint_path=None if checkpoint_path == "" else checkpoint_path,
    load_from_HF=load_from_hf,
)
processor = Sam3Processor(model=model, device=device, confidence_threshold=threshold)
state = processor.set_image(Image.open(in_path).convert("RGB"))
state = processor.set_text_prompt(prompt=prompt, state=state)
masks = state.get("masks")
scores = state.get("scores")
if masks is None or scores is None or int(scores.numel()) == 0:
    raise RuntimeError(f"SAM3 produced no foreground masks for prompt: {prompt!r}")
best_idx = int(scores.argmax().item())
best_mask = masks[best_idx]
if best_mask.ndim == 3:
    best_mask = best_mask.squeeze(0)
mask = best_mask.detach().cpu().numpy().astype(np.uint8)
np.save(out_path, mask)
"""
        with tempfile.TemporaryDirectory(prefix="teleportraits_sam3_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            raw_mask_path = tmp_path / "raw_mask.npy"
            generated.convert("RGB").save(image_path)

            cmd = [
                "conda",
                "run",
                "-n",
                self.conda_env,
                "python",
                "-c",
                script,
                str(image_path),
                str(raw_mask_path),
                self.prompt,
                str(self.confidence_threshold),
                self.device,
                str(checkpoint_path) if checkpoint_path is not None else "",
                "1" if load_from_hf else "0",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                stderr = proc.stderr.strip()
                stdout = proc.stdout.strip()
                details = stderr or stdout or "unknown error"
                raise RuntimeError(
                    "SAM3 extraction failed in conda env "
                    f"{self.conda_env!r}. Details: {details}"
                )

            mask = np.load(raw_mask_path).astype(bool)

        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_closing(mask, iterations=2)
        mask = _largest_component(mask, self.min_area_ratio)
        return mask.astype(np.float32)


class TransformersPersonMaskExtractor:
    def __init__(
        self,
        model_id: str = "facebook/mask2former-swin-large-coco-panoptic",
        score_threshold: float = 0.3,
        device: int = -1,
    ) -> None:
        self.model_id = model_id
        self.score_threshold = score_threshold
        self.device = device
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("transformers is required for person segmentation") from exc

        self._pipeline = pipeline(
            "image-segmentation",
            model=self.model_id,
            device=self.device,
        )

    def extract(self, image: Image.Image) -> Optional[np.ndarray]:
        self._load()
        outputs = self._pipeline(image)

        mask = None
        for item in outputs:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            if "person" not in label or score < self.score_threshold:
                continue

            candidate = item.get("mask")
            if isinstance(candidate, Image.Image):
                arr = np.asarray(candidate.convert("L"), dtype=np.float32) / 255.0
            else:
                arr = np.asarray(candidate, dtype=np.float32)
                if arr.max() > 1.0:
                    arr = arr / 255.0

            candidate_mask = arr > 0.5
            mask = candidate_mask if mask is None else (mask | candidate_mask)

        if mask is None:
            return None

        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_closing(mask, iterations=2)
        return mask.astype(np.float32)


def reference_person_mask(image: Image.Image, extractor: Optional[TransformersPersonMaskExtractor]) -> np.ndarray:
    if extractor is None:
        return _heuristic_foreground_mask(image)

    try:
        mask = extractor.extract(image)
    except Exception:
        return _heuristic_foreground_mask(image)

    if mask is None:
        return _heuristic_foreground_mask(image)

    return mask.astype(np.float32)
