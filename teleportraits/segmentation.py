from __future__ import annotations

from dataclasses import dataclass
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
