from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class OpenposeMapExtractor:
    pretrained_model_name_or_path: str = "lllyasviel/Annotators"
    model_dir: str = ""
    device: str = "cuda"

    def __post_init__(self) -> None:
        self._detector = None

    def extract(self, image: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        detector = self._get_detector()
        image_rgb = image.convert("RGB")
        try:
            pose_map = detector(image_rgb, hand_and_face=False)
        except TypeError:
            pose_map = detector(image_rgb)

        if isinstance(pose_map, Image.Image):
            out = pose_map.convert("RGB")
        else:
            out = Image.fromarray(np.asarray(pose_map, dtype=np.uint8)).convert("RGB")

        if target_size is not None and out.size != target_size:
            out = out.resize(target_size, Image.Resampling.BICUBIC)
        return out

    def _get_detector(self):
        if self._detector is not None:
            return self._detector

        try:
            from controlnet_aux import OpenposeDetector
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "OpenPose feature requires 'controlnet_aux'. "
                "Install it with: pip install controlnet-aux"
            ) from exc

        source = self.pretrained_model_name_or_path
        detector_dir = self.model_dir.strip()
        if detector_dir:
            local_dir = Path(detector_dir).expanduser()
            if local_dir.exists():
                source = str(local_dir.resolve())

        detector = OpenposeDetector.from_pretrained(source)
        self._detector = detector
        return detector
