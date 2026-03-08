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
            from controlnet_aux.open_pose.body import Body
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "OpenPose feature requires 'controlnet_aux' and 'huggingface_hub'. "
                "Install them with: pip install controlnet-aux huggingface_hub"
            ) from exc

        body_model_path = self._resolve_body_checkpoint(hf_hub_download)
        detector = OpenposeDetector(Body(body_model_path), hand_estimation=None, face_estimation=None)
        if getattr(detector, "body_estimation", None) is not None and hasattr(detector.body_estimation, "to"):
            detector.body_estimation.to(self.device)
        self._detector = detector
        return detector

    def _resolve_body_checkpoint(self, hf_hub_download) -> str:
        detector_dir = self.model_dir.strip()
        if detector_dir:
            local_path = Path(detector_dir).expanduser()
            local_checkpoint = self._find_local_body_checkpoint(local_path)
            if local_checkpoint is not None:
                return str(local_checkpoint)

        source = self.pretrained_model_name_or_path.strip() or "lllyasviel/Annotators"
        return hf_hub_download(source, "body_pose_model.pth")

    @staticmethod
    def _find_local_body_checkpoint(path: Path) -> Optional[Path]:
        if path.is_file():
            return path if path.name == "body_pose_model.pth" else None

        if not path.is_dir():
            return None

        candidates = [
            path / "body_pose_model.pth",
            path / "annotator" / "ckpts" / "body_pose_model.pth",
            path / "ckpts" / "body_pose_model.pth",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None
