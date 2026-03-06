from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


def normalize_depth_for_control(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        return np.zeros_like(depth, dtype=np.float32)

    valid = depth[finite_mask]
    lo = float(np.percentile(valid, 2.0))
    hi = float(np.percentile(valid, 98.0))
    if hi <= lo + 1e-8:
        lo = float(valid.min())
        hi = float(valid.max())

    if hi <= lo + 1e-8:
        norm = np.zeros_like(depth, dtype=np.float32)
    else:
        norm = (depth - lo) / (hi - lo)
        norm = np.clip(norm, 0.0, 1.0).astype(np.float32)

    norm = np.where(finite_mask, norm, 0.0).astype(np.float32)
    return norm


def depth_to_control_image(depth_norm: np.ndarray) -> Image.Image:
    arr = np.clip(depth_norm * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


@dataclass
class MogeDepthMapExtractor:
    pretrained_model_name_or_path: str = "Ruicheng/moge-2-vitl-normal"
    checkpoint_dir: str = "./pretrained/moge"
    model_version: str = "v2"
    device: str = "cuda"
    use_fp16: bool = True
    conda_env: str = ""

    def extract(self, image: Image.Image) -> np.ndarray:
        vendor_root = Path(__file__).resolve().parent / "third_party"
        moge_pkg_dir = vendor_root / "moge"
        if not moge_pkg_dir.exists():
            raise FileNotFoundError(f"Vendored MoGe package not found: {moge_pkg_dir}")
        model_source = self._resolve_model_source()

        script = """
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

vendor_root = Path(sys.argv[1]).resolve()
input_path = Path(sys.argv[2]).resolve()
output_path = Path(sys.argv[3]).resolve()
model_name = sys.argv[4]
model_version = sys.argv[5]
device_name = sys.argv[6]
use_fp16 = sys.argv[7] == "1"
resolution_level = int(sys.argv[8])

if str(vendor_root) not in sys.path:
    sys.path.insert(0, str(vendor_root))

from moge.model import import_model_class_by_version

device = torch.device(device_name)
model_cls = import_model_class_by_version(model_version)
model = model_cls.from_pretrained(model_name).to(device).eval()

effective_fp16 = use_fp16 and device.type == "cuda"
if effective_fp16:
    model.half()

rgb = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.float32) / 255.0
image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device=device, dtype=torch.float32)

output = model.infer(
    image_tensor,
    resolution_level=resolution_level,
    use_fp16=effective_fp16,
)
depth = output["depth"]
if depth.ndim == 3:
    depth = depth.squeeze(0)
depth = depth.float().cpu().numpy().astype(np.float32)
np.save(output_path, depth)
"""

        with tempfile.TemporaryDirectory(prefix="teleportraits_moge_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "scene.png"
            depth_path = tmp_path / "depth.npy"
            image.convert("RGB").save(image_path)

            if self.conda_env.strip():
                cmd = [
                    "conda",
                    "run",
                    "-n",
                    self.conda_env.strip(),
                    "python",
                    "-c",
                    script,
                    str(vendor_root),
                    str(image_path),
                    str(depth_path),
                    model_source,
                    self.model_version,
                    self.device,
                    "1" if self.use_fp16 else "0",
                    "9",
                ]
            else:
                cmd = [
                    sys.executable,
                    "-c",
                    script,
                    str(vendor_root),
                    str(image_path),
                    str(depth_path),
                    model_source,
                    self.model_version,
                    self.device,
                    "1" if self.use_fp16 else "0",
                    "9",
                ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                stderr = proc.stderr.strip()
                stdout = proc.stdout.strip()
                details = stderr or stdout or "unknown error"
                raise RuntimeError(f"MoGe depth extraction failed. Details: {details}")

            depth = np.load(depth_path).astype(np.float32)

        return normalize_depth_for_control(depth)

    def _resolve_model_source(self) -> str:
        checkpoint_dir = self.checkpoint_dir.strip()
        if not checkpoint_dir:
            return self.pretrained_model_name_or_path

        checkpoint_path = Path(checkpoint_dir).expanduser()
        if checkpoint_path.is_file():
            return str(checkpoint_path.resolve())

        if checkpoint_path.is_dir():
            candidates = [
                checkpoint_path / "model.pt",
                checkpoint_path / self.model_version / "model.pt",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate.resolve())
            raise FileNotFoundError(
                f"MoGe checkpoint not found in {checkpoint_path}. "
                "Expected model.pt or <version>/model.pt."
            )

        return self.pretrained_model_name_or_path
