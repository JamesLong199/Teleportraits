from __future__ import annotations

import numpy as np
from PIL import Image


def load_binary_mask(path: str, target_size: tuple[int, int], invert: bool) -> np.ndarray:
    mask_img = Image.open(path).convert("L").resize(target_size, Image.Resampling.NEAREST)
    mask_np = np.asarray(mask_img, dtype=np.float32) / 255.0
    if invert:
        mask_np = 1.0 - mask_np
    return (mask_np > 0.5).astype(np.float32)
