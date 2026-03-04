from pathlib import Path

import numpy as np
from PIL import Image

from teleportraits.masks import load_binary_mask


def test_foreground_mask_override_loader() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fg_mask.png"
        arr = np.zeros((6, 6), dtype=np.uint8)
        arr[2:5, 1:4] = 255
        Image.fromarray(arr, mode="L").save(p)

        mask = load_binary_mask(str(p), target_size=(6, 6), invert=False)
        assert mask.shape == (6, 6)
        assert mask[0, 0] == 0.0
        assert mask[3, 2] == 1.0
