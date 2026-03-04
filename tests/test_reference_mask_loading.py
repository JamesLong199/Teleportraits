from pathlib import Path

import numpy as np
from PIL import Image

from teleportraits.masks import load_binary_mask


def test_load_binary_mask_threshold_and_invert(tmp_path: Path) -> None:
    arr = np.array([[0, 255], [64, 200]], dtype=np.uint8)
    p = tmp_path / "mask.png"
    Image.fromarray(arr, mode="L").save(p)

    m = load_binary_mask(str(p), target_size=(2, 2), invert=False)
    assert m.shape == (2, 2)
    assert m[0, 0] == 0.0
    assert m[0, 1] == 1.0
    assert m[1, 0] == 0.0
    assert m[1, 1] == 1.0

    mi = load_binary_mask(str(p), target_size=(2, 2), invert=True)
    assert mi[0, 0] == 1.0
    assert mi[0, 1] == 0.0
