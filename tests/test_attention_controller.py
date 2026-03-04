import torch

from teleportraits.attention import AttentionWindow, MaskGuidedAttentionController


def test_controller_token_mask_resizing() -> None:
    controller = MaskGuidedAttentionController(window=AttentionWindow(0, 10))
    mask = torch.zeros((8, 8), dtype=torch.float32)
    mask[2:6, 2:6] = 1.0
    controller.set_reference_mask(mask)

    tokens = controller.token_mask(
        spatial_hw=(4, 4),
        seq_len=16,
        batch=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert tokens.shape == (1, 16, 1)
    assert float(tokens.max()) <= 1.0
    assert float(tokens.min()) >= 0.0


def test_capture_and_inject() -> None:
    controller = MaskGuidedAttentionController(window=AttentionWindow(0, 10))
    controller.set_reference_mask(torch.ones((8, 8), dtype=torch.float32))
    controller.set_step(step_index=3, timestep=100)

    key = torch.ones((1, 16, 32), dtype=torch.float32)
    value = torch.ones((1, 16, 32), dtype=torch.float32) * 2

    controller.set_mode(controller.MODE_CAPTURE)
    controller.capture("layer", key, value, spatial_hw=(4, 4))

    controller.set_mode(controller.MODE_INJECT)
    injected_k, injected_v = controller.inject("layer", key, value)

    assert injected_k.shape[1] == 32
    assert injected_v.shape[1] == 32
