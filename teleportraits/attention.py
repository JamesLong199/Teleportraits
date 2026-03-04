from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AttentionWindow:
    start_step: int
    end_step: int

    def contains(self, step_index: int) -> bool:
        return self.start_step <= step_index <= self.end_step


class ReferenceKVStore:
    def __init__(self) -> None:
        self._store: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def put(self, layer_name: str, timestep: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self._store[(layer_name, timestep)] = (key.detach(), value.detach())

    def get(self, layer_name: str, timestep: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self._store.get((layer_name, timestep))

    def clear(self) -> None:
        self._store.clear()


class MaskGuidedAttentionController:
    MODE_OFF = "off"
    MODE_CAPTURE = "capture"
    MODE_INJECT = "inject"

    def __init__(self, window: AttentionWindow) -> None:
        self.mode: str = self.MODE_OFF
        self.window = window
        self.current_step: int = -1
        self.current_timestep: int = -1
        self.reference_mask: Optional[torch.Tensor] = None
        self.kv_store = ReferenceKVStore()

    def clear(self) -> None:
        self.kv_store.clear()

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def set_step(self, step_index: int, timestep: int) -> None:
        self.current_step = step_index
        self.current_timestep = timestep

    def set_reference_mask(self, mask: torch.Tensor) -> None:
        if mask.ndim == 2:
            mask = mask[None, None, :, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        self.reference_mask = mask.float()

    def should_operate(self) -> bool:
        if self.mode == self.MODE_OFF:
            return False
        return self.window.contains(self.current_step)

    def token_mask(
        self,
        spatial_hw: Optional[Tuple[int, int]],
        seq_len: int,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.reference_mask is None:
            return torch.ones((batch, seq_len, 1), device=device, dtype=dtype)

        mask = self.reference_mask.to(device=device, dtype=dtype)

        if spatial_hw is None:
            h = int(round(math.sqrt(seq_len)))
            w = max(seq_len // max(h, 1), 1)
            while h * w < seq_len:
                w += 1
        else:
            h, w = spatial_hw

        resized = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        tokens = resized.flatten(start_dim=2).transpose(1, 2)

        if tokens.shape[1] > seq_len:
            tokens = tokens[:, :seq_len, :]
        elif tokens.shape[1] < seq_len:
            pad = seq_len - tokens.shape[1]
            tokens = F.pad(tokens, (0, 0, 0, pad), value=0.0)

        if tokens.shape[0] == 1 and batch > 1:
            tokens = tokens.repeat(batch, 1, 1)

        return tokens

    def capture(
        self,
        layer_name: str,
        key_cond: torch.Tensor,
        value_cond: torch.Tensor,
        spatial_hw: Optional[Tuple[int, int]],
    ) -> None:
        if not self.should_operate() or self.mode != self.MODE_CAPTURE:
            return

        mask = self.token_mask(
            spatial_hw=spatial_hw,
            seq_len=key_cond.shape[1],
            batch=key_cond.shape[0],
            device=key_cond.device,
            dtype=key_cond.dtype,
        )

        masked_key = key_cond * mask
        masked_value = value_cond * mask
        self.kv_store.put(layer_name, self.current_timestep, masked_key, masked_value)

    def inject(
        self,
        layer_name: str,
        key_cond: torch.Tensor,
        value_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.should_operate() or self.mode != self.MODE_INJECT:
            return key_cond, value_cond

        ref = self.kv_store.get(layer_name, self.current_timestep)
        if ref is None:
            return key_cond, value_cond

        ref_key, ref_value = ref
        ref_key = ref_key.to(device=key_cond.device, dtype=key_cond.dtype)
        ref_value = ref_value.to(device=value_cond.device, dtype=value_cond.dtype)

        if ref_key.shape[0] == 1 and key_cond.shape[0] > 1:
            ref_key = ref_key.repeat(key_cond.shape[0], 1, 1)
            ref_value = ref_value.repeat(value_cond.shape[0], 1, 1)

        key_cond = torch.cat([key_cond, ref_key], dim=1)
        value_cond = torch.cat([value_cond, ref_value], dim=1)
        return key_cond, value_cond


class MaskGuidedSelfAttentionProcessor:
    def __init__(self, base_processor: object, controller: MaskGuidedAttentionController, layer_name: str) -> None:
        self.base_processor = base_processor
        self.controller = controller
        self.layer_name = layer_name

    def __call__(
        self,
        attn: object,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            return self.base_processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        if not self.controller.should_operate():
            return self.base_processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        residual = hidden_states
        spatial_hw: Optional[Tuple[int, int]] = None

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            spatial_hw = (height, width)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, _, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        if batch_size % 2 != 0:
            hidden_states = _run_attention(attn, query, key, value, attention_mask)
        else:
            half = batch_size // 2
            query_u, query_c = query[:half], query[half:]
            key_u, key_c = key[:half], key[half:]
            value_u, value_c = value[:half], value[half:]

            self.controller.capture(self.layer_name, key_c, value_c, spatial_hw)
            key_c, value_c = self.controller.inject(self.layer_name, key_c, value_c)

            out_u = _run_attention(attn, query_u, key_u, value_u, None)
            out_c = _run_attention(attn, query_c, key_c, value_c, None)
            hidden_states = torch.cat([out_u, out_c], dim=0)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def _run_attention(
    attn: object,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, key.shape[1], query.shape[0])

    attention_probs = attn.get_attention_scores(query, key, attention_mask)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)
    return hidden_states


def install_mask_guided_processors(
    pipe: object,
    controller: MaskGuidedAttentionController,
    target_prefixes: Tuple[str, ...],
) -> Dict[str, object]:
    original_processors = dict(pipe.unet.attn_processors)
    updated = {}

    for name, processor in pipe.unet.attn_processors.items():
        should_patch = any(name.startswith(prefix) for prefix in target_prefixes) and ".attn1.processor" in name
        if should_patch:
            updated[name] = MaskGuidedSelfAttentionProcessor(processor, controller, name)
        else:
            updated[name] = processor

    pipe.unet.set_attn_processor(updated)
    return original_processors


def restore_processors(pipe: object, original_processors: Dict[str, object]) -> None:
    pipe.unet.set_attn_processor(original_processors)
