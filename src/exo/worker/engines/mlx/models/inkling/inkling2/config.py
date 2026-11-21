# type: ignore
# Vendored from ml-explore/mlx-lm @ 0c4c99192230f1943dd468d297f0b8a64ae7b7c2 (PR #2086, Inkling 2), mlx_lm/models/inkling2/config.py
# (splice.py: mlx_lm/layers/common.py). MIT licence, Copyright 2026 Apple Inc. Local change: imports only (see __init__.py).
# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
    "TextArgs",
    "VisionArgs",
]


@dataclass
class TextArgs(BaseModelArgs):
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    sliding_window_size: int
    local_layer_ids: list[int]
    d_rel: int
    rel_extent: int
    log_scaling_n_floor: int
    log_scaling_alpha: float
    rms_norm_eps: float
    qk_l2_clip_threshold: float
    rel_l2_clip_threshold: float
    sconv_kernel_size: int
    n_layers_per_attn_res_block: int
    n_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    mlp_gate_softcap: float
    mlp_up_softcap: float
    route_scale: float
    final_logit_softcapping: float
    loop_config: dict | None = None

    def __post_init__(self):
        loop = self.loop_config or {}
        self.loop_core = range(loop.get("core_start", 0), loop.get("core_end", 0))


@dataclass
class VisionArgs(BaseModelArgs):
    hidden_size: int = 1024
    num_hidden_layers: int = 23
    num_attention_heads: int = 16
    num_channels: int = 3
    native_patch_size: int = 20
    adapter_merge_size: int = 2
    posemb_grid_size: int = 32
    projection_hidden_dim: int = 4096
    projector_bias: bool = False
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = int(self.hidden_size * self.mlp_ratio)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vision_config: dict | None = None
    image_token_id: int = 200054

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
        self.vision = None
        if self.vision_config:
            self.vision = VisionArgs.from_dict(self.vision_config)
