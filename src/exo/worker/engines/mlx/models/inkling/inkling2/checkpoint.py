# type: ignore
# Vendored from ml-explore/mlx-lm @ 0c4c99192230f1943dd468d297f0b8a64ae7b7c2 (PR #2086, Inkling 2), mlx_lm/models/inkling2/checkpoint.py
# (splice.py: mlx_lm/layers/common.py). MIT licence, Copyright 2026 Apple Inc. Local change: imports only (see __init__.py).
# Copyright © 2026 Apple Inc.

import re

import mlx.core as mx
from mlx.utils import tree_unflatten

from mlx_lm.models.switch_layers import SwitchLinear

__all__ = [
    "sanitize",
]


_NVFP4_E2M1_MAX = 6.0
_NVFP4_E4M3_MAX = 448.0

_RENAMES = [
    (r"^model\.llm\.unembed\.", "lm_head."),
    (r"^model\.llm\.embed\.", "model.language_model.embed_tokens."),
    (r"^model\.llm\.", "model.language_model."),
    (r"\.attn_res\.mlp_queries\.(\d+)$", r".layers.\1.mlp_input.query"),
    (r"\.attn_res\.output_query$", ".final_input.query"),
    (r"\.attn\.wq_du\.", ".self_attn.q_proj."),
    (r"\.attn\.wk_dv\.", ".self_attn.k_proj."),
    (r"\.attn\.wv_dv\.", ".self_attn.v_proj."),
    (r"\.attn\.wr_du\.", ".self_attn.r_proj."),
    (r"\.attn\.wo_ud\.", ".self_attn.o_proj."),
    (r"\.attn\.gate_attn\.", ".self_attn.gate_proj."),
    (r"\.attn\.attn_sink_proj\.", ".self_attn.sink_proj."),
    (r"\.controllers\.attn\.", ".q_message_gate."),
    (r"\.controllers\.(k|v|mlp)\.", r".\1_message_gate."),
    (r"_message_gate\.down_weight$", "_message_gate.down_proj.weight"),
    (r"_message_gate\.forget_weight$", "_message_gate.forget_gate_proj.weight"),
    (r"_message_gate\.gain_weight$", "_message_gate.message_gate_proj.weight"),
    (r"\.attn\.", ".self_attn."),
    (r"\.attn_sconv\.", ".post_attention_convolution."),
    (r"\.mlp_sconv\.", ".post_mlp_convolution."),
    (r"sconv\.weight$", "sconv.conv1d.weight"),
    (r"convolution\.weight$", "convolution.conv1d.weight"),
    (r"sconv\.static_w$", "sconv.static_weight"),
    (r"sconv\.weight_proj$", "sconv.weight_proj.weight"),
    (r"\.mlp_norm\.", ".post_attention_layernorm."),
    (r"\.attn_norm\.", ".input_layernorm."),
    (r"\.(q|k|v)_input_norm\.", r".\1_input_layernorm."),
    (r"\.mlp\.gate\.bias$", ".mlp.gate.e_score_correction_bias"),
    (r"^model\.visual\.backbone\.ln_pre\.", "model.vision_tower.pre_layernorm."),
    (r"^model\.visual\.backbone\.", "model.vision_tower."),
    (r"^model\.visual\.proj_in\.", "model.vision_tower.merger.linear_1."),
    (r"^model\.visual\.proj_out\.", "model.vision_tower.merger.linear_2."),
    (r"embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"(vision_tower\.layers\.\d+)\.ln_1\.", r"\1.layernorm_before."),
    (r"(vision_tower\.layers\.\d+)\.ln_2\.", r"\1.layernorm_after."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2."),
    (r"(vision_tower\.layers\.\d+)\.ls_(\d)\.gamma$", r"\1.lambda_\2"),
]


def _rename(key: str) -> str:
    if m := re.match(r"model\.llm\.attn_res\.attn_queries\.(\d+)$", key):
        # Layer 0 has no attention input, so query i is for layer i + 1.
        layer = int(m.group(1)) + 1
        return f"model.language_model.layers.{layer}.attention_input.query"
    for pattern, replacement in _RENAMES:
        key = re.sub(pattern, replacement, key)
    return key


def sanitize(model, weights: dict[str, mx.array]) -> dict[str, mx.array]:
    experts = re.compile(r"^(.+\.mlp\.experts)\.(w13|w2)_weight(\w*)$")
    amax, denominator = {}, {}
    message_queries = {}
    sanitized = {}
    for k, v in weights.items():
        if k.startswith(("model.llm.", "model.visual.")):
            k = _rename(k)
            if k.endswith("conv1d.weight"):
                v = v.swapaxes(1, 2)
        if k.startswith("model.vision_tower.") and model.args.vision is None:
            continue
        if k.endswith(("cross_pass_queries", "cross_pass_kv_queries")):
            message_queries["kv" if "_kv_" in k else "q"] = v
            continue
        if k.endswith(
            (
                ".mlp.gate.weight",
                ".forget_gate_proj.weight",
                ".message_gate_proj.weight",
            )
        ):
            v = v.astype(mx.float32)
        elif m := experts.match(k):
            prefix, which, suffix = m.groups()
            name = f"{prefix}.{'gate_up_proj' if which == 'w13' else 'down_proj'}"
            if suffix == "_amax":
                amax[name] = v
            elif suffix == "_outer_scale_denominator":
                denominator[name] = v
            elif suffix == "_scale":
                sanitized[f"{name}.scales"] = v
            else:
                sanitized[f"{name}.weight"] = v.view(mx.uint32)
            continue
        sanitized[k] = v

    if message_queries:
        # The checkpoint has [query, mlp] and [key, value] rows per looped layer.
        dims = message_queries["q"].shape[-1]
        q = message_queries["q"].reshape(-1, 2, dims)
        kv = message_queries["kv"].reshape(-1, 2, dims)
        query = mx.concatenate([q[:, :1], kv, q[:, 1:]], axis=1)
        sanitized["model.language_model.message_input.query"] = query

    # Pipeline stages can miss the files of the other layers.
    for name in amax.keys() & denominator.keys():
        scale = _NVFP4_E2M1_MAX * _NVFP4_E4M3_MAX
        scale = scale / denominator[name].astype(mx.float32)
        sanitized[f"{name}.global_scale"] = amax[name].astype(mx.float32) * scale

    # mlx-lm does not read this quantization config, so set the experts here.
    modes = {(4, 16): "nvfp4", (4, 32): "mxfp4", (8, 32): "mxfp8"}
    quantized = []
    for path, module in model.named_modules():
        scales = sanitized.get(f"{path}.scales")
        if not isinstance(module, SwitchLinear) or scales is None:
            continue
        if scales.dtype != mx.uint8:
            continue
        in_dims = module.weight.shape[-1]
        group_size = in_dims // scales.shape[-1]
        bits = sanitized[f"{path}.weight"].shape[-1] * 32 // in_dims
        if (bits, group_size) not in modes:
            continue
        ql = module.to_quantized(
            group_size,
            bits,
            mode=modes[bits, group_size],
            global_scale=f"{path}.global_scale" in sanitized,
        )
        quantized.append((path, ql))
    if quantized:
        model.update_modules(tree_unflatten(quantized))
    return sanitized
