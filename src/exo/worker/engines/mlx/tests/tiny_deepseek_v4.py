# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
"""A tiny randomly initialized DeepSeek V4 for tests: 8-bit dense weights,
mxfp4 experts, uncompressed, ratio-4 and ratio-128 layers."""

from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import deepseek_v4 as dv4
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

VOCAB_SIZE = 512
SLIDING_WINDOW = 16
CONFIG: dict[str, object] = {
    "model_type": "deepseek_v4",
    "vocab_size": VOCAB_SIZE,
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "q_lora_rank": 128,
    "o_lora_rank": 64,
    "o_groups": 4,
    "head_dim": 128,
    "qk_rope_head_dim": 64,
    "sliding_window": SLIDING_WINDOW,
    "compress_ratios": [0, 4, 128, 4],
    "index_n_heads": 4,
    "index_head_dim": 64,
    "index_topk": 64,
    "compress_rope_theta": 160000.0,
    "moe_intermediate_size": 64,
    "n_routed_experts": 32,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "num_hash_layers": 1,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "num_nextn_predict_layers": 0,
    "max_position_embeddings": 4096,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "rms_norm_eps": 1e-6,
}


def tiny_model(index_topk: int = 64) -> dv4.Model:
    mx.random.seed(0)
    args = dv4.ModelArgs.from_dict({**CONFIG, "index_topk": index_topk})
    model = dv4.Model(args)
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda _, module: type(module) is nn.Linear,
    )
    keep_fp32 = model.cast_predicate
    params = []
    for path, leaf in tree_flatten(model.parameters()):
        value = cast(mx.array, leaf)
        if path.endswith("tid2eid"):
            value = mx.random.randint(0, args.n_routed_experts, value.shape)
        elif mx.issubdtype(value.dtype, mx.floating):
            value = mx.random.normal(value.shape) * 0.05
            if path.endswith("norm.weight"):
                value = value + 1.0
            if keep_fp32(path):
                value = value.astype(mx.float32)
            else:
                value = value.astype(mx.bfloat16)
        params.append((path, value))
    model.update(tree_unflatten(params))
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) and module.mode == "affine":
            out_dims = module.weight.shape[0]
            in_dims = module.scales.shape[1] * module.group_size
            weight = mx.random.normal((out_dims, in_dims)) * 0.05
            quantized = mx.quantize(weight.astype(mx.bfloat16), group_size=64, bits=8)
            module.weight, module.scales, module.biases = quantized
        elif isinstance(module, QuantizedSwitchLinear):
            experts, out_dims, _ = module.weight.shape
            in_dims = module.scales.shape[-1] * module.group_size
            weight = mx.random.normal((experts, out_dims, in_dims)) * 0.05
            quantized = mx.quantize(weight, group_size=32, bits=4, mode="mxfp4")
            module.weight, module.scales = quantized[0], quantized[1]
    mx.eval(model.parameters())
    return model
