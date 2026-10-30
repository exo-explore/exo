"""Type stubs for mlx_lm.kernels"""

import mlx.core as mx

def expert_routing(
    logits: mx.array,
    expert_bias: mx.array,
    *,
    top_k: int,
    multiplier: float,
    score_eps: float,
    route_scale: float,
) -> list[mx.array]: ...
