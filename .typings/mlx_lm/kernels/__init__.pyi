"""Type stubs for mlx_lm.kernels (the parts exo uses)"""

import mlx.core as mx
import mlx.nn as nn

def pre_norm(stream: mx.array, norm: nn.RMSNorm, *, dtype: mx.Dtype) -> mx.array: ...
def norm_add_norm(
    branch: mx.array,
    stream: mx.array,
    *,
    alpha: mx.array,
    beta: mx.array,
    post_weight: mx.array | None,
    post_eps: float,
    next_norm: nn.RMSNorm | None,
) -> tuple[mx.array, mx.array | None]: ...
