from typing import Optional

import mlx.core as mx

def compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array: ...
def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = ...,
    mask: Optional[mx.array] = ...,
    use_kernel: bool = ...,
) -> tuple[mx.array, mx.array]: ...
def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = ...,
    mask: Optional[mx.array] = ...,
) -> tuple[mx.array, mx.array]: ...
def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = ...,
) -> tuple[mx.array, mx.array]: ...
