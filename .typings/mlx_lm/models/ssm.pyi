from typing import Optional, Tuple

import mlx.core as mx

def ssm_update(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = ...,
    time_step_limit: Tuple[float, float] = ...,
    mask: Optional[mx.array] = ...,
    lengths: Optional[mx.array] = ...,
) -> tuple[mx.array, mx.array]: ...
