from tinygrad.tensor import Tensor

from exo.worker.engines.tinygrad.layers.attention import LinearWeight, linear_forward


def swiglu_mlp(
    x: Tensor,
    gate_up_proj: LinearWeight,
    down_proj: LinearWeight,
) -> Tensor:
    gate_up = linear_forward(x, gate_up_proj)
    half = gate_up.shape[-1] // 2
    gate = gate_up[..., :half]
    up = gate_up[..., half:]
    activated = gate.silu() * up
    return linear_forward(activated, down_proj)
