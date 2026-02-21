from tinygrad.tensor import Tensor

from exo.worker.engines.tinygrad.layers.attention import LinearWeight, linear_forward


def swiglu_mlp(
    x: Tensor,
    gate_proj: LinearWeight,
    up_proj: LinearWeight,
    down_proj: LinearWeight,
) -> Tensor:
    gate = linear_forward(x, gate_proj)
    up = linear_forward(x, up_proj)
    activated = gate.silu() * up
    return linear_forward(activated, down_proj)
