from tinygrad import Tensor
from exo.worker.engines.tinygrad.layers.attention import linear_forward

def swiglu_mlp(
    x: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
) -> Tensor:
    gate = linear_forward(x, gate_proj)
    up = linear_forward(x, up_proj)
    activated = gate.silu() * up
    return linear_forward(activated, down_proj)
