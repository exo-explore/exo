from vllm.triton_utils import tl as tl, triton as triton

def rms_norm_gated(
    x,
    weight,
    bias,
    z=None,
    eps: float = 1e-06,
    group_size=None,
    norm_before_gate: bool = True,
): ...
