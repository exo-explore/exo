from vllm.triton_utils import tl as tl, triton as triton

@triton.jit
def fast_exp(x): ...
