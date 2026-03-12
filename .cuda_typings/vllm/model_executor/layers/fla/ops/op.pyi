from .utils import is_gather_supported as is_gather_supported
from _typeshed import Incomplete
from vllm.triton_utils import tl as tl, tldevice as tldevice, triton as triton

exp: Incomplete
log: Incomplete
log2: Incomplete

@triton.jit
def gather(src, index, axis, _builder=None) -> None: ...

gather: Incomplete
make_tensor_descriptor: Incomplete
