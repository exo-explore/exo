from typing import Dict, List, Any , Tuple
import numpy as np
from tinygrad import dtypes
from .metal_model_shard import MetalModelShard, MetalKernelMetadata
from .utils import KernelOperation , Linearizer,Kernel

class MetalKernelCompiler:
    def __init__(self):
        self.kernels: Dict[str, str] = {}
        self.kernel_order: List[str] = []
        self.kernel_metadata: Dict[str, MetalKernelMetadata] = {}

    def _generate_buffer_bindings(self, num_inputs: int) -> str:
        bindings = []
        for i in range(num_inputs):
            bindings.append(f"device const float* input{i} [[buffer({i})]]")
        bindings.append(f"device float* output [[buffer({num_inputs})]]")
        return ",\n            ".join(bindings)

    def _convert_shape_to_metal(self, shape: List[int]) -> str:
        return f"uint3({', '.join(str(x) for x in shape)})"

    def _generate_index_calculation(self, dims: int) -> str:
        if dims == 1:
            return "gid.x"
        elif dims == 2:
            return "gid.x + grid_size.x * gid.y"
        else:
            return "gid.x + grid_size.x * (gid.y + grid_size.y * gid.z)"

    def _convert_dtype_to_metal(self, dtype: Any) -> str:
        dtype_map = {
            dtypes.float32: "float",
            dtypes.float16: "half",
            dtypes.int32: "int",
            dtypes.int16: "short",
            dtypes.int8: "char",
            dtypes.uint32: "uint",
            dtypes.uint16: "ushort",
            dtypes.uint8: "uchar",
        }
        return dtype_map.get(dtype, "float")
