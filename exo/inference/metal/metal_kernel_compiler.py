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

    def _convert_op_to_metal(self, op: Any) -> KernelOperation:
        """Convert TinyGrad operations to Metal operations"""
        if hasattr(op, "op"):
            op_type = op.op
        else:
            op_type = type(op).__name__

        op_converters = {
            # Basic arithmetic
            "add": lambda x: KernelOperation("add", [x.arg1, x.arg2], x.out),
            "sub": lambda x: KernelOperation("sub", [x.arg1, x.arg2], x.out),
            "mul": lambda x: KernelOperation("mul", [x.arg1, x.arg2], x.out),
            "div": lambda x: KernelOperation("div", [x.arg1, x.arg2], x.out),
            
            # Activation functions
            "relu": lambda x: KernelOperation("relu", [x.arg1], x.out),
            "tanh": lambda x: KernelOperation("tanh", [x.arg1], x.out),
            "sigmoid": lambda x: KernelOperation("sigmoid", [x.arg1], x.out),
            
            # Memory operations
            "load": lambda x: KernelOperation("load", [x.arg1], x.out, {"offset": x.offset if hasattr(x, "offset") else 0}),
            "store": lambda x: KernelOperation("store", [x.arg1], x.out, {"offset": x.offset if hasattr(x, "offset") else 0}),
            
            # Matrix operations
            "matmul": lambda x: KernelOperation("matmul", [x.arg1, x.arg2], x.out, {
                "M": x.M if hasattr(x, "M") else None,
                "N": x.N if hasattr(x, "N") else None,
                "K": x.K if hasattr(x, "K") else None
            }),
            
            # Reduction operations
            "sum": lambda x: KernelOperation("reduce_sum", [x.arg1], x.out, {"axis": x.axis if hasattr(x, "axis") else None}),
            "max": lambda x: KernelOperation("reduce_max", [x.arg1], x.out, {"axis": x.axis if hasattr(x, "axis") else None}),
        }

        if op_type in op_converters:
            return op_converters[op_type](op)
        else:
            return KernelOperation("unknown", [], f"unknown_{op_type}")
