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

    def _generate_metal_op_code(self, op: KernelOperation) -> str:
        """Generate Metal code for a single operation"""
        metal_op_templates = {
            "add": "{output} = {inputs[0]} + {inputs[1]};",
            "sub": "{output} = {inputs[0]} - {inputs[1]};",
            "mul": "{output} = {inputs[0]} * {inputs[1]};",
            "div": "{output} = {inputs[0]} / {inputs[1]};",
            "relu": "{output} = max({inputs[0]}, 0.0f);",
            "tanh": "{output} = tanh({inputs[0]});",
            "sigmoid": "{output} = 1.0f / (1.0f + exp(-{inputs[0]}));",
            "load": "{output} = input_buffer[{attributes[offset]} + idx];",
            "store": "output_buffer[{attributes[offset]} + idx] = {inputs[0]};",
            "matmul": """
                float sum = 0.0f;
                for (int k = 0; k < {attributes[K]}; k++) {{
                    sum += {inputs[0]}[i * {attributes[K]} + k] * {inputs[1]}[k * {attributes[N]} + j];
                }}
                {output} = sum;
            """,
            "reduce_sum": """
                float sum = 0.0f;
                for (int i = 0; i < {attributes[axis]}; i++) {{
                    sum += {inputs[0]}[i];
                }}
                {output} = sum;
            """,
            "reduce_max": """
                float max_val = -INFINITY;
                for (int i = 0; i < {attributes[axis]}; i++) {{
                    max_val = max(max_val, {inputs[0]}[i]);
                }}
                {output} = max_val;
            """
        }

        template = metal_op_templates.get(op.op_type, "// Unknown operation: {op_type}")
        return template.format(
            output=op.output,
            inputs=op.inputs,
            attributes=op.attributes or {},
            op_type=op.op_type
        )

    def compile_kernel_to_metal(self, kernel: Kernel) -> Tuple[str, MetalKernelMetadata]:
        """Compile a TinyGrad kernel to Metal shader code"""
        
        # Analyze kernel operations
        lin = Linearizer(kernel.name, kernel.uops, {})
        ops = []
        for buf in lin.bufs:
            for op in buf.ops:
                metal_op = self._convert_op_to_metal(op)
                ops.append(metal_op)

        # Create metadata
        metadata = MetalKernelMetadata(
            name=kernel.name,
            input_shapes=kernel.input_shapes,
            output_shape=kernel.output_shape,
            work_group_size=kernel.work_group_size,
            global_size=kernel.global_size,
            buffer_sizes=[np.prod(shape) for shape in kernel.input_shapes],
            op_sequence=ops
        )

        # Generate Metal code
        metal_code = f"""
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void {kernel.name}(
            {self._generate_buffer_bindings(len(metadata.input_shapes))},
            uint3 gid [[thread_position_in_grid]],
            uint3 grid_size [[threads_per_grid]])
        {{
            // Calculate global index
            const uint idx = {self._generate_index_calculation(len(kernel.output_shape))};
            if (idx >= {np.prod(kernel.output_shape)}) return;
            
            // Declare temporary variables
            {self._generate_temp_declarations(ops)}
            
            // Kernel computation
            {self._generate_computation_code(ops)}
        }}
        """
        
        return metal_code, metadata
