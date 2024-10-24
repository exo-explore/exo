from dataclasses import dataclass
from importlib.resources import contents
import ggml
import numpy as np
from typing import Optional

@dataclass
class OpsOutput:
    out_tensor: ggml.ggml_tensor_p
    out_shape: tuple

class Linear:
    def __init__(self, ctx: ggml.ggml_context_p, graph: ggml.ggml_cgraph_p, w: np.ndarray, bias: Optional[np.ndarray], backend: ggml.ggml_backend_t):
        self.ctx = ctx
        self.gf = graph
        self.w = w
        self.backend = backend
        # Tensors should be inserted in row column order is
        self.w_tensor = ggml.ggml_new_tensor_2d(self.ctx, ggml.GGML_TYPE_F32, self.w.shape[1], self.w.shape[0])
        if bias is not None:
            self.bias = bias
            self.bias_tensor = ggml.ggml_new_tensor_2d(self.ctx, ggml.GGML_TYPE_F32, self.bias.shape[0], self.bias.shape[1])

    def forward(self, x: np.ndarray) -> OpsOutput:
        # TODO: better check as matmul in ggml internally transposes
        # n x n matrix is inversed therefore a transpose is required here if they're equal
        if self.w.shape[0] != x.shape[0]:
            x = x.T
        x_tensor = ggml.ggml_new_tensor_2d(self.ctx, ggml.GGML_TYPE_F32, x.shape[1], x.shape[0])
        if hasattr(self, "bias") and hasattr(self, "bias_tensor"):
            x2_tensor = ggml.ggml_mul_mat(self.ctx, self.w_tensor, x_tensor)
            forward_tensor = ggml.ggml_add(self.ctx, x2_tensor, self.bias_tensor)
        else:
            forward_tensor = ggml.ggml_mul_mat(self.ctx, self.w_tensor, x_tensor)

        ggml.ggml_build_forward_expand(self.gf, forward_tensor)
        buffer = ggml.ggml_backend_alloc_ctx_tensors(self.ctx, self.backend)
        ggml.ggml_backend_tensor_set(self.w_tensor, self.w.ctypes.data, 0, ggml.ggml_nbytes(self.w_tensor))
        ggml.ggml_backend_tensor_set(x_tensor, x.ctypes.data, 0,ggml.ggml_nbytes(x_tensor))

        if hasattr(self, "bias") and hasattr(self, "bias_tensor"):
            ggml.ggml_backend_tensor_set(self.bias_tensor, self.bias.ctypes.data, 0, ggml.ggml_nbytes(self.bias_tensor))

        out_shape = (forward_tensor.contents.ne[0], forward_tensor.contents.ne[1])
        print(out_shape)
        return OpsOutput(out_tensor=forward_tensor, out_shape=out_shape)

# Quick Test to see if ops work
if __name__== "__main__":
    backend = ggml.ggml_backend_cpu_init()
    params = ggml.ggml_init_params(
                mem_size=ggml.ggml_tensor_overhead() * 6 + ggml.ggml_graph_overhead() + 10000,
                no_alloc=True,
            )
    ctx = ggml.ggml_init(params)
    gf = ggml.ggml_new_graph(ctx)
    x = np.array([[4, 5]], dtype=np.float32)
    w = np.array([[2, 3]], dtype=np.float32)

    linear = Linear(ctx, gf, w,None, backend)
    output_op = linear.forward(x)
    out_array = np.empty(output_op.out_shape, dtype=np.float32)
    ggml.ggml_backend_graph_compute(backend, gf)
    ggml.ggml_backend_tensor_get(output_op.out_tensor, out_array.ctypes.data,0,ggml.ggml_nbytes(output_op.out_tensor))
    print(out_array)
