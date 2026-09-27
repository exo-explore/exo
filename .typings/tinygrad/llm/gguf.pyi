from tinygrad.tensor import Tensor

def ggml_data_to_tensor(
    tensor: Tensor, element_count: int, ggml_type: int
) -> Tensor: ...
