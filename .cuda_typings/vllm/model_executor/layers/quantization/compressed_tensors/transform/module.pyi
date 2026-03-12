import torch
from collections.abc import Callable as Callable
from compressed_tensors.transform import (
    TransformArgs as TransformArgs,
    TransformScheme as TransformScheme,
)
from torch import Tensor as Tensor
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import LinearBase as LinearBase
from vllm.model_executor.layers.quantization.compressed_tensors.transform.utils import (
    TransformTuple as TransformTuple,
)
from vllm.model_executor.layers.utils import (
    dispatch_unquantized_gemm as dispatch_unquantized_gemm,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.parameter import SharedWeightParameter as SharedWeightParameter

class HadamardTransform(torch.nn.Module):
    transforms: dict[int, TransformTuple]
    weight: SharedWeightParameter
    scales: dict[int, float]
    def __init__(
        self,
        transforms: dict[int, TransformTuple],
        layer: torch.nn.Module,
        weight_loader: Callable,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
    ) -> None: ...
    def process_weights_after_loading(self) -> None: ...
    def forward(self, value: Tensor, part_id: int = 0) -> Tensor: ...
