import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from dataclasses import dataclass
from torch.nn.parameter import Parameter
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
    QuantizeMethodBase as QuantizeMethodBase,
    method_has_implemented_embedding as method_has_implemented_embedding,
)
from vllm.model_executor.layers.utils import (
    dispatch_unquantized_gemm as dispatch_unquantized_gemm,
)
from vllm.model_executor.parameter import BasevLLMParameter as BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform

DEFAULT_VOCAB_PADDING_SIZE: int

class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    def embedding(
        self, layer: torch.nn.Module, input_: torch.Tensor
    ) -> torch.Tensor: ...

def pad_vocab_size(vocab_size: int, pad_to: int = ...) -> int: ...
def vocab_range_from_per_partition_vocab_size(
    per_partition_vocab_size: int, rank: int, offset: int = 0
) -> Sequence[int]: ...
def vocab_range_from_global_vocab_size(
    global_vocab_size: int, rank: int, world_size: int, offset: int = 0
) -> Sequence[int]: ...
@dataclass
class VocabParallelEmbeddingShardIndices:
    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int
    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int
    @property
    def num_org_elements(self) -> int: ...
    @property
    def num_added_elements(self) -> int: ...
    @property
    def num_org_elements_padded(self) -> int: ...
    @property
    def num_added_elements_padded(self) -> int: ...
    @property
    def num_org_vocab_padding(self) -> int: ...
    @property
    def num_added_vocab_padding(self) -> int: ...
    @property
    def num_elements_padded(self) -> int: ...
    def __post_init__(self) -> None: ...

def get_masked_input_and_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...

class VocabParallelEmbedding(CustomOp):
    tp_size: Incomplete
    num_embeddings: Incomplete
    padding_size: Incomplete
    org_vocab_size: Incomplete
    org_vocab_size_padded: Incomplete
    num_embeddings_padded: Incomplete
    shard_indices: Incomplete
    embedding_dim: Incomplete
    quant_method: QuantizeMethodBase
    num_added_embeddings: Incomplete
    num_embeddings_per_partition: Incomplete
    num_org_embeddings_per_partition: Incomplete
    num_added_embeddings_per_partition: Incomplete
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = ...,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    def get_sharded_to_full_mapping(self) -> list[int] | None: ...
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor): ...
    def forward_native(self, input_): ...
    def forward_cuda(self, input_): ...
    def extra_repr(self) -> str: ...

class ParallelLMHead(VocabParallelEmbedding):
    quant_config: Incomplete
    bias: Incomplete
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = ...,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None: ...
    weight: Incomplete
    def tie_weights(self, embed_tokens: VocabParallelEmbedding): ...
    def forward(self, input_) -> None: ...
