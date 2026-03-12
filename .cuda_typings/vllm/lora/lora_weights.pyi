import torch
import torch.types
from _typeshed import Incomplete
from collections.abc import Sequence as GenericSequence
from vllm.lora.peft_helper import PEFTHelper as PEFTHelper
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available

class LoRALayerWeights:
    module_name: Incomplete
    rank: Incomplete
    lora_alpha: Incomplete
    lora_a: Incomplete
    lora_b: Incomplete
    scaling: Incomplete
    def __init__(
        self,
        module_name: str,
        rank: int,
        lora_alpha: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scaling: float | None = None,
    ) -> None: ...
    def optimize(self) -> LoRALayerWeights: ...
    @property
    def input_dim(self) -> int: ...
    @property
    def output_dim(self) -> int: ...
    @property
    def is_packed(self) -> bool: ...
    @classmethod
    def from_config(
        cls, module_name: str, peft_helper: PEFTHelper
    ) -> LoRALayerWeights: ...
    @classmethod
    def create_dummy_lora_weights(
        cls,
        module_name: str,
        input_dim: int,
        output_dim: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.types.Device,
    ) -> LoRALayerWeights: ...

class PackedLoRALayerWeights(LoRALayerWeights):
    lora_alphas: Incomplete
    scaling: Incomplete
    def __init__(
        self,
        module_name: str,
        rank: int,
        lora_alphas: list[int | None],
        lora_a: list[torch.Tensor | None],
        lora_b: list[torch.Tensor | None],
        scaling: list[float] | None = None,
    ) -> None: ...
    @classmethod
    def pack(
        cls, loras: GenericSequence[LoRALayerWeights | None]
    ) -> PackedLoRALayerWeights: ...
    @classmethod
    def pack_moe(
        cls,
        loras: GenericSequence[LoRALayerWeights | None],
        module_name: str,
        is_non_gated_moe: bool = False,
    ) -> PackedLoRALayerWeights: ...
    def optimize(self) -> PackedLoRALayerWeights: ...
    @property
    def input_dim(self) -> int: ...
    @property
    def output_dim(self) -> int: ...
    @property
    def is_packed(self) -> bool: ...
