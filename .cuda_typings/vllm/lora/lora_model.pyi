import torch
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.lora.lora_weights import LoRALayerWeights as LoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper as PEFTHelper
from vllm.lora.utils import (
    get_lora_id as get_lora_id,
    is_base_embedding_weights as is_base_embedding_weights,
    parse_fine_tuned_lora_name as parse_fine_tuned_lora_name,
)
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig as TensorizerConfig,
)
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available

logger: Incomplete

class LoRAModel:
    id: Incomplete
    rank: Incomplete
    loras: dict[str, LoRALayerWeights]
    def __init__(
        self, lora_model_id: int, rank: int, loras: dict[str, LoRALayerWeights]
    ) -> None: ...
    def clone(self, lora_model_id: int) -> LoRAModel: ...
    def get_lora(self, module_name: str) -> LoRALayerWeights | None: ...
    def check_lora_name(self, lora_name: str) -> bool: ...
    @classmethod
    def from_lora_tensors(
        cls,
        lora_model_id: int,
        tensors: dict[str, torch.Tensor],
        peft_helper: PEFTHelper,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        model_vocab_size: int | None = None,
        weights_mapper: WeightsMapper | None = None,
        skip_prefixes: list[str] | None = None,
    ) -> LoRAModel: ...
    @classmethod
    def from_local_checkpoint(
        cls,
        lora_dir: str,
        expected_lora_modules: set[str],
        peft_helper: PEFTHelper,
        *,
        lora_model_id: int | None = None,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        model_vocab_size: int | None = None,
        weights_mapper: WeightsMapper | None = None,
        tensorizer_config_dict: dict | None = None,
        skip_prefixes: list[str] | None = None,
    ) -> LoRAModel: ...
