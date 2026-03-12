from _typeshed import Incomplete
from dataclasses import dataclass, field
from typing import Literal
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig as TensorizerConfig,
)

logger: Incomplete

@dataclass
class PEFTHelper:
    r: int
    lora_alpha: int
    target_modules: list[str] | str
    bias: Literal["none"] = field(default="none")
    modules_to_save: list[str] | None = field(default=None)
    use_rslora: bool = field(default=False)
    use_dora: bool = field(default=False)
    vllm_lora_scaling_factor: float = field(default=1.0)
    vllm_max_position_embeddings: int | None = field(default=False)
    def __post_init__(self) -> None: ...
    @classmethod
    def from_dict(cls, config_dict: dict) -> PEFTHelper: ...
    @classmethod
    def from_local_dir(
        cls,
        lora_path: str,
        max_position_embeddings: int | None,
        tensorizer_config_dict: dict | None = None,
    ) -> PEFTHelper: ...
    def validate_legal(self, lora_config: LoRAConfig) -> None: ...
