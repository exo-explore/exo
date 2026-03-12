from _typeshed import Incomplete
from typing import Any
from vllm.logger import init_logger as init_logger

logger: Incomplete

class ModelArchitectureConfig:
    architectures: list[str] | None
    model_type: str
    text_model_type: str | None
    hidden_size: int
    total_num_hidden_layers: int
    total_num_attention_heads: int
    head_size: int
    vocab_size: int
    total_num_kv_heads: int
    num_experts: int
    quantization_config: dict[str, Any] | None
    is_deepseek_mla: bool
    derived_max_model_len_and_key: tuple[float, str | None]
