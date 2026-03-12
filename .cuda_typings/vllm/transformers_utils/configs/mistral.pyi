from _typeshed import Incomplete
from transformers import PretrainedConfig
from typing import Any
from vllm.logger import init_logger as init_logger

logger: Incomplete

def adapt_config_dict(
    config_dict: dict[str, Any], defaults: dict[str, Any]
) -> PretrainedConfig: ...
