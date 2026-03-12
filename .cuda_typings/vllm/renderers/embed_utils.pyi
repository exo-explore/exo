import torch
from vllm.config import ModelConfig as ModelConfig
from vllm.exceptions import VLLMValidationError as VLLMValidationError

def safe_load_prompt_embeds(
    model_config: ModelConfig, embed: bytes
) -> torch.Tensor: ...
