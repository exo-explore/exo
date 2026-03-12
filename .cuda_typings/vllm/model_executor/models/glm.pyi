import abc
from .utils import PPMissingLayer as PPMissingLayer
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM as LlamaForCausalLM

class GlmForCausalLM(LlamaForCausalLM, metaclass=abc.ABCMeta):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
