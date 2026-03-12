import abc
from .utils import PPMissingLayer as PPMissingLayer
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM as LlamaForCausalLM

class Ernie4_5ForCausalLM(LlamaForCausalLM, metaclass=abc.ABCMeta):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
