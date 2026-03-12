import abc
from _typeshed import Incomplete
from vllm.model_executor.models.llama import LlamaForCausalLM as LlamaForCausalLM

class Phi3ForCausalLM(LlamaForCausalLM, metaclass=abc.ABCMeta):
    packed_modules_mapping: Incomplete
