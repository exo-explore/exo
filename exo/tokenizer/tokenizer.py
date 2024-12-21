from abc import ABC, abstractmethod
from typing import List, Dict, Any
import importlib
class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    @abstractmethod
    def apply_chat_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool = True, **kwargs) -> str:
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        pass

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        pass

# TOKENIZER_CLASSES = {
#     ### llama
#     "llama-3.2-1b": "LlamaTokenizer",
#     "llama-3.2-3b": "LlamaTokenizer",
#     "llama-3.1-8b": "LlamaTokenizer",
#     "llama-3.1-70b": "LlamaTokenizer",
#     "llama-3.1-70b-bf16": "LlamaTokenizer",
#     "llama-3-8b": "LlamaTokenizer",
#     "llama-3-70b": "LlamaTokenizer",
#     "llama-3.1-405b": "LlamaTokenizer",
#     "llama-3.1-405b-8bit": "LlamaTokenizer",
#     ### mistral
#     "mistral-nemo": "MistralTokenizer",
#     "mistral-large": "MistralTokenizer",
#     ### deepseek
#     "deepseek-coder-v2-lite": "DeepSeekTokenizer",
#     "deepseek-coder-v2.5": "DeepSeekTokenizer",
#     ### llava
#     "llava-1.5-7b-hf": "LlavaTokenizer",
#     ### qwen
#     "qwen-2.5-0.5b": "QwenTokenizer",
#     "qwen-2.5-coder-1.5b": "QwenTokenizer",
#     "qwen-2.5-coder-3b": "QwenTokenizer",
#     "qwen-2.5-coder-7b": "QwenTokenizer",
#     "qwen-2.5-coder-14b": "QwenTokenizer",
#     "qwen-2.5-coder-32b": "QwenTokenizer",
#     "qwen-2.5-7b": "QwenTokenizer",
#     "qwen-2.5-math-7b": "QwenTokenizer",
#     "qwen-2.5-14b": "QwenTokenizer",
#     "qwen-2.5-72b": "QwenTokenizer",
#     "qwen-2.5-math-72b": "QwenTokenizer",
#     ### nemotron
#     "nemotron-70b": "NemotronTokenizer",
#     "nemotron-70b-bf16": "NemotronTokenizer",
#     ### gemma
#     "gemma2-9b": "GemmaTokenizer",
#     "gemma2-27b": "GemmaTokenizer",
#     ### dummy
#     "dummy": "DummyTokenizer",
# }

# TOKENIZER_CLASSES_ALL_LLAMA = {key: "LlamaTokenizer" for key in TOKENIZER_CLASSES.keys()}

def resolve_tokenizer(model_id: str, model_path: str) -> Tokenizer:
    # tokenizer_class = TOKENIZER_CLASSES[model_id]
    # tokenizer_class = TOKENIZER_CLASSES_ALL_LLAMA[model_id]
    # tokenizer_class = "SpmTokenizer" if 'gemma' in model_id else "TikTokenizer"
    tokenizer_class = "TikTokenizer"
    tokenizer_module = importlib.import_module("exo.tokenizer")
    tokenizer_class_obj = getattr(tokenizer_module, tokenizer_class)
    return tokenizer_class_obj(model_path)
