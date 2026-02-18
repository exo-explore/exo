"""Abstract Engine interface for multi-backend support.

This module defines the Engine abstraction that allows runner.py to work
with multiple backends (MLX, PyTorch, etc.) through a unified interface.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Any, Protocol

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse


class DistributedGroup(Protocol):
    pass


class Tokenizer(Protocol):

    has_tool_calling: bool
    tool_call_start: str | None
    tool_call_end: str | None
    tool_parser: Any | None
    think_start: str
    think_start_id: int


class Model(Protocol):
    pass


class KVCache(Protocol):
    pass


class Engine(ABC):

    @abstractmethod
    def initialize_distributed_group(
        self, bound_instance: BoundInstance
    ) -> DistributedGroup | None:
       
        raise NotImplementedError

    @abstractmethod
    def load_model_and_tokenizer(
        self,
        bound_instance: BoundInstance,
        group: DistributedGroup | None,
        on_timeout: Callable[[], None] | None = None,
    ) -> tuple[Model, Tokenizer]:
       
        raise NotImplementedError

    @abstractmethod
    def warmup_inference(
        self,
        model: Model,
        tokenizer: Tokenizer,
        group: DistributedGroup | None,
    ) -> int:
      
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        model: Model,
        tokenizer: Tokenizer,
        task: TextGenerationTaskParams,
        prompt: str,
        kv_cache: KVCache | None = None,
        group: DistributedGroup | None = None,
    ) -> Generator[GenerationResponse]:
        
        raise NotImplementedError

    @abstractmethod
    def apply_chat_template(
        self, tokenizer: Tokenizer, task_params: TextGenerationTaskParams
    ) -> str:
     
        raise NotImplementedError

    @abstractmethod
    def detect_thinking_prompt_suffix(
        self, prompt: str, tokenizer: Tokenizer
    ) -> bool:
     
        raise NotImplementedError

    @abstractmethod
    def any_cancel(self, want_to_cancel: bool, group: DistributedGroup | None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_kv_cache(self, group: DistributedGroup | None) -> KVCache | None:
       
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError
