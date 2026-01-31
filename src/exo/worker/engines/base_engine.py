from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Tuple, Union

from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse


class Engine(ABC):
    """
    Abstract Base Class for an inference engine.
    This defines the common interface that all engines (MLX, PyTorch, etc.) must implement.
    """

    def __init__(self, bound_instance: BoundInstance):
        self.bound_instance = bound_instance
        self.model: Any = None
        self.tokenizer: Any = None
        self.group: Any = None  # For distributed communication

    @abstractmethod
    def initialize_distributed_group(self) -> Any:
        """Initializes the distributed communication group."""
        pass

    @abstractmethod
    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Loads the model and tokenizer for the assigned shard."""
        pass

    @abstractmethod
    def warmup_inference(self) -> int:
        """Warms up the inference engine. Returns number of tokens generated."""
        pass

    @abstractmethod
    async def generate(
        self,
        task_params: ChatCompletionTaskParams,
    ) -> AsyncGenerator[Union[GenerationResponse, ToolCallResponse], None]:
        """Generates text based on a prompt. Yields GenerationResponse or ToolCallResponse chunks."""
        # Note: This must be async def with a yield to be an async generator
        # Abstract methods need at least one yield for type checkers
        yield  # type: ignore[misc]

