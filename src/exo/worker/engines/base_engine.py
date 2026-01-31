from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Callable, Tuple, Union

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse

# Type alias for timeout callback
TimeoutCallback = Callable[[], None]


class Engine(ABC):
    """
    Abstract Base Class for an inference engine.
    This defines the common interface that all engines (MLX, PyTorch, etc.) must implement.

    The Engine encapsulates all backend-specific logic including:
    - Distributed communication setup
    - Model/tokenizer loading
    - Inference warmup
    - Text generation with model-specific patches
    - Tool call parsing

    Image generation is NOT part of this interface as it's MLX-specific for now.
    """

    def __init__(self, bound_instance: BoundInstance):
        self.bound_instance = bound_instance
        self.model: Any = None
        self.tokenizer: Any = None
        self.group: Any = None  # For distributed communication

    @property
    def shard_metadata(self):
        """Shortcut to access the shard metadata."""
        return self.bound_instance.bound_shard

    @property
    def model_id(self) -> str:
        """Shortcut to access the model ID."""
        return self.bound_instance.bound_shard.model_card.model_id

    @abstractmethod
    def initialize_distributed_group(self) -> Any:
        """
        Initializes the distributed communication group for multi-device inference.

        Returns:
            The initialized group object (backend-specific type).
        """

    @abstractmethod
    def load_model_and_tokenizer(
        self, on_timeout: TimeoutCallback | None = None
    ) -> Tuple[Any, Any]:
        """
        Loads the model and tokenizer for the assigned shard.

        Args:
            on_timeout: Optional callback to invoke if loading times out.

        Returns:
            Tuple of (model, tokenizer).
        """

    @abstractmethod
    def warmup_inference(self) -> int:
        """
        Warms up the inference engine by running a small generation.

        Returns:
            Number of tokens generated during warmup.
        """

    @abstractmethod
    def generate(
        self,
        task_params: TextGenerationTaskParams,
    ) -> Generator[Union[GenerationResponse, ToolCallResponse], None, None]:
        """
        Generates text based on task parameters.

        This method should handle all model-specific quirks including:
        - Chat template application
        - Thinking model detection and tag insertion
        - Model-specific token filtering (e.g., Kimi tool call markers)
        - Tool call parsing

        Args:
            task_params: The text generation task parameters.

        Yields:
            GenerationResponse for regular tokens, ToolCallResponse for tool calls.
        """
        # Abstract - subclasses must implement

    def check_debug_prompts(self, task_params: TextGenerationTaskParams) -> None:
        """
        Check for debug prompts in the input. Can be overridden by subclasses.

        The default implementation does nothing. MLX engines may implement
        special debug triggers like EXO_RUNNER_MUST_FAIL.
        """
        return  # B027: Intentionally empty, subclasses can override

    def cleanup(self) -> None:
        """
        Clean up resources when shutting down. Can be overridden by subclasses.

        The default implementation does nothing.
        """
        return  # B027: Intentionally empty, subclasses can override
