"""
Engine abstraction for inference backends.

This module defines the Engine interface that all inference backends must implement.
It also provides generic post-processing utilities that work across backends.

Design Philosophy:
- Engine interface is backend-agnostic (no MLX/PyTorch types)
- Generic post-processing (tool calls, thinking tags) lives in base class
- Backend-specific logic (tokenizer patching, model loading) lives in subclasses
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Any

import loguru
from pydantic import ValidationError

from exo.shared.types.api import ImageEditsTaskParams, ImageGenerationTaskParams
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ImageGenerationResponse,
    PartialImageResponse,
    ToolCallItem,
    ToolCallResponse,
)

# Type alias for timeout callback
TimeoutCallback = Callable[[], None]

# Type alias for tool parser function
ToolParser = Callable[[str], dict[str, Any] | list[dict[str, Any]]]

logger: "loguru.Logger" = loguru.logger


class Engine(ABC):
    """
    Abstract Base Class for an inference engine.

    This defines the common interface that all engines (MLX, PyTorch, etc.) must implement.
    The Engine encapsulates all backend-specific logic including:
    - Distributed communication setup
    - Model/tokenizer loading
    - Inference warmup
    - Text generation

    Subclasses should use the protected helper methods for common post-processing:
    - _parse_tool_calls(): Parse tool calls from generation stream
    - _wrap_thinking_output(): Prepend thinking tags for reasoning models
    - _filter_tokens(): Filter out specific tokens from stream

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

    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def initialize_distributed_group(self) -> Any:
        """
        Initialize the distributed communication group for multi-device inference.

        Returns:
            The initialized group object (backend-specific type).
        """

    @abstractmethod
    def load_model_and_tokenizer(
        self, on_timeout: TimeoutCallback | None = None
    ) -> tuple[Any, Any]:
        """
        Load the model and tokenizer for the assigned shard.

        Args:
            on_timeout: Optional callback to invoke if loading times out.

        Returns:
            Tuple of (model, tokenizer).
        """

    @abstractmethod
    def warmup_inference(self) -> int:
        """
        Warm up the inference engine by running a small generation.

        Returns:
            Number of tokens generated during warmup.
        """

    @abstractmethod
    def generate(
        self,
        task_params: TextGenerationTaskParams,
    ) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
        """
        Generate text based on task parameters.

        This method should:
        1. Apply chat template to build the prompt
        2. Run inference with the backend
        3. Apply any model-specific post-processing using helper methods
        4. Yield responses

        Args:
            task_params: The text generation task parameters.

        Yields:
            GenerationResponse for regular tokens, ToolCallResponse for tool calls.
        """

    # =========================================================================
    # Optional overridable methods
    # =========================================================================

    def check_debug_prompts(self, task_params: TextGenerationTaskParams) -> None:
        """
        Check for debug prompts in the input. Can be overridden by subclasses.

        The default implementation does nothing. Engines may implement
        special debug triggers like EXO_RUNNER_MUST_FAIL.
        """
        return  # B027: Intentionally empty, subclasses can override

    def should_cancel(self, want_to_cancel: bool) -> bool:
        """
        Determine if the current generation should be cancelled.

        For distributed engines, this may coordinate across devices
        (e.g., MLX uses mx_any to synchronize cancellation).

        Args:
            want_to_cancel: Whether this node wants to cancel.

        Returns:
            True if generation should be cancelled.
        """
        return want_to_cancel

    def initialize_image_model(self) -> None:
        """
        Initialize the image generation model.

        The default implementation raises NotImplementedError.
        Engines that support image generation should override this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image generation"
        )

    def warmup_image_generator(self) -> Any:
        """
        Warm up the image generation model.

        Returns:
            A PIL Image or None after warmup.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image generation"
        )

    def generate_image(
        self,
        task_params: ImageGenerationTaskParams | ImageEditsTaskParams,
    ) -> Generator[ImageGenerationResponse | PartialImageResponse, None, None]:
        """
        Generate or edit images based on task parameters.

        Args:
            task_params: The image generation or editing task parameters.

        Yields:
            ImageGenerationResponse or PartialImageResponse.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image generation"
        )

    def cleanup(self) -> None:
        """
        Clean up resources when shutting down. Can be overridden by subclasses.

        The default implementation does nothing.
        """
        return  # B027: Intentionally empty, subclasses can override

    # =========================================================================
    # Protected helper methods for post-processing
    # These are generic and can be used by any engine implementation
    # =========================================================================

    def _parse_tool_calls(
        self,
        responses: Generator[GenerationResponse | ToolCallResponse, None, None],
        tool_call_start: str,
        tool_call_end: str,
        tool_parser: ToolParser,
    ) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
        """
        Parse tool calls from a generation stream.

        This is a generic post-processor that works with any backend.
        It detects tool call markers in the token stream and parses them.

        Args:
            responses: The generation stream to process.
            tool_call_start: Token marking start of tool call (e.g., "<tool_call>").
            tool_call_end: Token marking end of tool call (e.g., "</tool_call>").
            tool_parser: Function to parse tool call text into structured format.

        Yields:
            GenerationResponse for regular tokens, ToolCallResponse for tool calls.
        """
        in_tool_call = False
        tool_call_text_parts: list[str] = []

        for response in responses:
            # Pass through ToolCallResponse unchanged (already parsed by model-specific logic)
            if isinstance(response, ToolCallResponse):
                yield response
                continue

            # Check for tool call start marker
            if response.text == tool_call_start:
                in_tool_call = True
                continue

            # Check for tool call end marker
            if in_tool_call and response.text == tool_call_end:
                try:
                    parsed = tool_parser("".join(tool_call_text_parts).strip())
                    logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
                    if isinstance(parsed, list):
                        tools = [self._validate_tool_call(tool) for tool in parsed]
                    else:
                        tools = [self._validate_tool_call(parsed)]
                    yield ToolCallResponse(tool_calls=tools, usage=response.usage)
                except (
                    json.JSONDecodeError,
                    ValidationError,
                    ValueError,
                    AttributeError,
                ) as e:
                    logger.opt(exception=e).warning("tool call parsing failed")
                    # Yield the raw text if parsing fails
                    yield response.model_copy(
                        update={
                            "text": tool_call_start
                            + "".join(tool_call_text_parts)
                            + tool_call_end
                        }
                    )

                in_tool_call = False
                tool_call_text_parts = []
                continue

            # Accumulate text while in tool call
            if in_tool_call:
                tool_call_text_parts.append(response.text)
                # Handle interrupted tool calls (generation ended mid-tool-call)
                if response.finish_reason is not None:
                    logger.info(
                        "tool call parsing interrupted, yield partial tool call as text"
                    )
                    yield GenerationResponse(
                        text=tool_call_start + "".join(tool_call_text_parts),
                        token=0,
                        finish_reason=response.finish_reason,
                        usage=None,
                    )
                continue

            # Pass through regular tokens
            yield response

    def _wrap_thinking_output(
        self,
        responses: Generator[GenerationResponse | ToolCallResponse, None, None],
        think_start: str,
        think_start_id: int,
    ) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
        """
        Prepend thinking tags for reasoning models.

        For models that inject thinking tags in the prompt (like GLM-4.7),
        this prepends the thinking tag to the output stream so the frontend
        can properly parse thinking content.

        Args:
            responses: The generation stream to process.
            think_start: The thinking start tag (e.g., "<think>").
            think_start_id: Token ID for the thinking start tag.

        Yields:
            Responses with thinking tag prepended to first token.
        """
        first = True
        for response in responses:
            if isinstance(response, ToolCallResponse):
                yield response
                continue
            if first:
                first = False
                yield response.model_copy(
                    update={"text": think_start, "token": think_start_id}
                )
            yield response

    def _filter_tokens(
        self,
        responses: Generator[GenerationResponse | ToolCallResponse, None, None],
        tokens_to_filter: set[str],
    ) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
        """
        Filter out specific tokens from the generation stream.

        Useful for removing model-specific markers that shouldn't be in output.

        Args:
            responses: The generation stream to process.
            tokens_to_filter: Set of token strings to remove.

        Yields:
            Responses with specified tokens filtered out.
        """
        for response in responses:
            if isinstance(response, ToolCallResponse):
                yield response
                continue
            if response.text in tokens_to_filter:
                continue
            yield response

    @staticmethod
    def _validate_tool_call(obj: dict[str, Any]) -> ToolCallItem:
        """
        Validate and convert a tool call dictionary to ToolCallItem.

        Args:
            obj: Dictionary with 'name' and 'arguments' keys.

        Returns:
            Validated ToolCallItem.

        Raises:
            ValidationError: If required fields are missing.
        """
        name = obj.get("name")
        args = obj.get("arguments")
        if name is not None and args is not None and isinstance(name, str):
            raw_id: object = obj.get("id")
            extra = {"id": str(raw_id)} if raw_id is not None else {}
            return ToolCallItem(
                **extra,
                name=name,
                arguments=json.dumps(args) if not isinstance(args, str) else args,
            )
        raise ValidationError.from_exception_data("Invalid tool call structure", [])
