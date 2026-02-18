"""
MLX inference engine implementation.

This module provides the MlxEngine class for running inference on Apple Silicon
using the MLX framework. It handles all MLX-specific logic including:
- Distributed communication via MLX
- Model loading with mlx_lm
- KV prefix cache for faster prefill
- Model-specific tokenizer patching (Kimi, GLM)
- GPT-OSS output parsing
"""

from collections.abc import Generator
from functools import cache
from typing import Any, Union

import loguru
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    HarmonyError,  # pyright: ignore[reportUnknownVariableType]
    Role,
    StreamableParser,
    load_harmony_encoding,
)

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
from exo.worker.engines.base_engine import Engine, TimeoutCallback
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    detect_thinking_prompt_suffix,
    initialize_mlx,
    load_mlx_items,
    mlx_force_oom,
    mx_any,
)

# Re-export for type stubs - MLX doesn't provide strong typing guarantees
# so we define our own interface for what we need from nn.Module


class Model(nn.Module):
    """Type stub for MLX language models with required interface."""

    layers: list[nn.Module]

    def __call__(
        self,
        x: mx.array,
        cache: list[KVCache] | None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...


logger: "loguru.Logger" = loguru.logger


@cache
def _get_gpt_oss_encoding():
    """Get the GPT-OSS encoding for parsing."""
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


class MlxEngine(Engine):
    """
    Inference engine for MLX models on Apple Silicon.

    Handles all MLX-specific logic including:
    - Model patches for Kimi, GLM, GPT-OSS
    - KV prefix cache for improved prefill performance
    - Thinking model detection and tag insertion
    - Tool call parsing
    """

    def __init__(self, bound_instance: BoundInstance):
        super().__init__(bound_instance)
        self._kv_prefix_cache: KVPrefixCache | None = None
        self._image_model: Any = None

    def initialize_distributed_group(self) -> Any:
        """Initialize MLX distributed communication."""
        self.group = initialize_mlx(self.bound_instance)
        return self.group

    def load_model_and_tokenizer(
        self, on_timeout: TimeoutCallback | None = None
    ) -> tuple[Any, Any]:
        """Load MLX model and tokenizer, create KV prefix cache."""
        self.model, self.tokenizer = load_mlx_items(
            self.bound_instance, self.group, on_timeout=on_timeout
        )
        # Create KV prefix cache for faster prefill on repeated prompts
        self._kv_prefix_cache = KVPrefixCache(self.group)
        return self.model, self.tokenizer

    def warmup_inference(self) -> int:
        """Warm up MLX inference."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded before warmup.")
        return warmup_inference(
            model=self.model, tokenizer=self.tokenizer, group=self.group
        )

    def generate(
        self,
        task_params: TextGenerationTaskParams,
    ) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
        """
        Generate text using MLX with all model-specific handling applied.

        This method:
        1. Builds prompt using chat template
        2. Creates MLX generation stream
        3. Applies model-specific patches (Kimi, GLM, GPT-OSS)
        4. Applies thinking model wrapper if needed
        5. Parses tool calls if enabled
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded before generation.")

        model_id_lower = self.model_id.lower()
        tokenizer: TokenizerWrapper = self.tokenizer

        # Build prompt using chat template
        prompt = apply_chat_template(tokenizer, task_params)

        # Create base generator with KV prefix cache
        mlx_generator: Generator[GenerationResponse | ToolCallResponse, None, None] = (
            mlx_generate(
                model=self.model,
                tokenizer=tokenizer,
                task=task_params,
                prompt=prompt,
                kv_prefix_cache=self._kv_prefix_cache,
                group=self.group,
            )
        )

        # Apply thinking model wrapper if needed
        if (
            detect_thinking_prompt_suffix(prompt, tokenizer)
            and tokenizer.think_start is not None
            and tokenizer.think_start_id is not None
        ):
            mlx_generator = self._wrap_thinking_output(
                mlx_generator, tokenizer.think_start, tokenizer.think_start_id
            )

        # Apply model-specific patches
        if "kimi" in model_id_lower:
            # Filter Kimi-specific markers
            mlx_generator = self._filter_tokens(
                mlx_generator,
                {"<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"},
            )
            self._patch_kimi_tokenizer(tokenizer)
        elif "glm" in model_id_lower:
            self._patch_glm_tokenizer(tokenizer)
        elif isinstance(self.model, GptOssModel):
            encoding = _get_gpt_oss_encoding()
            mlx_generator = self._parse_gpt_oss(mlx_generator, encoding)

        # Apply tool call parsing if enabled (and not GPT-OSS which has its own)
        if tokenizer.has_tool_calling and not isinstance(self.model, GptOssModel):
            assert tokenizer.tool_call_start
            assert tokenizer.tool_call_end
            assert tokenizer.tool_parser
            mlx_generator = self._parse_tool_calls(
                mlx_generator,
                tokenizer.tool_call_start,
                tokenizer.tool_call_end,
                tokenizer.tool_parser,
            )

        yield from mlx_generator

    def check_debug_prompts(self, task_params: TextGenerationTaskParams) -> None:
        """Check for debug prompts and trigger special behaviors."""
        if len(task_params.input) == 0:
            logger.debug("Empty message list in debug prompt check")
            return
        prompt = task_params.input[0].content
        if not prompt:
            return

        if "EXO RUNNER MUST FAIL" in prompt:
            logger.info("raising exception")
            raise Exception("Artificial runner exception - for testing purposes only.")
        if "EXO RUNNER MUST OOM" in prompt:
            mlx_force_oom()
        if "EXO RUNNER MUST TIMEOUT" in prompt:
            import time

            time.sleep(100)

    def cleanup(self) -> None:
        """Clean up MLX resources."""
        del self.model, self.tokenizer, self.group
        mx.clear_cache()
        import gc

        gc.collect()

    def should_cancel(self, want_to_cancel: bool) -> bool:
        """Coordinate cancellation across MLX distributed group."""
        return mx_any(want_to_cancel, self.group)

    def initialize_image_model(self) -> None:
        """Initialize MLX image model."""
        from exo.worker.engines.image import initialize_image_model

        self._image_model = initialize_image_model(self.bound_instance)

    def warmup_image_generator(self) -> Any:
        """Warm up MLX image model."""
        from exo.worker.engines.image import warmup_image_generator

        assert self._image_model is not None
        return warmup_image_generator(model=self._image_model)

    def generate_image(
        self,
        task_params: ImageGenerationTaskParams | ImageEditsTaskParams,
    ) -> Generator[ImageGenerationResponse | PartialImageResponse, None, None]:
        """Generate images using MLX."""
        from exo.worker.engines.image import generate_image

        assert self._image_model is not None
        yield from generate_image(model=self._image_model, task=task_params)

    # =========================================================================
    # MLX-specific private methods
    # =========================================================================

    def _parse_gpt_oss(
        self,
        responses: Generator[GenerationResponse | ToolCallResponse, None, None],
        encoding: Any,
    ) -> Generator[GenerationResponse | ToolCallResponse, None, None]:
        """Parse GPT-OSS model outputs to match standard format."""
        stream = StreamableParser(encoding, role=Role.ASSISTANT)
        thinking = False
        current_tool_name: str | None = None
        tool_arg_parts: list[str] = []

        for response in responses:
            if isinstance(response, ToolCallResponse):
                yield response
                continue

            try:
                stream.process(response.token)
            except HarmonyError:
                logger.error("Encountered critical Harmony Error, returning early")
                return

            delta = stream.last_content_delta
            ch = stream.current_channel
            recipient = stream.current_recipient

            if recipient != current_tool_name:
                if current_tool_name is not None:
                    prefix = "functions."
                    if current_tool_name.startswith(prefix):
                        current_tool_name = current_tool_name[len(prefix) :]
                    yield ToolCallResponse(
                        tool_calls=[
                            ToolCallItem(
                                name=current_tool_name,
                                arguments="".join(tool_arg_parts).strip(),
                            )
                        ],
                        usage=response.usage,
                    )
                    tool_arg_parts = []
                current_tool_name = recipient

            # If inside a tool call, accumulate arguments
            if current_tool_name is not None:
                if delta:
                    tool_arg_parts.append(delta)
                continue

            if ch == "analysis" and not thinking:
                thinking = True
                yield response.model_copy(update={"text": "<think>"})

            if ch != "analysis" and thinking:
                thinking = False
                yield response.model_copy(update={"text": "</think>"})

            if delta:
                yield response.model_copy(update={"text": delta})

            if response.finish_reason is not None:
                if thinking:
                    yield response.model_copy(update={"text": "</think>"})
                yield response

    @staticmethod
    def _patch_kimi_tokenizer(tokenizer: TokenizerWrapper) -> None:
        """
        Patch tokenizer for Kimi-K2 tool calling support.

        Kimi uses a specific format for tool calls:
            functions.multiply:0 <|tool_call_argument_begin|> {"a": 2, "b": 3}
        """
        import ast
        import json
        from typing import Any

        import regex as re

        _func_name_regex = re.compile(
            r"^\s*(.+)[:](\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
        )
        _func_arg_regex = re.compile(
            r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL
        )

        tool_call_start = "<|tool_call_begin|>"
        tool_call_end = "<|tool_call_end|>"

        def _deserialize(value: str) -> Any:  # pyright: ignore[reportAny]
            try:
                return json.loads(value)  # pyright: ignore[reportAny]
            except Exception:
                pass
            try:
                return ast.literal_eval(value)  # pyright: ignore[reportAny]
            except Exception:
                pass
            return value

        def parse_tool_call(text: str, tools: Any | None = None):
            func_name_match = _func_name_regex.search(text)
            if func_name_match is None:
                raise ValueError(
                    f"Could not parse function name from tool call: {text!r}"
                )
            original_func_name = func_name_match.group(1)
            tool_id = func_name_match.group(2)
            func_name = original_func_name[original_func_name.find(".") + 1 :]

            func_args_match = _func_arg_regex.search(text)
            if func_args_match is None:
                raise ValueError(
                    f"Could not parse function args from tool call: {text!r}"
                )
            func_args = func_args_match.group(1)
            arg_dct = _deserialize(func_args)  # pyright: ignore[reportAny]

            return dict(
                id=f"{original_func_name}:{tool_id}",
                name=func_name,
                arguments=arg_dct,  # pyright: ignore[reportAny]
            )

        tokenizer._tool_call_start = tool_call_start
        tokenizer._tool_call_end = tool_call_end
        tokenizer._tool_parser = parse_tool_call

    @staticmethod
    def _patch_glm_tokenizer(tokenizer: TokenizerWrapper) -> None:
        """
        Patch tokenizer for GLM-4 tool calling support.

        GLM uses XML-like format:
            <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>
        """
        import ast
        import json
        from typing import Any

        import regex as re

        _func_name_regex = re.compile(r"^(.*?)<arg_key>", re.DOTALL)
        _func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\n|\s)*<arg_value>(.*?)(?:</arg_value>|(?=<arg_key>)|$)",
            re.DOTALL,
        )

        tool_call_start = "<tool_call>"
        tool_call_end = "</tool_call>"

        def _is_string_type(
            tool_name: str,
            arg_name: str,
            tools: list[Any] | None,
        ) -> bool:
            if tools is None:
                return False
            for tool in tools:  # pyright: ignore[reportAny]
                func = tool["function"]  # pyright: ignore[reportAny]
                if func["name"] == tool_name:
                    params = func["parameters"]  # pyright: ignore[reportAny]
                    if params is None:
                        return False
                    props = params.get("properties", {})  # pyright: ignore[reportAny]
                    arg_props = props.get(arg_name, {})  # pyright: ignore[reportAny]
                    arg_type = arg_props.get("type", None)  # pyright: ignore[reportAny]
                    return arg_type == "string"  # pyright: ignore[reportAny]
            return False

        def _deserialize(value: str) -> Any:  # pyright: ignore[reportAny]
            try:
                return json.loads(value)  # pyright: ignore[reportAny]
            except Exception:
                pass
            try:
                return ast.literal_eval(value)  # pyright: ignore[reportAny]
            except Exception:
                pass
            return value

        def parse_tool_call(text: str, tools: list[Any] | None = None):
            func_name_match = _func_name_regex.search(text)
            if func_name_match is None:
                raise ValueError(
                    f"Could not parse function name from tool call: {text!r}"
                )
            func_name = func_name_match.group(1)

            pairs = _func_arg_regex.findall(text)
            arg_dct: dict[str, Any] = {}
            for key, value in pairs:  # pyright: ignore[reportAny]
                arg_key = key.strip()  # pyright: ignore[reportAny]
                arg_val = value.strip()  # pyright: ignore[reportAny]
                if not _is_string_type(func_name, arg_key, tools):  # pyright: ignore[reportAny]
                    arg_val = _deserialize(arg_val)  # pyright: ignore[reportAny]
                arg_dct[arg_key] = arg_val
            return dict(name=func_name, arguments=arg_dct)

        tokenizer._tool_call_start = tool_call_start
        tokenizer._tool_call_end = tool_call_end
        tokenizer._tool_parser = parse_tool_call
