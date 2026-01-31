from collections.abc import Generator
from functools import cache
from typing import Any, Tuple, Union

import loguru
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    load_harmony_encoding,
)

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.base_engine import Engine, TimeoutCallback
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
from exo.worker.engines.mlx.patches import (
    filter_kimi_tokens,
    parse_gpt_oss,
    parse_thinking_models,
    parse_tool_calls,
    patch_glm_tokenizer,
    patch_kimi_tokenizer,
)
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    detect_thinking_prompt_suffix,
    initialize_mlx,
    load_mlx_items,
    mlx_force_oom,
)

# These are wrapper functions to fix the fact that mlx is not strongly typed in the same way that EXO is.
# For example - MLX has no guarantee of the interface that nn.Module will expose. But we need a guarantee that it has a  __call__() function


class Model(nn.Module):
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
    Inference engine for MLX models.
    Handles all MLX-specific logic including model patches for Kimi, GLM, GPT-OSS, etc.
    Includes KV prefix cache for improved prefill performance.
    """

    def __init__(self, bound_instance: BoundInstance):
        super().__init__(bound_instance)
        self._kv_prefix_cache: KVPrefixCache | None = None

    def initialize_distributed_group(self) -> Any:
        self.group = initialize_mlx(self.bound_instance)
        return self.group

    def load_model_and_tokenizer(
        self, on_timeout: TimeoutCallback | None = None
    ) -> Tuple[Any, Any]:
        self.model, self.tokenizer = load_mlx_items(
            self.bound_instance, self.group, on_timeout=on_timeout
        )
        # Create KV prefix cache for faster prefill on repeated prompts
        self._kv_prefix_cache = KVPrefixCache(self.group)
        return self.model, self.tokenizer

    def warmup_inference(self) -> int:
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded before warmup.")
        return warmup_inference(
            model=self.model, tokenizer=self.tokenizer, group=self.group
        )

    def generate(
        self,
        task_params: TextGenerationTaskParams,
    ) -> Generator[Union[GenerationResponse, ToolCallResponse], None, None]:
        """
        Generate text using the MLX model with all model-specific patches applied.

        This method encapsulates all MLX-specific logic:
        - Prompt building with chat templates
        - Kimi token filtering
        - GLM tokenizer patching
        - GPT-OSS output parsing
        - Thinking model detection
        - Tool call parsing
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded before generation.")

        model_id_lower = self.model_id.lower()
        tokenizer: TokenizerWrapper = self.tokenizer

        # Build prompt using chat template
        prompt = apply_chat_template(tokenizer, task_params)

        # Create base generator with KV prefix cache for faster prefill
        mlx_generator = mlx_generate(
            model=self.model,
            tokenizer=tokenizer,
            task=task_params,
            prompt=prompt,
            kv_prefix_cache=self._kv_prefix_cache,
            group=self.group,
        )

        # Apply thinking model wrapper if needed
        if detect_thinking_prompt_suffix(prompt, tokenizer):
            mlx_generator = parse_thinking_models(mlx_generator, tokenizer)

        # Apply model-specific patches
        if "kimi" in model_id_lower:
            mlx_generator = filter_kimi_tokens(mlx_generator)
            patch_kimi_tokenizer(tokenizer)
        elif "glm" in model_id_lower:
            patch_glm_tokenizer(tokenizer)
        elif isinstance(self.model, GptOssModel):
            encoding = _get_gpt_oss_encoding()
            mlx_generator = parse_gpt_oss(mlx_generator, encoding)

        # Apply tool call parsing if enabled (and not GPT-OSS which has its own parsing)
        if tokenizer.has_tool_calling and not isinstance(self.model, GptOssModel):
            assert tokenizer.tool_call_start
            assert tokenizer.tool_call_end
            assert tokenizer.tool_parser
            mlx_generator = parse_tool_calls(
                mlx_generator,
                tokenizer.tool_call_start,
                tokenizer.tool_call_end,
                tokenizer.tool_parser,
                logger,
            )

        # Yield responses from the patched generator
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
