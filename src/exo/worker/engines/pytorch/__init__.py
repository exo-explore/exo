"""
PyTorch inference engine implementation.

This module provides the PytorchEngine class for running inference on NVIDIA GPUs
using PyTorch and HuggingFace Transformers. It supports:
- Model loading via AutoModelForCausalLM
- Pipeline parallelism for distributed inference
- Tool call parsing (using base class helpers)
"""

from collections.abc import Generator
from typing import Any, Tuple, Union

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from exo.shared.types.api import CompletionTokensDetails, PromptTokensDetails, Usage
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.base_engine import Engine, TimeoutCallback
from exo.worker.engines.pytorch.auto_parallel import (
    MockDistributedGroup,
    pipeline_auto_parallel,
)

# Default max tokens if not specified in request (matches MLX's default)
MAX_TOKENS: int = 32168


class PytorchEngine(Engine):
    """
    PyTorch-based inference engine using HuggingFace Transformers.

    This engine supports NVIDIA GPUs and can run on Linux systems.
    Pipeline parallelism is supported for distributed inference.

    Note: This is a minimal implementation. For production use, consider:
    - Streaming with TextIteratorStreamer
    - KV cache management with past_key_values
    - Proper tool call support
    """

    def __init__(self, bound_instance: BoundInstance):
        super().__init__(bound_instance)
        self._model_name = bound_instance.instance.shard_assignments.model_id

    def initialize_distributed_group(self) -> Any:
        """Initialize distributed group (mock for now)."""
        # TODO: Implement torch.distributed.init_process_group for real parallelism
        self.group = MockDistributedGroup()
        return self.group

    def load_model_and_tokenizer(
        self, on_timeout: TimeoutCallback | None = None
    ) -> Tuple[Any, Any]:
        """Load HuggingFace model and tokenizer."""
        shard_meta = self.shard_metadata
        if not isinstance(shard_meta, PipelineShardMetadata):
            raise TypeError(
                f"PytorchEngine requires PipelineShardMetadata, got {type(shard_meta).__name__}"
            )

        logger.info(
            f"Loading model: {self._model_name} for shard {shard_meta.device_rank}/{shard_meta.world_size}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self._model_name)

        # Apply pipeline parallelism
        self.model = pipeline_auto_parallel(self.model, self.group, shard_meta)

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.to("cuda")
            logger.info("Model moved to GPU.")
        else:
            logger.warning("CUDA not available, model will run on CPU.")

        return self.model, self.tokenizer

    def warmup_inference(self) -> int:
        """Warmup not implemented for PyTorch yet."""
        # TODO: Run a small generation to warm up CUDA kernels
        return 0

    def generate(
        self,
        task_params: TextGenerationTaskParams,
    ) -> Generator[Union[GenerationResponse, ToolCallResponse], None, None]:
        """
        Generate text using PyTorch.

        Note: This is a basic implementation that generates all tokens at once.
        For streaming, use TextIteratorStreamer from transformers.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model_and_tokenizer first.")

        # Build prompt - use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and task_params.input:
            messages = [
                {"role": "user", "content": msg.content} for msg in task_params.input
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif task_params.input:
            prompt = task_params.input[-1].content
        else:
            raise ValueError("No input messages provided")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate (synchronously for now)
        # TODO: Use TextIteratorStreamer for streaming
        max_tokens = task_params.max_output_tokens or MAX_TOKENS
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]

        # Calculate token counts for usage stats
        prompt_tokens = inputs["input_ids"].shape[1]
        completion_tokens = outputs.shape[1] - prompt_tokens
        total_tokens = prompt_tokens + completion_tokens

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=PromptTokensDetails(),
            completion_tokens_details=CompletionTokensDetails(),
        )

        # Yield a single response (not streaming yet)
        # TODO: For streaming, yield multiple responses as tokens are generated
        yield GenerationResponse(
            text=generated_text,
            token=0,  # Dummy token ID
            finish_reason="stop",
            stats=None,
            usage=usage,
        )

    def cleanup(self) -> None:
        """Clean up PyTorch resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()
