"""
PyTorch inference engine implementation.

This module provides the PytorchEngine class for running inference on NVIDIA GPUs
using PyTorch and HuggingFace Transformers. It supports:
- Model loading via AutoModelForCausalLM
- Pipeline parallelism for distributed inference
- Streaming text generation with TextIteratorStreamer
- Tool call parsing (using base class helpers)
"""

from collections.abc import Generator
from threading import Thread
from typing import Any, Tuple, Union

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

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
    Streaming is implemented via TextIteratorStreamer.
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
        Generate text using PyTorch with streaming.

        Uses TextIteratorStreamer to yield tokens as they are generated,
        providing a responsive streaming experience similar to MLX.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model_and_tokenizer first.")

        # Build prompt - use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and task_params.input:
            messages = [
                {"role": msg.role, "content": msg.content} for msg in task_params.input
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

        prompt_tokens = inputs["input_ids"].shape[1]

        # Create streamer for real-time token output
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Set up generation parameters
        max_tokens = task_params.max_output_tokens or MAX_TOKENS
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "do_sample": task_params.temperature is not None
            and task_params.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Add optional parameters if provided
        if task_params.temperature is not None and task_params.temperature > 0:
            generation_kwargs["temperature"] = task_params.temperature
        if task_params.top_p is not None:
            generation_kwargs["top_p"] = task_params.top_p
        if task_params.top_k is not None:
            generation_kwargs["top_k"] = task_params.top_k

        # Run generation in background thread
        def generate_in_thread() -> None:
            with torch.no_grad():
                self.model.generate(**generation_kwargs)

        thread = Thread(target=generate_in_thread)
        thread.start()

        # Stream tokens as they are generated
        completion_tokens = 0
        try:
            for new_text in streamer:
                if new_text:  # Skip empty strings
                    completion_tokens += 1
                    yield GenerationResponse(
                        text=new_text,
                        token=completion_tokens,  # Approximate token count
                        finish_reason=None,
                        stats=None,
                        usage=None,
                    )
        finally:
            thread.join()

        # Final response with usage stats
        total_tokens = prompt_tokens + completion_tokens
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=PromptTokensDetails(),
            completion_tokens_details=CompletionTokensDetails(),
        )

        yield GenerationResponse(
            text="",
            token=0,
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
