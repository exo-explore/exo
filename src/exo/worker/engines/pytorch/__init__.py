from typing import Any, AsyncGenerator, Tuple, Union
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from exo.worker.engines.pytorch.auto_parallel import pipeline_auto_parallel, MockDistributedGroup, PipelineShardMetadata
from exo.shared.types.worker.instances import BoundInstance
from exo.worker.engines.base_engine import Engine
from exo.shared.types.api import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.shared.types.chunks import TokenChunk

class PytorchEngine(Engine):
    def __init__(self, bound_instance: BoundInstance):
        super().__init__(bound_instance)
        self.model_name = bound_instance.instance.shard_assignments.model_id
        self.shard_metadata = bound_instance.bound_shard

    def initialize_distributed_group(self) -> Any:
        # For now, we'll just set a mock group.
        # In a real implementation, this would use torch.distributed.init_process_group
        self.group = MockDistributedGroup()
        return self.group

    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        logger.info(f"Loading model: {self.model_name} for shard {self.shard_metadata.device_rank}/{self.shard_metadata.world_size}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Apply pipeline parallelism
        self.model = pipeline_auto_parallel(self.model, self.group, self.shard_metadata)

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')
            logger.info("Model moved to GPU.")
        else:
            logger.warning("CUDA not available, model will run on CPU.")
        
        return self.model, self.tokenizer

    def warmup_inference(self) -> int:
        # Simple warmup
        return 0

    async def generate(
        self,
        task_params: ChatCompletionTaskParams,
    ) -> AsyncGenerator[Union[GenerationResponse, ToolCallResponse], None]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model_and_tokenizer first.")

        # Extract the prompt from the messages
        # task_params.messages is a list of ChatCompletionMessage
        # We'll just take the last user message for now
        prompt = task_params.messages[-1].content
        if not isinstance(prompt, str):
             # Handle list of content blocks if necessary, but simple string for now
             prompt = str(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        # Generate (synchronously for now)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=task_params.max_tokens or 50)
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Yield a single response for now (since generate is not streaming)
        # In a real implementation, we'd use a Streamer
        yield GenerationResponse(
            text=generated_text,
            token=0, # Dummy token ID
            finish_reason="stop",
            stats=None
        )
