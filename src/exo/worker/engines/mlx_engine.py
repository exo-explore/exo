
from collections.abc import Callable, Generator

from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.base_engine import (
    DistributedGroup,
    Engine,
    KVCache,
    Model,
    Tokenizer,
)
from exo.worker.engines.mlx import Model as MlxModel
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
from exo.worker.engines.mlx.utils_mlx import (
    Group,
    apply_chat_template,
    detect_thinking_prompt_suffix,
    initialize_mlx,
    load_mlx_items,
    mx_any,
)
from exo.worker.runner.bootstrap import logger

import mlx.core as mx


class MlxEngine(Engine):
   

    def __init__(self) -> None:
       
        self._group: mx.distributed.Group | None = None

    def initialize_distributed_group(
        self, bound_instance: BoundInstance
    ) -> DistributedGroup | None:
        
        # Single-node instances don't need distributed init
        if len(bound_instance.instance.shard_assignments.node_to_runner) <= 1:
            return None

        self._group = initialize_mlx(bound_instance)
        return self._group

    def load_model_and_tokenizer(
        self,
        bound_instance: BoundInstance,
        group: DistributedGroup | None,
        on_timeout: Callable[[], None] | None = None,
    ) -> tuple[Model, Tokenizer]:
      
        # Cast DistributedGroup to MLX Group type
        mlx_group: Group | None = group if isinstance(group, (Group, type(None))) else None
        return load_mlx_items(bound_instance, mlx_group, on_timeout=on_timeout)

    def warmup_inference(
        self,
        model: Model,
        tokenizer: Tokenizer,
        group: DistributedGroup | None,
    ) -> int:
      
        assert isinstance(model, MlxModel)
        # Cast DistributedGroup to MLX Group type
        mlx_group: Group | None = group if isinstance(group, (Group, type(None))) else None
        return warmup_inference(model=model, tokenizer=tokenizer, group=mlx_group)

    def generate(
        self,
        model: Model,
        tokenizer: Tokenizer,
        task: TextGenerationTaskParams,
        prompt: str,
        kv_cache: KVCache | None = None,
        group: DistributedGroup | None = None,
    ) -> Generator[GenerationResponse]:
       
        assert isinstance(model, MlxModel)
        kv_prefix_cache = kv_cache if isinstance(kv_cache, KVPrefixCache) else None
        # Cast DistributedGroup to MLX Group type
        mlx_group: Group | None = group if isinstance(group, (Group, type(None))) else None
        return mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=kv_prefix_cache,
            group=mlx_group,
        )

    def apply_chat_template(
        self, tokenizer: Tokenizer, task_params: TextGenerationTaskParams
    ) -> str:
     
        return apply_chat_template(tokenizer, task_params)

    def detect_thinking_prompt_suffix(
        self, prompt: str, tokenizer: Tokenizer
    ) -> bool:
        
        return detect_thinking_prompt_suffix(prompt, tokenizer)

    def any_cancel(self, want_to_cancel: bool, group: DistributedGroup | None) -> bool:
      
        # Cast DistributedGroup to MLX Group type
        mlx_group: Group | None = group if isinstance(group, (Group, type(None))) else None
        return mx_any(want_to_cancel, mlx_group)

    def create_kv_cache(self, group: DistributedGroup | None) -> KVCache | None:
       
        # Cast DistributedGroup to MLX Group type
        mlx_group: Group | None = group if isinstance(group, (Group, type(None))) else None
        return KVPrefixCache(mlx_group)

    def cleanup(self) -> None:
       
        mx.clear_cache()
        import gc

        gc.collect()
