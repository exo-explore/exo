import contextlib
import os
from dataclasses import dataclass
from typing import Callable, Self

import mlx.core as mx
from exo_core.engine import EngineBuilder
from exo_core.tokenizers.tool_parsers import make_mlx_parser
from exo_core.types.chunks import ErrorChunk, PrefillProgressChunk
from exo_core.types.common import CommandId, ModelId
from exo_core.types.instances import BoundInstance
from exo_core.types.runner_response import GenerationResponse, ToolCallResponse
from exo_core.types.tasks import TaskId, TextGeneration
from exo_core.utils.channels import MpReceiver, MpSender
from loguru import logger
from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_engine.batch_generator import BatchGenerator, SequentialGenerator
from mlx_engine.cache import KVPrefixCache
from mlx_engine.generator.batch_generate import ExoBatchGenerator
from mlx_engine.generator.generate import (
    mlx_generate,
    warmup_inference,
)
from mlx_engine.types import Model
from mlx_engine.utils_mlx import initialize_mlx, load_mlx_items


@dataclass
class MlxBuilder(EngineBuilder[BoundInstance, TextGeneration, GenerationResponse | ToolCallResponse]):
    model_id: ModelId
    bound_instance: BoundInstance
    event_sender: MpSender[tuple[CommandId, ErrorChunk | PrefillProgressChunk]]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None

    @classmethod
    def create(
        cls,
        bound_instance: BoundInstance,
        event_sender: MpSender[tuple[CommandId, ErrorChunk | PrefillProgressChunk]],
        cancel_receiver: MpReceiver[TaskId],
    ) -> Self:
        return cls(
            bound_instance.instance.shard_assignments.model_id,
            bound_instance,
            event_sender,
            cancel_receiver,
        )

    def connect(self) -> None:
        self.group = initialize_mlx(self.bound_instance)

    def load(
        self,
        on_timeout: Callable[[], None],
        on_layer_loaded: Callable[[int, int], None],
    ) -> None:
        self.inference_model, self.tokenizer = load_mlx_items(
            self.bound_instance,
            self.group,
            on_timeout=on_timeout,
            on_layer_loaded=on_layer_loaded,
        )

    def build(self) -> SequentialGenerator | BatchGenerator:
        assert self.inference_model
        assert self.tokenizer

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        kv_prefix_cache = KVPrefixCache(self.group)

        from functools import partial

        device_rank = 0 if self.group is None else self.group.rank()
        generate_fn = partial(
            mlx_generate, model=self.inference_model, tokenizer=self.tokenizer
        )
        warmup_fn = partial(
            warmup_inference,
            model=self.inference_model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )

        if os.environ.get("EXO_NO_BATCH"):
            logger.info("using SequentialGenerator (batching disabled)")
            return SequentialGenerator(
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                _generate_fn=generate_fn,
                _warmup_fn=warmup_fn,
            )

        logger.info("using BatchGenerator")
        gen = ExoBatchGenerator(
            model=self.inference_model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=kv_prefix_cache,
            model_id=self.model_id,
        )
        return BatchGenerator(
            tokenizer=self.tokenizer,
            group=self.group,
            tool_parser=tool_parser,
            kv_prefix_cache=kv_prefix_cache,
            model_id=self.model_id,
            device_rank=device_rank,
            cancel_receiver=self.cancel_receiver,
            event_sender=self.event_sender,
            _gen=gen,
        )

    def close(self):
        with contextlib.suppress(NameError, AttributeError):
            del self.inference_model, self.tokenizer
