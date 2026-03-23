import contextlib
import os
from dataclasses import dataclass
from typing import Callable, Self

from exo_core.constants import EXO_MODELS_DIR
from exo_core.engine import Engine, EngineBuilder
from exo_core.types.chunks import ErrorChunk, PrefillProgressChunk
from exo_core.types.common import CommandId, ModelId
from exo_core.types.instances import BoundInstance
from exo_core.types.runner_response import GenerationResponse, ToolCallResponse
from exo_core.types.tasks import TaskId, TextGeneration
from exo_core.utils.channels import MpReceiver, MpSender
from loguru import logger
from mlx_engine.batch_generator import BatchGenerator

from vllm_engine.vllm_generator import VllmBatchEngine, load_vllm_engine


@dataclass
class VllmBuilder(EngineBuilder[BoundInstance, TextGeneration, GenerationResponse | ToolCallResponse]):
    model_id: ModelId
    model_path: str
    trust_remote_code: bool
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[tuple[CommandId, ErrorChunk | PrefillProgressChunk]]
    bound_instance: BoundInstance

    @classmethod
    def create(
        cls,
        bound_instance: BoundInstance,
    cancel_receiver: MpReceiver[TaskId],
    event_sender: MpSender[tuple[CommandId, ErrorChunk | PrefillProgressChunk]],
    ) -> Self:
        mid = bound_instance.instance.shard_assignments.model_id
        return cls(
            mid,
            str(EXO_MODELS_DIR / mid.normalize()),
            bound_instance.bound_shard.model_card.trust_remote_code,
            cancel_receiver,
            event_sender,
            bound_instance,
        )

    def connect(self) -> None:
        raise NotImplementedError(
            "Multiple node VLLM instances are not supported at the moment!"
        )

    def load(
        self,
        on_timeout: Callable[[], None],
        on_layer_loaded: Callable[[int, int], None],
    ) -> None:
        self._engine, self._tool_parser, self._prefix_cache = load_vllm_engine(
            model_path=self.model_path,
            model_id=self.model_id,
            trust_remote_code=self.trust_remote_code,
            n_layers=self.bound_instance.bound_shard.model_card.n_layers,
            on_layer_loaded=on_layer_loaded,
        )

    def build(self) -> Engine[TextGeneration, GenerationResponse]:
        gen = VllmBatchEngine(
            engine=self._engine,
            model_id=self.model_id,
            prefix_cache=self._prefix_cache,
        )
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        tokenizer = TokenizerWrapper(self._engine.get_tokenizer())
        max_concurrent = 1 if os.environ.get("EXO_NO_BATCH") else 8

        logger.info(f"using BatchGenerator (vLLM, max_concurrent={max_concurrent})")
        return BatchGenerator(
            tokenizer=tokenizer,
            group=None,
            tool_parser=self._tool_parser,
            kv_prefix_cache=None,
            model_id=self.model_id,
            device_rank=0,
            cancel_receiver=self.cancel_receiver,
            event_sender=self.event_sender,
            _gen=gen,
            max_concurrent_requests=max_concurrent,
        )

    def close(self) -> None:
        with contextlib.suppress(NameError, AttributeError):
            del self._engine, self._prefix_cache, self._tool_parser
