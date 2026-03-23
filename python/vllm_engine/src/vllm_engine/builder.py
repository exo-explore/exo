from dataclasses import dataclass
from typing import Self, Callable
from exo_core.constants import EXO_MODELS_DIR
from exo_core.engine import EngineBuilder, Engine
from exo_core.types.common import ModelId
from exo_core.types.instances import BoundInstance
from exo_core.types.tasks import TextGeneration
from exo_core.types.runner_response import GenerationResponse
from vllm_engine.vllm_generator import VllmBatchEngine
from vllm_engine.vllm_generator import load_vllm_engine


@dataclass
class VllmBuilder(EngineBuilder[BoundInstance, TextGeneration, GenerationResponse]):
    model_id: ModelId
    model_path: str
    trust_remote_code: bool
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    bound_instance: BoundInstance

    @classmethod
    def create(
        cls,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        cancel_receiver: MpReceiver[TaskId],
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
