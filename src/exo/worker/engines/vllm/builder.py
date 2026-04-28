import contextlib
import os
from collections.abc import Generator
from dataclasses import dataclass

from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.engines.vllm.engine import VllmEngine
from exo.worker.engines.vllm.generator import VllmBatchEngine, load_vllm_engine
from exo.worker.runner.bootstrap import logger


@dataclass
class VllmBuilder(Builder):
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]

    def connect(self, bound_instance: BoundInstance) -> None:
        raise NotImplementedError(
            "Multiple node VLLM instances are not supported at the moment!"
        )

    def load(
        self,
        bound_instance: BoundInstance,
    ) -> Generator[ModelLoadingResponse]:
        from exo.worker.engines.vllm.kv_connector import (
            ExoKVProducerConnector,
            _patch_gdn_capture,
            _patch_vllm_for_connector,
        )

        # Apply bypass patches before vLLM init reads its connector registry
        # and the unifier touches hybrid kv-cache specs.
        _patch_vllm_for_connector(ExoKVProducerConnector)
        _patch_gdn_capture()

        kv_connector_cls: type[object] | None = ExoKVProducerConnector
        # overlapping = not os.environ.get("EXO_NO_OVERLAPPING_PREFILL_SENDS")

        def on_layer_loaded(loaded: int, total: int) -> None:
            pass

        self._bound_runner_id = bound_instance.bound_runner_id
        self._engine, self._tool_parser = load_vllm_engine(
            model_id=self.model_id,
            trust_remote_code=bound_instance.bound_shard.model_card.trust_remote_code,
            n_layers=bound_instance.bound_shard.model_card.n_layers,
            on_layer_loaded=on_layer_loaded,
            kv_connector_cls=kv_connector_cls,
        )
        return
        yield

    def build(self) -> Engine:
        gen = VllmBatchEngine(
            engine=self._engine,
            model_id=self.model_id,
        )
        try:
            max_concurrent = (
                1
                if bool(os.getenv("EXO_NO_BATCH", False))
                else EXO_MAX_CONCURRENT_REQUESTS
            )
        except Exception:
            max_concurrent = EXO_MAX_CONCURRENT_REQUESTS

        logger.info(f"using VllmEngine (max_concurrent={max_concurrent})")
        return VllmEngine(
            tool_parser=self._tool_parser,
            model_id=self.model_id,
            cancel_receiver=self.cancel_receiver,
            event_sender=self.event_sender,
            _gen=gen,
            max_concurrent_requests=max_concurrent,
        )

    def close(self) -> None:
        with contextlib.suppress(NameError, AttributeError):
            del self._engine, self._tool_parser
