from dataclasses import dataclass
from collections.abc import Generator
from exo.worker.engines.base import Builder, Engine
from exo.worker.engines.vllm.vllm_generator import load_vllm_engine
from exo.shared.types.common import ModelId
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.events import Event
from exo.utils.channels import MpReceiver, MpSender
from mlx_lm.tokenizer_utils import TokenizerWrapper
from exo.disaggregated.streaming_connector import StreamingConnector
from exo.disaggregated.batch_connector import BatchConnector


@dataclass
class VllmBuilder(Builder):
    model_id: ModelId
    model_path: str
    trust_remote_code: bool
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]

    def connect(self, bound_instance: BoundInstance) -> None:
        raise NotImplementedError(
            "Multiple node VLLM instances are not supported at the moment!"
        )

    def load(
        self,
        bound_instance: BoundInstance,
    ) -> Generator[ModelLoadingResponse]:
        kv_connector_cls: type[object] | None = None
        overlapping = not os.environ.get("EXO_NO_OVERLAPPING_PREFILL_SENDS")
        if overlapping:
            kv_connector_cls = StreamingConnector
        else:
            kv_connector_cls = BatchConnector

        self._bound_runner_id = bound_instance.bound_runner_id
        self._engine, self._tool_parser, self._prefix_cache = load_vllm_engine(
            model_path=self.model_path,
            model_id=self.model_id,
            trust_remote_code=self.trust_remote_code,
            n_layers=bound_instance.bound_shard.model_card.n_layers,
            on_layer_loaded=on_layer_loaded,
            kv_connector_cls=kv_connector_cls,
        )
        return
        yield

    def build(self) -> Engine:
        from exo.worker.engines.vllm.vllm_generator import VllmBatchEngine

        gen = VllmBatchEngine(
            engine=self._engine,
            model_id=self.model_id,
            prefix_cache=self._prefix_cache,
        )
        tokenizer = TokenizerWrapper(self._engine.get_tokenizer())
        max_concurrent = 1 if os.environ.get("EXO_NO_BATCH") else 8

        from exo.master.placement import random_ephemeral_port

        prefill_port = random_ephemeral_port()
        overlapping = not os.environ.get("EXO_NO_OVERLAPPING_PREFILL_SENDS")
        try:
            from exo.disaggregated.prefill_server import start_prefill_server

            from exo.shared.types.events import RunnerStatusUpdated
            from exo.shared.types.worker.runners import RunnerReady, RunnerRunning

            runner_id = self._bound_runner_id

            def _on_prefill_status(running: bool) -> None:
                port = prefill_port
                if running:
                    self.event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id,
                            runner_status=RunnerRunning(prefill_server_port=port),
                        )
                    )
                else:
                    self.event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id,
                            runner_status=RunnerReady(prefill_server_port=port),
                        )
                    )

            self._prefill_server = start_prefill_server(
                engine=self._engine,
                bind_address="0.0.0.0",
                port=prefill_port,
                overlapping=overlapping,
                prefix_cache=self._prefix_cache,
                on_status_change=_on_prefill_status,
            )
            self._prefill_server_port = prefill_port
        except Exception:
            logger.opt(exception=True).warning("Failed to start prefill server")
            self._prefill_server = None
            self._prefill_server_port = None

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
            if hasattr(self, "_prefill_server") and self._prefill_server is not None:
                self._prefill_server.shutdown()
            del self._engine, self._prefix_cache, self._tool_parser
