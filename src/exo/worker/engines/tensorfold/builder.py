from collections.abc import Generator
from typing import override

from exo.download.download_utils import build_model_path
from exo.shared.types.common import ModelId
from exo.shared.types.events import RunnerEvent
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance, TensorFoldInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.engines.tensorfold.capabilities import LoadedModel, probe_binary
from exo.worker.engines.tensorfold.config import NativeSettings
from exo.worker.engines.tensorfold.engine import TensorFoldEngine
from exo.worker.engines.tensorfold.process import NativeProcess


class TensorFoldBuilder(Builder):
    def __init__(
        self,
        model_id: ModelId,
        event_sender: MpSender[RunnerEvent],
        cancel_receiver: MpReceiver[TaskId],
    ):
        self._model_id = model_id
        self._cancel_receiver = cancel_receiver
        self._process: NativeProcess | None = None
        self._loaded: LoadedModel | None = None

    @override
    def connect(self, bound_instance: BoundInstance) -> None:
        raise ValueError("TensorFold instances do not join MLX distributed groups")

    @override
    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        instance = bound_instance.instance
        shard = bound_instance.bound_shard
        if not isinstance(instance, TensorFoldInstance):
            raise ValueError("TensorFold requires a TensorFold instance")
        if (
            shard.world_size != 1
            or shard.device_rank != 0
            or shard.start_layer != 0
            or shard.end_layer != shard.n_layers
            or len(instance.shard_assignments.node_to_runner) != 1
        ):
            raise ValueError(
                "TensorFold currently loads one complete model on one node"
            )
        if shard.model_card.model_id != self._model_id:
            raise ValueError("TensorFold's assigned model changed before loading")
        settings = NativeSettings.from_environment()
        capabilities = probe_binary(settings.binary)
        formats = capabilities.families.get(shard.model_card.model_type or "", [])
        if shard.model_card.tensorfold_format not in formats:
            raise ValueError(
                "TensorFold does not advertise this model architecture and format"
            )
        model_path = build_model_path(self._model_id)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"TensorFold model has not been downloaded: {model_path}"
            )
        self._process = NativeProcess(settings)
        self._loaded = self._process.start(
            model_path, str(self._model_id), instance.context_length, instance.parallel
        )
        yield ModelLoadingResponse(layers_loaded=shard.n_layers, total=shard.n_layers)

    @override
    def build(self) -> Engine:
        if self._process is None or self._loaded is None:
            raise RuntimeError(
                "TensorFold must load its model before building an engine"
            )
        process = self._process
        try:
            engine = TensorFoldEngine(process, self._loaded, self._cancel_receiver)
        except BaseException:
            process.close()
            raise
        self._process = None
        return engine

    @override
    def close(self) -> None:
        if self._process is not None:
            self._process.close()
            self._process = None
