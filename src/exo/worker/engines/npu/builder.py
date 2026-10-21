from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

from loguru import logger

from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine

from .device import NpuDeviceProfile
from .engine import NpuEngine
from .runtime import NpuExecutionSession


@dataclass
class NpuBuilder(Builder):
    """
    Builder responsible for detecting, loading, compiling, and instantiating
    dedicated NPU/Apple ANE execution engines.
    """

    model_id: ModelId
    event_sender: MpSender[Event] | None = None
    cancel_receiver: MpReceiver[TaskId] | None = None
    device_profile: NpuDeviceProfile | None = None
    session: NpuExecutionSession | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        """Initialize NPU hardware profile and execution context."""
        logger.info(
            f"Connecting NpuBuilder for runner {bound_instance.bound_runner_id} "
            f"on node {bound_instance.bound_node_id}"
        )
        if self.device_profile is None:
            self.device_profile = NpuDeviceProfile.detect_local_device()

        logger.info(
            f"Discovered NPU Architecture: {self.device_profile.architecture} "
            f"({self.device_profile.tera_operations_per_second} TOPS, "
            f"Efficiency: {self.device_profile.thermal_efficiency_factor}x)"
        )

    def load(
        self,
        bound_instance: BoundInstance,
    ) -> Generator[ModelLoadingResponse]:
        """
        Load model weights, compile subgraphs for NPU execution, and report progress.
        """
        assert self.device_profile is not None
        model_card = bound_instance.bound_shard.model_card
        total_layers = model_card.n_layers or 16

        logger.info(
            f"Loading and compiling model {self.model_id} onto NPU "
            f"(Total layers: {total_layers})"
        )

        # Initialize execution session
        self.session = NpuExecutionSession(
            device_profile=self.device_profile,
            model_path=str(self.model_id),
            hidden_dimension=model_card.hidden_size or 2048,
        )

        for layer_index in range(1, total_layers + 1):
            # Report loading progress per layer
            yield ModelLoadingResponse(layers_loaded=layer_index, total=total_layers)

        self.session.initialize_graph()
        logger.info("NPU model compilation completed.")

    def build(self) -> Engine:
        """Instantiate ready-to-serve NpuEngine."""
        assert self.session is not None
        return NpuEngine(model_id=self.model_id, session=self.session)

    def close(self) -> None:
        """Clean up builder and session resources."""
        if self.session is not None:
            self.session.close()
            self.session = None
        logger.info("NpuBuilder closed.")
