from typing import Protocol

from shared.types.graphs.topology import Topology
from shared.types.models.common import ModelId
from shared.types.models.model import ModelInfo
from shared.types.models.sources import ModelSource
from shared.types.worker.common import InstanceId
from shared.types.worker.downloads import DownloadProgress
from shared.types.worker.instances import Instance


class ClusterAPI(Protocol):
    def get_topology(self) -> Topology: ...

    def list_instances(self) -> list[Instance]: ...

    def get_instance(self, instance_id: InstanceId) -> Instance: ...

    def create_instance(self, model_id: ModelId) -> InstanceId: ...

    def remove_instance(self, instance_id: InstanceId) -> None: ...

    def get_model_data(self, model_id: ModelId) -> ModelInfo: ...

    def download_model(self, model_id: ModelId, model_source: ModelSource) -> None: ...

    def get_download_progress(self, model_id: ModelId) -> DownloadProgress: ...
