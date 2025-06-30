from typing import NewType
from shared.types.model import ModelId, DownloadProgress, ModelMeta, ModelSource
from shared.types.instance import InstanceId, Instance
from shared.types.model import Topology


RequestId = NewType("RequestId", str)


class ControlPlaneAPI:
    def get_cluster_state() -> ClusterState:
        ...

    def get_topology() -> Topology:
        ...

    def get_instances() -> list[Instance]:
        ...

    def get_instance(instance_id: InstanceId) -> Instance:
        ...

    def create_instance(model_id: ModelId) -> InstanceId:
        ...
    
    def delete_instance(instance_id: InstanceId) -> None:
        ...
    
    def get_model_meta(model_id: ModelId) -> ModelMeta:
        ...

    def download_model(model_source: ModelSource) -> InstanceId:
        ...
        
    def get_model_download_progress(instance_id: InstanceId) -> DownloadProgress:
        ...

    def create_chat_completion_request() -> RequestId:
        ...
