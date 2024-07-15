import grpc
import numpy as np
from typing import Optional, Tuple

# These would be generated from the .proto file
from . import node_service_pb2
from . import node_service_pb2_grpc

from ..peer_handle import PeerHandle
from exo.inference.shard import Shard
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities

class GRPCPeerHandle(PeerHandle):
    def __init__(self, id: str, address: str, device_capabilities: DeviceCapabilities):
        self._id = id
        self.address = address
        self._device_capabilities = device_capabilities
        self.channel = None
        self.stub = None

    def id(self) -> str:
        return self._id

    def device_capabilities(self) -> DeviceCapabilities:
        return self._device_capabilities

    async def connect(self):
        self.channel = grpc.aio.insecure_channel(self.address, options=[
            ('grpc.max_metadata_size', 32*1024*1024)
        ])
        self.stub = node_service_pb2_grpc.NodeServiceStub(self.channel)

    async def is_connected(self) -> bool:
        return self.channel is not None and self.channel.get_state() == grpc.ChannelConnectivity.READY

    async def disconnect(self):
        if self.channel:
            await self.channel.close()
        self.channel = None
        self.stub = None

    async def send_prompt(self, shard: Shard, prompt: str, request_id: Optional[str] = None) -> Optional[np.array]:
        request = node_service_pb2.PromptRequest(prompt=prompt, shard=node_service_pb2.Shard(model_id=shard.model_id, start_layer=shard.start_layer, end_layer=shard.end_layer, n_layers=shard.n_layers), request_id=request_id)
        response = await self.stub.SendPrompt(request)

        if not response.tensor_data or not response.shape or not response.dtype:
            return None

        return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

    async def send_tensor(self, shard: Shard, tensor: np.ndarray, request_id: Optional[str] = None) -> Optional[np.array]:
        request = node_service_pb2.TensorRequest(
            shard=node_service_pb2.Shard(model_id=shard.model_id, start_layer=shard.start_layer, end_layer=shard.end_layer, n_layers=shard.n_layers),
            tensor = node_service_pb2.Tensor(
                tensor_data=tensor.tobytes(),
                shape=tensor.shape,
                dtype=str(tensor.dtype)
            ),
            request_id=request_id
        )
        response = await self.stub.SendTensor(request)

        if not response.tensor_data or not response.shape or not response.dtype:
            return None

        return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

    async def get_inference_result(self, request_id: str) -> Tuple[Optional[np.ndarray], bool]:
        request = node_service_pb2.GetInferenceResultRequest(request_id=request_id)
        response = await self.stub.GetInferenceResult(request)
        if response.tensor is None:
            return None, response.is_finished
        return np.frombuffer(response.tensor.tensor_data, dtype=np.dtype(response.tensor.dtype)).reshape(response.tensor.shape), response.is_finished

    async def reset_shard(self, shard: Shard) -> None:
        request = node_service_pb2.ResetShardRequest(shard=node_service_pb2.Shard(model_id=shard.model_id, start_layer=shard.start_layer, end_layer=shard.end_layer, n_layers=shard.n_layers))
        await self.stub.ResetShard(request)

    async def collect_topology(self, max_depth: int) -> Topology:
        request = node_service_pb2.CollectTopologyRequest(max_depth=max_depth)
        response = await self.stub.CollectTopology(request)
        topology = Topology()
        for node_id, capabilities in response.nodes.items():
            device_capabilities = DeviceCapabilities(model=capabilities.model, chip=capabilities.chip, memory=capabilities.memory)
            topology.update_node(node_id, device_capabilities)
        for node_id, peers in response.peer_graph.items():
            for peer_id in peers.peer_ids:
                topology.add_edge(node_id, peer_id)
        return topology
