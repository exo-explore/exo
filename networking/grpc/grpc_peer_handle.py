import grpc
import numpy as np
from typing import Optional

# These would be generated from the .proto file
from . import node_service_pb2
from . import node_service_pb2_grpc

from ..peer_handle import PeerHandle
from inference.shard import Shard
from topology.topology import Topology
from topology.device_capabilities import DeviceCapabilities

class GRPCPeerHandle(PeerHandle):
    def __init__(self, id: str, address: str, device_capabilities: DeviceCapabilities):
        self._id = id
        self.address = address
        self._device_capabilities = device_capabilities

    def id(self) -> str:
        return self._id

    def device_capabilities(self) -> DeviceCapabilities:
        return self._device_capabilities

    async def connect(self):
        self.channel = grpc.aio.insecure_channel(self.address)
        self.stub = node_service_pb2_grpc.NodeServiceStub(self.channel)

    async def disconnect(self):
        await self.channel.close()

    async def send_prompt(self, shard: Shard, prompt: str) -> Optional[np.array]:
        request = node_service_pb2.PromptRequest(prompt=prompt, shard=node_service_pb2.Shard(model_id=shard.model_id, start_layer=shard.start_layer, end_layer=shard.end_layer, n_layers=shard.n_layers))
        response = await self.stub.SendPrompt(request)
        print(f"Sent prompt to {self.address}: {prompt}")

        if not response.tensor_data or not response.shape or not response.dtype:
            return None

        return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

    async def send_tensor(self, shard: Shard, tensor: np.ndarray) -> Optional[np.array]:
        request = node_service_pb2.TensorRequest(
            shard=node_service_pb2.Shard(model_id=shard.model_id, start_layer=shard.start_layer, end_layer=shard.end_layer, n_layers=shard.n_layers),
            tensor = node_service_pb2.Tensor(
                tensor_data=tensor.tobytes(),
                shape=tensor.shape,
                dtype=str(tensor.dtype)
            ),
        )
        response = await self.stub.SendTensor(request)

        if not response.tensor_data or not response.shape or not response.dtype:
            return None

        return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

    async def reset_shard(self, shard: Shard) -> None:
        request = node_service_pb2.ResetShardRequest(shard=node_service_pb2.Shard(model_id=shard.model_id, start_layer=shard.start_layer, end_layer=shard.end_layer, n_layers=shard.n_layers))
        await self.stub.ResetShard(request)
        print(f"Reset shard {shard} on {self.address}")

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
