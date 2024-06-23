import grpc
import numpy as np
from typing import Optional

# These would be generated from the .proto file
from . import node_service_pb2
from . import node_service_pb2_grpc

from ..peer_handle import PeerHandle

class GRPCPeerHandle(PeerHandle):
    def __init__(self, id: str, address: str):
        self._id = id
        self.address = address

    def id(self) -> str:
        return self._id

    async def connect(self):
        self.channel = grpc.aio.insecure_channel(self.address)
        self.stub = node_service_pb2_grpc.NodeServiceStub(self.channel)

    async def disconnect(self):
        await self.channel.close()

    async def send_prompt(self, prompt: str) -> None:
        request = node_service_pb2.PromptRequest(prompt=prompt)
        await self.stub.SendPrompt(request)
        print(f"Sent prompt to {self.address}: {prompt}")

    async def send_tensor(self, tensor: np.ndarray, target: Optional[str] = None) -> None:
        request = node_service_pb2.TensorRequest(
            tensor_data=tensor.tobytes(),
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            target=target
        )
        await self.stub.SendTensor(request)
        if target:
            print(f"Sent tensor to {self.address} with target {target}: shape {tensor.shape}")
        else:
            print(f"Sent tensor to {self.address}: shape {tensor.shape}")

    async def reset_shard(self, shard_id: str) -> None:
        request = node_service_pb2.ResetShardRequest(shard_id=shard_id)
        await self.stub.ResetShard(request)
        print(f"Reset shard {shard_id} on {self.address}")
