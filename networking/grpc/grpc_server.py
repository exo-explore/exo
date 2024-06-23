import grpc
from concurrent import futures
import numpy as np

from . import node_service_pb2
from . import node_service_pb2_grpc

from orchestration import Node

class GRPCServer(node_service_pb2_grpc.NodeServiceServicer):
    def __init__(self, node: Node, host: str, port: int):
        self.node = node
        self.host = host
        self.port = port
        self.server = None

    async def start(self) -> None:
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        node_service_pb2_grpc.add_NodeServiceServicer_to_server(self, self.server)
        listen_addr = f'{self.host}:{self.port}'
        self.server.add_insecure_port(listen_addr)
        await self.server.start()
        print(f"Server started, listening on {listen_addr}")

    async def stop(self) -> None:
        if self.server:
            await self.server.stop(5)  # 5 seconds grace period
            print("Server stopped")

    async def SendPrompt(self, request, context):
        prompt = request.prompt
        target = request.target if request.HasField('target') else None
        if target and target != self.node.node_id:
            await self.node.process_prompt(prompt, target)
        else:
            # Process the prompt locally
            # You'd need to implement this method in the Node class
            await self.node.process_prompt(prompt)
        return node_service_pb2.Empty()

    async def SendTensor(self, request, context):
        tensor = np.frombuffer(request.tensor_data, dtype=np.dtype(request.dtype)).reshape(request.shape)
        target = request.target if request.HasField('target') else None
        if target and target != self.node.node_id:
            await self.node.process_tensor(tensor, target)
        else:
            # Process the tensor locally
            await self.node.inference_strategy.process_inference(tensor)
        return node_service_pb2.Empty()

    async def ResetShard(self, request, context):
        print(f"Received ResetShard request: {request}")
        # TODO
        # shard_id = request.shard_id
        # You'd need to implement this method in the Node class
        # await self.node.reset_shard(shard_id)
        return node_service_pb2.Empty()
