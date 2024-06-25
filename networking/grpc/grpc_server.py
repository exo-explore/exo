import grpc
from concurrent import futures
import numpy as np

from . import node_service_pb2
from . import node_service_pb2_grpc
from inference.shard import Shard

from orchestration import Node

class GRPCServer(node_service_pb2_grpc.NodeServiceServicer):
    def __init__(self, node: Node, host: str, port: int):
        self.node = node
        self.host = host
        self.port = port
        self.server = None

    async def start(self) -> None:
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_metadata_size', 128*1024)
        ])
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
        shard = Shard(model_id=request.shard.model_id, start_layer=request.shard.start_layer, end_layer=request.shard.end_layer, n_layers=request.shard.n_layers)
        prompt = request.prompt
        result = await self.node.process_prompt(shard, prompt)
        tensor_data = result.tobytes() if result is not None else None
        return node_service_pb2.Tensor(tensor_data=tensor_data, shape=result.shape, dtype=str(result.dtype))

    async def SendTensor(self, request, context):
        shard = Shard(model_id=request.shard.model_id, start_layer=request.shard.start_layer, end_layer=request.shard.end_layer, n_layers=request.shard.n_layers)
        tensor = np.frombuffer(request.tensor.tensor_data, dtype=np.dtype(request.tensor.dtype)).reshape(request.tensor.shape)
        result = await self.node.process_tensor(shard, tensor)
        print("SendTensor tensor result", result)
        tensor_data = result.tobytes() if result is not None else None
        return node_service_pb2.Tensor(tensor_data=tensor_data, shape=result.shape, dtype=str(result.dtype))

    async def ResetShard(self, request, context):
        shard = Shard(model_id=request.shard.model_id, start_layer=request.shard.start_layer, end_layer=request.shard.end_layer, n_layers=request.shard.n_layers)
        print(f"Received ResetShard request: {shard}")
        await self.node.reset_shard(shard)
        return node_service_pb2.Empty()

    async def CollectTopology(self, request, context):
        max_depth = request.max_depth
        topology = await self.node.collect_topology(max_depth)
        nodes = {node_id: node_service_pb2.DeviceCapabilities(model=cap.model, chip=cap.chip, memory=cap.memory) for node_id, cap in topology.nodes.items()}
        peer_graph = {node_id: node_service_pb2.Peers(peer_ids=peers) for node_id, peers in topology.peer_graph.items()}
        return node_service_pb2.Topology(nodes=nodes, peer_graph=peer_graph)
