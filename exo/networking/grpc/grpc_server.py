import grpc
from concurrent import futures
import numpy as np

from . import node_service_pb2
from . import node_service_pb2_grpc
from exo import DEBUG
from exo.inference.shard import Shard
from exo.orchestration import Node

class GRPCServer(node_service_pb2_grpc.NodeServiceServicer):
    def __init__(self, node: Node, host: str, port: int):
        self.node = node
        self.host = host
        self.port = port
        self.server = None

    async def start(self) -> None:
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_metadata_size', 32*1024*1024)
        ])
        node_service_pb2_grpc.add_NodeServiceServicer_to_server(self, self.server)
        listen_addr = f'{self.host}:{self.port}'
        self.server.add_insecure_port(listen_addr)
        await self.server.start()
        if DEBUG >= 1: print(f"Server started, listening on {listen_addr}")

    async def stop(self) -> None:
        if self.server:
            await self.server.stop(grace=5)
            await self.server.wait_for_termination()
            if DEBUG >= 1: print("Server stopped and all connections are closed")

    async def SendPrompt(self, request, context):
        shard = Shard(model_id=request.shard.model_id, start_layer=request.shard.start_layer, end_layer=request.shard.end_layer, n_layers=request.shard.n_layers)
        prompt = request.prompt
        request_id = request.request_id
        result = await self.node.process_prompt(shard, prompt, request_id)
        if DEBUG >= 2: print(f"SendPrompt {shard=} {prompt=} {request_id=} result: {result}")
        tensor_data = result.tobytes() if result is not None else None
        return node_service_pb2.Tensor(tensor_data=tensor_data, shape=result.shape, dtype=str(result.dtype)) if result is not None else node_service_pb2.Tensor()

    async def SendTensor(self, request, context):
        shard = Shard(model_id=request.shard.model_id, start_layer=request.shard.start_layer, end_layer=request.shard.end_layer, n_layers=request.shard.n_layers)
        tensor = np.frombuffer(request.tensor.tensor_data, dtype=np.dtype(request.tensor.dtype)).reshape(request.tensor.shape)
        request_id = request.request_id
        inference_state = request.inference_state

        result = await self.node.process_tensor(shard, tensor, request_id, inference_state)
        if DEBUG >= 2: print(f"SendTensor tensor {shard=} {tensor=} {request_id=} result: {result}")
        tensor_data = result.tobytes() if result is not None else None
        return node_service_pb2.Tensor(tensor_data=tensor_data, shape=result.shape, dtype=str(result.dtype)) if result is not None else node_service_pb2.Tensor()

    async def GetInferenceResult(self, request, context):
        request_id = request.request_id
        result = await self.node.get_inference_result(request_id)
        if DEBUG >= 2: print(f"GetInferenceResult {request_id=}: {result}")
        tensor_data = result[0].tobytes() if result[0] is not None else None
        return node_service_pb2.InferenceResult(tensor=node_service_pb2.Tensor(tensor_data=tensor_data, shape=result[0].shape, dtype=str(result[0].dtype)), is_finished=result[1]) if result[0] is not None else node_service_pb2.InferenceResult(is_finished=result[1])

    async def ResetShard(self, request, context):
        shard = Shard(model_id=request.shard.model_id, start_layer=request.shard.start_layer, end_layer=request.shard.end_layer, n_layers=request.shard.n_layers)
        if DEBUG >= 2: print(f"Received ResetShard request: {shard}")
        await self.node.reset_shard(shard)
        return node_service_pb2.Empty()

    async def CollectTopology(self, request, context):
        max_depth = request.max_depth
        visited = set(request.visited)
        topology = await self.node.collect_topology(visited, max_depth)
        nodes = {node_id: node_service_pb2.DeviceCapabilities(model=cap.model, chip=cap.chip, memory=cap.memory) for node_id, cap in topology.nodes.items()}
        peer_graph = {node_id: node_service_pb2.Peers(peer_ids=peers) for node_id, peers in topology.peer_graph.items()}
        return node_service_pb2.Topology(nodes=nodes, peer_graph=peer_graph)

    async def GlobalReset(self, request, context):
        base_shard = Shard(model_id=request.base_shard.model_id, start_layer=request.base_shard.start_layer, end_layer=request.base_shard.end_layer, n_layers=request.base_shard.n_layers)
        visited = set(request.visited)
        max_depth = request.max_depth
        await self.node.global_reset(base_shard, visited, max_depth)
        return node_service_pb2.Empty()
