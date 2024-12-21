import grpc
from concurrent import futures
import numpy as np
from asyncio import CancelledError

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
    self.server = grpc.aio.server(
      futures.ThreadPoolExecutor(max_workers=10),
      options=[
        ("grpc.max_metadata_size", 32*1024*1024),
        ("grpc.max_send_message_length", 128*1024*1024),
        ("grpc.max_receive_message_length", 128*1024*1024),
      ],
    )
    node_service_pb2_grpc.add_NodeServiceServicer_to_server(self, self.server)
    listen_addr = f"{self.host}:{self.port}"
    self.server.add_insecure_port(listen_addr)
    await self.server.start()
    if DEBUG >= 1: print(f"Server started, listening on {listen_addr}")

  async def stop(self) -> None:
    if self.server:
      try:
        await self.server.stop(grace=5)
        await self.server.wait_for_termination()
      except CancelledError:
        pass
      if DEBUG >= 1: print("Server stopped and all connections are closed")

  async def SendPrompt(self, request, context):
    shard = Shard(
      model_id=request.shard.model_id,
      start_layer=request.shard.start_layer,
      end_layer=request.shard.end_layer,
      n_layers=request.shard.n_layers,
    )
    prompt = request.prompt
    request_id = request.request_id
    result = await self.node.process_prompt(shard, prompt, request_id)
    if DEBUG >= 5: print(f"SendPrompt {shard=} {prompt=} {request_id=} result: {result}")
    tensor_data = result.tobytes() if result is not None else None
    return node_service_pb2.Tensor(tensor_data=tensor_data, shape=result.shape, dtype=str(result.dtype)) if result is not None else node_service_pb2.Tensor()

  async def SendTensor(self, request, context):
    shard = Shard(
      model_id=request.shard.model_id,
      start_layer=request.shard.start_layer,
      end_layer=request.shard.end_layer,
      n_layers=request.shard.n_layers,
    )
    tensor = np.frombuffer(request.tensor.tensor_data, dtype=np.dtype(request.tensor.dtype)).reshape(request.tensor.shape)
    request_id = request.request_id

    result = await self.node.process_tensor(shard, tensor, request_id)
    if DEBUG >= 5: print(f"SendTensor tensor {shard=} {tensor=} {request_id=} result: {result}")
    tensor_data = result.tobytes() if result is not None else None
    return node_service_pb2.Tensor(tensor_data=tensor_data, shape=result.shape, dtype=str(result.dtype)) if result is not None else node_service_pb2.Tensor()
  
  async def SendExample(self, request, context):
    shard = Shard(
      model_id=request.shard.model_id,
      start_layer=request.shard.start_layer,
      end_layer=request.shard.end_layer,
      n_layers=request.shard.n_layers,
    )
    example = np.frombuffer(request.example.tensor_data, dtype=np.dtype(request.example.dtype)).reshape(request.example.shape)
    target = np.frombuffer(request.target.tensor_data, dtype=np.dtype(request.target.dtype)).reshape(request.target.shape)
    length = np.frombuffer(request.length.tensor_data, dtype=np.dtype(request.length.dtype)).reshape(request.length.shape)
    train = request.train
    request_id = request.request_id

    if train and not shard.is_first_layer():
      loss, grad = await self.node.process_example(shard, example, target, length, train, request_id)
      tensor_data = grad.tobytes()
      grad_tensor = node_service_pb2.Tensor(tensor_data=tensor_data, shape=grad.shape, dtype=str(grad.dtype))
      return node_service_pb2.Loss(loss=loss, grads=grad_tensor)
    else:
      loss = await self.node.process_example(shard, example, target, length, train, request_id)
      return node_service_pb2.Loss(loss=loss, grads=None)
    
  async def CollectTopology(self, request, context):
    max_depth = request.max_depth
    visited = set(request.visited)
    topology = self.node.current_topology
    nodes = {
      node_id:
        node_service_pb2.DeviceCapabilities(
          model=cap.model,
          chip=cap.chip,
          memory=cap.memory,
          flops=node_service_pb2.DeviceFlops(fp32=cap.flops.fp32, fp16=cap.flops.fp16, int8=cap.flops.int8),
        )
      for node_id, cap in topology.nodes.items()
    }
    peer_graph = {
      node_id: node_service_pb2.PeerConnections(
        connections=[
          node_service_pb2.PeerConnection(to_id=conn.to_id, description=conn.description)
          for conn in connections
        ]
      )
      for node_id, connections in topology.peer_graph.items()
    }
    if DEBUG >= 5: print(f"CollectTopology {max_depth=} {visited=} {nodes=} {peer_graph=}")
    return node_service_pb2.Topology(nodes=nodes, peer_graph=peer_graph)

  async def SendResult(self, request, context):
    request_id = request.request_id
    result = request.result
    is_finished = request.is_finished
    if DEBUG >= 5: print(f"Received SendResult request: {request_id=} {result=} {is_finished=}")
    self.node.on_token.trigger_all(request_id, result, is_finished)
    return node_service_pb2.Empty()

  async def SendOpaqueStatus(self, request, context):
    request_id = request.request_id
    status = request.status
    if DEBUG >= 8: print(f"Received SendOpaqueStatus request: {request_id=} {status=}")
    self.node.on_opaque_status.trigger_all(request_id, status)
    return node_service_pb2.Empty()

  async def HealthCheck(self, request, context):
    return node_service_pb2.HealthCheckResponse(is_healthy=True)
