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
      futures.ThreadPoolExecutor(max_workers=32),
      options=[
        ("grpc.max_metadata_size", 32*1024*1024),
        ("grpc.max_send_message_length", 256*1024*1024),
        ("grpc.max_receive_message_length", 256*1024*1024),
        ("grpc.keepalive_time_ms", 10000),
        ("grpc.keepalive_timeout_ms", 5000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 10000),
        ("grpc.http2.min_ping_interval_without_data_ms", 5000),
        ("grpc.max_concurrent_streams", 100),
        ("grpc.tcp_nodelay", 1),
        ("grpc.optimization_target", "throughput"),
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
    sequence_number = request.sequence_number if hasattr(request, 'sequence_number') else None
    trace_parent = request.trace_parent if hasattr(request, 'trace_parent') else None
    
    # Update trace context if sequence number or trace parent is provided
    if sequence_number is not None or trace_parent is not None:
      from exo.orchestration.tracing import tracer, TraceContext
      context = TraceContext(
        request_id=request_id,
        sequence_number=sequence_number or 0,
        trace_parent=trace_parent
      )
      tracer.set_context(request_id, context)
    
    await self.node.process_prompt(shard, prompt, request_id)
    if DEBUG >= 5: print(f"SendPrompt {shard=} {prompt=} {request_id=} {sequence_number=}")
    return node_service_pb2.Empty()

  async def SendTensor(self, request, context):
    shard = Shard(
      model_id=request.shard.model_id,
      start_layer=request.shard.start_layer,
      end_layer=request.shard.end_layer,
      n_layers=request.shard.n_layers,
    )
    tensor = np.frombuffer(request.tensor.tensor_data, dtype=np.dtype(request.tensor.dtype)).reshape(request.tensor.shape)
    request_id = request.request_id
    sequence_number = request.sequence_number if hasattr(request, 'sequence_number') else None
    trace_parent = request.trace_parent if hasattr(request, 'trace_parent') else None
    
    # Update trace context if sequence number or trace parent is provided
    if sequence_number is not None or trace_parent is not None:
      from exo.orchestration.tracing import tracer, TraceContext
      context = TraceContext(
        request_id=request_id,
        sequence_number=sequence_number or 0,
        trace_parent=trace_parent
      )
      tracer.set_context(request_id, context)
    
    await self.node.process_tensor(shard, tensor, request_id)
    if DEBUG >= 5: print(f"SendTensor tensor {shard=} {tensor=} {request_id=} {sequence_number=}")
    return node_service_pb2.Empty()
  
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
    sequence_number = request.sequence_number if hasattr(request, 'sequence_number') else None
    trace_parent = request.trace_parent if hasattr(request, 'trace_parent') else None
    
    # Update trace context if sequence number or trace parent is provided
    if sequence_number is not None or trace_parent is not None:
      from exo.orchestration.tracing import tracer, TraceContext
      context = TraceContext(
        request_id=request_id,
        sequence_number=sequence_number or 0,
        trace_parent=trace_parent
      )
      tracer.set_context(request_id, context)

    if train and not shard.is_first_layer():
      loss, grad = await self.node.process_example(shard, example, target, length, train, request_id)
      tensor_data = grad.tobytes()
      grad_tensor = node_service_pb2.Tensor(tensor_data=tensor_data, shape=grad.shape, dtype=str(grad.dtype))
      return node_service_pb2.Loss(loss=loss, grads=grad_tensor)
    else:
      loss = await self.node.process_example(shard, example, target, length, train, request_id)
      return node_service_pb2.Loss(loss=loss, grads=None)
    
  async def CollectTopology(
    self,
    request: node_service_pb2.CollectTopologyRequest,
    context: grpc.aio.ServicerContext,
  ) -> node_service_pb2.CollectTopologyResponse:
    # Convert visited list back to set
    visited = set(request.visited)
    if DEBUG >= 2: print(f"[GRPCServer] CollectTopology request with {visited=} {request.max_depth=}")
    
    # Get topology from node
    topology = await self.node.collect_topology(visited, request.max_depth)
    if DEBUG >= 2: print(f"[GRPCServer] Got topology: {topology}")
    
    # Convert Topology to proto message
    proto_topology = node_service_pb2.CollectTopologyResponse.Topology()
    
    # Convert nodes and their capabilities
    for node_id, capabilities in topology.nodes.items():
      # Create DeviceFlops
      flops = node_service_pb2.CollectTopologyResponse.DeviceFlops(
        fp32=capabilities.flops.fp32,
        fp16=capabilities.flops.fp16,
        int8=capabilities.flops.int8
      )
      
      # Create DeviceCapabilities
      device_caps = node_service_pb2.CollectTopologyResponse.DeviceCapabilities(
        model=capabilities.model,
        chip=capabilities.chip,
        memory=capabilities.memory,
        flops=flops
      )
      
      # Get connections for this node
      connections = []
      if node_id in topology.peer_graph:
        for conn in topology.peer_graph[node_id]:
          proto_conn = node_service_pb2.CollectTopologyResponse.PeerConnection(
            to_id=conn.to_id,
            description=conn.description if conn.description else None
          )
          connections.append(proto_conn)
      
      # Create Node with its connections
      node = node_service_pb2.CollectTopologyResponse.Node(
        id=node_id,
        capabilities=device_caps,
        connections=connections
      )
      proto_topology.nodes.append(node)
    
    # Set active node if present
    if topology.active_node_id:
      proto_topology.active_node_id = topology.active_node_id
    
    if DEBUG >= 2: print(f"[GRPCServer] Sending topology response with {len(proto_topology.nodes)} nodes")
    return node_service_pb2.CollectTopologyResponse(topology=proto_topology)

  async def SendNewToken(self, request, context):
    request_id = request.request_id
    token = request.token
    is_finished = request.is_finished
    sequence_number = request.sequence_number if hasattr(request, 'sequence_number') else None
    trace_parent = request.trace_parent if hasattr(request, 'trace_parent') else None
    
    # Update trace context if sequence number or trace parent is provided
    if sequence_number is not None or trace_parent is not None:
      from exo.orchestration.tracing import tracer, TraceContext
      context = TraceContext(
        request_id=request_id,
        sequence_number=sequence_number or 0,
        trace_parent=trace_parent
      )
      tracer.set_context(request_id, context)
    
    if DEBUG >= 5: print(f"Received SendNewToken request: {request_id=} {token=} {is_finished=} {sequence_number=}")
    self.node.on_token.trigger_all(request_id, token, is_finished)
    return node_service_pb2.Empty()

  async def SendOpaqueStatus(self, request, context):
    request_id = request.request_id
    status = request.status
    trace_parent = request.trace_parent if hasattr(request, 'trace_parent') else None
    
    # Update trace context if trace parent is provided
    if trace_parent is not None:
      from exo.orchestration.tracing import tracer, TraceContext
      context = TraceContext(
        request_id=request_id,
        sequence_number=0,
        trace_parent=trace_parent
      )
      tracer.set_context(request_id, context)
    
    if DEBUG >= 8: print(f"Received SendOpaqueStatus request: {request_id=} {status=}")
    self.node.on_opaque_status.trigger_all(request_id, status)
    return node_service_pb2.Empty()

  async def HealthCheck(self, request, context):
    return node_service_pb2.HealthCheckResponse(is_healthy=True)
