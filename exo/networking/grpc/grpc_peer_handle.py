import grpc
import numpy as np
import asyncio
from typing import Optional, Tuple, List

from . import node_service_pb2
from . import node_service_pb2_grpc

from ..peer_handle import PeerHandle
from exo.inference.shard import Shard
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.helpers import DEBUG


class GRPCPeerHandle(PeerHandle):
  def __init__(self, _id: str, address: str, desc: str, device_capabilities: DeviceCapabilities):
    self._id = _id
    self.address = address
    self.desc = desc
    self._device_capabilities = device_capabilities
    self.channel = None
    self.stub = None
    self.channel_options = [
      ("grpc.max_metadata_size", 64 * 1024 * 1024),
      ("grpc.max_receive_message_length", 256 * 1024 * 1024),
      ("grpc.max_send_message_length", 256 * 1024 * 1024),
      ("grpc.max_concurrent_streams", 100),
      ("grpc.http2.min_time_between_pings_ms", 10000),
      ("grpc.keepalive_time_ms", 20000),
      ("grpc.keepalive_timeout_ms", 10000),
      ("grpc.keepalive_permit_without_calls", 1),
      ("grpc.http2.max_pings_without_data", 0),
      ("grpc.tcp_nodelay", 1),
      ("grpc.optimization_target", "throughput"),
    ]

  def id(self) -> str:
    return self._id

  def addr(self) -> str:
    return self.address

  def description(self) -> str:
    return self.desc

  def device_capabilities(self) -> DeviceCapabilities:
    return self._device_capabilities

  async def connect(self):
    if self.channel is None:
      self.channel = grpc.aio.insecure_channel(
        self.address,
        options=self.channel_options,
        compression=grpc.Compression.Gzip
      )
      self.stub = node_service_pb2_grpc.NodeServiceStub(self.channel)
    await self.channel.channel_ready()

  async def is_connected(self) -> bool:
    return self.channel is not None and self.channel.get_state() == grpc.ChannelConnectivity.READY

  async def disconnect(self):
    if self.channel:
      await self.channel.close()
    self.channel = None
    self.stub = None

  async def _ensure_connected(self):
    if not await self.is_connected():
      try:
        await asyncio.wait_for(self.connect(), timeout=10.0)
      except asyncio.TimeoutError:
        if DEBUG >= 2: print(f"Connection timeout for {self._id}@{self.address}")
        await self.disconnect()
        raise

  async def health_check(self) -> bool:
    try:
      await self._ensure_connected()
      request = node_service_pb2.HealthCheckRequest()
      response = await asyncio.wait_for(self.stub.HealthCheck(request), timeout=5)
      return response.is_healthy
    except asyncio.TimeoutError:
      return False
    except Exception:
      if DEBUG >= 4:
        print(f"Health check failed for {self._id}@{self.address}.")
        import traceback
        traceback.print_exc()
      return False

  async def send_prompt(
    self,
    shard: Shard,
    prompt: str,
    request_id: Optional[str] = None,
    sequence_number: Optional[int] = None,
    trace_parent: Optional[str] = None
  ) -> None:
    request = node_service_pb2.SendPromptRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      prompt=prompt,
      request_id=request_id,
      sequence_number=sequence_number,
      trace_parent=trace_parent
    )
    await self.stub.SendPrompt(request)

  async def send_tensor(
    self,
    shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    sequence_number: Optional[int] = None,
    trace_parent: Optional[str] = None
  ) -> None:
    request = node_service_pb2.SendTensorRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      tensor=node_service_pb2.Tensor(
        tensor_data=tensor.tobytes(),
        shape=tensor.shape,
        dtype=str(tensor.dtype)
      ),
      request_id=request_id,
      sequence_number=sequence_number,
      trace_parent=trace_parent
    )
    await self.stub.SendTensor(request)

  async def send_example(
    self,
    shard: Shard,
    example: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    train: bool,
    request_id: Optional[str] = None,
    sequence_number: Optional[int] = None,
    trace_parent: Optional[str] = None
  ) -> Optional[np.array]:
    request = node_service_pb2.SendExampleRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      example=node_service_pb2.Tensor(tensor_data=example.tobytes(), shape=example.shape, dtype=str(example.dtype)),
      target=node_service_pb2.Tensor(tensor_data=target.tobytes(), shape=target.shape, dtype=str(target.dtype)),
      length=node_service_pb2.Tensor(tensor_data=length.tobytes(), shape=length.shape, dtype=str(length.dtype)),
      train=train,
      request_id=request_id,
      sequence_number=sequence_number,
      trace_parent=trace_parent
    )
    response = await self.stub.SendExample(request)
    loss = response.loss
    if train and not shard.is_first_layer():
      grads = np.frombuffer(response.grads.tensor_data, dtype=np.dtype(response.grads.dtype)).reshape(response.grads.shape)
      return loss, grads
    else:
      return loss

  async def send_loss(self, shard: Shard, tensor: np.ndarray, request_id: Optional[str] = None) -> Optional[np.array]:
    request = node_service_pb2.TensorRequest(
      shard=node_service_pb2.Shard(
        model_id=shard.model_id,
        start_layer=shard.start_layer,
        end_layer=shard.end_layer,
        n_layers=shard.n_layers,
      ),
      tensor=node_service_pb2.Tensor(tensor_data=tensor.tobytes(), shape=tensor.shape, dtype=str(tensor.dtype)),
      request_id=request_id,
    )
    response = await self.stub.SendLoss(request)

    if not response.tensor_data or not response.shape or not response.dtype:
      return None

    return np.frombuffer(response.tensor_data, dtype=np.dtype(response.dtype)).reshape(response.shape)

  async def collect_topology(self, visited: set[str], max_depth: int = 4) -> Topology:
    if DEBUG >= 2: print(f"[GRPCPeerHandle] Collecting topology from {self.id()} with {visited=} {max_depth=}")
    
    # Convert set to list for GRPC request
    request = node_service_pb2.CollectTopologyRequest(
      visited=list(visited),
      max_depth=max_depth
    )
    
    # Make GRPC call
    response = await self.stub.CollectTopology(request)
    if DEBUG >= 2: print(f"[GRPCPeerHandle] Got topology response from {self.id()}")
    
    # Convert proto topology to Topology object
    topology = Topology()
    proto_topology = response.topology
    
    # Convert nodes and their capabilities
    for node in proto_topology.nodes:
      # Convert DeviceCapabilities
      flops = DeviceFlops(
        fp32=node.capabilities.flops.fp32,
        fp16=node.capabilities.flops.fp16,
        int8=node.capabilities.flops.int8
      )
      capabilities = DeviceCapabilities(
        model=node.capabilities.model,
        chip=node.capabilities.chip,
        memory=node.capabilities.memory,
        flops=flops
      )
      
      # Add node to topology
      topology.update_node(node.id, capabilities)
      
      # Add connections
      for conn in node.connections:
        topology.add_edge(node.id, conn.to_id, conn.description if conn.HasField("description") else None)
    
    # Set active node
    if proto_topology.HasField("active_node_id"):
      topology.active_node_id = proto_topology.active_node_id
    
    if DEBUG >= 2: print(f"[GRPCPeerHandle] Converted topology from {self.id()} with {len(topology.nodes)} nodes")
    return topology

  async def send_new_token(
    self,
    request_id: str,
    token: int,
    is_finished: bool,
    sequence_number: Optional[int] = None,
    trace_parent: Optional[str] = None
  ) -> None:
    request = node_service_pb2.SendNewTokenRequest(
      request_id=request_id,
      token=token,
      is_finished=is_finished,
      sequence_number=sequence_number,
      trace_parent=trace_parent
    )
    await self.stub.SendNewToken(request)

  async def send_opaque_status(
    self,
    request_id: str,
    status: str,
    trace_parent: Optional[str] = None
  ) -> None:
    request = node_service_pb2.SendOpaqueStatusRequest(
      request_id=request_id,
      status=status,
      trace_parent=trace_parent
    )
    await self.stub.SendOpaqueStatus(request)
