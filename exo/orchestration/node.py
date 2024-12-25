import numpy as np
import json
import asyncio
import uuid
import time
import traceback
from typing import List, Dict, Optional, Tuple, Union, Set
from exo.networking import Discovery, PeerHandle, Server
from exo.inference.inference_engine import InferenceEngine, Shard
from exo.topology.topology import Topology
from exo.topology.device_capabilities import device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.topology.partitioning_strategy import Partition, PartitioningStrategy, map_partitions_to_shards
from exo import DEBUG
from exo.helpers import AsyncCallbackSystem
from exo.viz.topology_viz import TopologyViz
from exo.download.hf.hf_helpers import RepoProgressEvent
from exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.orchestration.tracing import tracer, TraceContext

class Node:
  def __init__(
    self,
    _id: str,
    server: Server,
    inference_engine: InferenceEngine,
    discovery: Discovery,
    partitioning_strategy: PartitioningStrategy = None,
    max_generate_tokens: int = 1024,
    default_sample_temperature: float = 0.0,
    topology_viz: Optional[TopologyViz] = None,
    shard_downloader: Optional[HFShardDownloader] = None,
  ):
    self.id = _id
    self.inference_engine = inference_engine
    self.server = server
    self.discovery = discovery
    self.partitioning_strategy = partitioning_strategy
    self.peers: List[PeerHandle] = {}
    self.topology: Topology = Topology()
    self.device_capabilities = UNKNOWN_DEVICE_CAPABILITIES
    self.buffered_token_output: Dict[str, Tuple[List[int], bool]] = {}
    self.buffered_logits: Dict[str, List[np.ndarray]] = {}
    self.buffered_inputs: Dict[str, List[np.ndarray]] = {}
    self.buffered_partials: Dict[str, List[np.ndarray]] = {}
    self.checkpoints: Dict[str, Dict[str, int]] = {}
    
    self.max_generate_tokens = max_generate_tokens
    self.topology_viz = topology_viz
    self.default_sample_temperature = default_sample_temperature
    self._on_token = AsyncCallbackSystem[str, Tuple[str, int, bool]]()
    self._on_opaque_status = AsyncCallbackSystem[str, Tuple[str, str]]()
    self._on_opaque_status.register("node_status").on_next(self.on_node_status)
    self.node_download_progress: Dict[str, RepoProgressEvent] = {}
    self.topology_inference_engines_pool: List[List[str]] = []
    self.shard_downloader = shard_downloader
    self.outstanding_requests = {}

  async def start(self, wait_for_peers: int = 0) -> None:
    self.device_capabilities = await device_capabilities()
    await self.server.start()
    await self.discovery.start()
    await self.update_peers(wait_for_peers)
    await self.collect_topology(set())
    if DEBUG >= 2: print(f"Collected topology: {self.topology}")
    asyncio.create_task(self.periodic_topology_collection(2.0))

  async def stop(self) -> None:
    await self.discovery.stop()
    await self.server.stop()

  def on_node_status(self, request_id, opaque_status):
    try:
      status_data = json.loads(opaque_status)
      status_type = status_data.get("type", "")
      if status_type == "supported_inference_engines":
        node_id = status_data.get("node_id")
        engines = status_data.get("engines", [])
        self.topology_inference_engines_pool.append(engines)
      elif status_type == "node_status":
        if status_data.get("status", "").startswith("start_"):
          self.current_topology.active_node_id = status_data.get("node_id")
        elif status_data.get("status", "").startswith("end_"):
          if status_data.get("node_id") == self.current_topology.active_node_id:
            self.current_topology.active_node_id = None

      download_progress = None
      if status_type == "download_progress":
        if DEBUG >= 8: print(f"Download progress from {status_data.get('node_id')}: {status_data.get('progress')}")
        download_progress = RepoProgressEvent.from_dict(status_data.get('progress'))
        self.node_download_progress[status_data.get('node_id')] = download_progress

      if self.topology_viz:
        self.topology_viz.update_visualization(self.topology, self.partitioning_strategy.partition(self.topology), self.id, self.node_download_progress)
    except Exception as e:
      if DEBUG >= 1: print(f"Error on_node_status: {e}")
      if DEBUG >= 1: traceback.print_exc()

  def get_supported_inference_engines(self):
    supported_engine_names = []
    if self.inference_engine.__class__.__name__ == 'MLXDynamicShardInferenceEngine':
      supported_engine_names.append('mlx')
      supported_engine_names.append('tinygrad')
    else:
      supported_engine_names.append('tinygrad')
    return supported_engine_names

  async def broadcast_supported_engines(self, supported_engines_names: List[str]):
    status_message = json.dumps({"type": "supported_inference_engines", "node_id": self.id, "engines": supported_engines_names})
    await self.broadcast_opaque_status("", status_message)

  def get_topology_inference_engines(self) -> List[List[str]]:
    return self.topology_inference_engines_pool
  
  async def process_inference_result(
    self,
    shard,
    result: np.ndarray,
    request_id: Optional[str] = None,
  ):
    context = tracer.get_context(request_id)
    if not context:
      context = TraceContext(request_id=request_id or str(uuid.uuid4()), sequence_number=0)
      tracer.set_context(request_id, context)

    is_finished = False
    try:
      with tracer.start_span(
        f"process_inference_result.{self.get_partition_index()}",
        context,
        extra_attributes={
          "partition_index": self.get_partition_index(),
          "node_id": self.id,
          "start_layer": shard.start_layer,
          "end_layer": shard.end_layer
        }
      ):
        if request_id not in self.buffered_token_output:
          self.buffered_token_output[request_id] = ([], False)
        
        if shard.is_last_layer():
          is_finished = len(self.buffered_token_output[request_id][0]) >= self.max_generate_tokens

          # Add span for sampling
          with tracer.start_span(
            "sample_token",
            context,
            extra_attributes={
              "temperature": self.default_sample_temperature,
              "result_shape": str(result.shape)
            }
          ):
            token = await self.inference_engine.sample(result, temp=self.default_sample_temperature)
          
          # Add span for tensor reshaping
          with tracer.start_span(
            "reshape_token",
            context,
            extra_attributes={
              "input_shape": str(token.shape),
              "target_shape": "(1, -1)"
            }
          ):
            forward = token.reshape(1, -1)
          
          # Increment sequence number for next forward pass
          next_sequence = context.sequence_number + 1
          # Create new context but preserve request span
          next_context = TraceContext(
            request_id=context.request_id, 
            sequence_number=next_sequence,
            request_span=context.request_span  # Preserve request span
          )
          tracer.set_context(request_id, next_context)
          
          # Add span for token processing
          with tracer.start_span(
            "process_token",
            context,
            extra_attributes={
              "token_value": token.item(),
              "sequence_number": context.sequence_number
            }
          ):
            self.buffered_token_output[request_id][0].append(token.item())
            is_finished = token.item() == self.inference_engine.tokenizer.eos_token_id or is_finished
            self.trigger_on_token_callbacks(request_id, token.item(), is_finished)
            await self.broadcast_new_token(request_id, token.item(), is_finished)
          
          if not is_finished:
            self.outstanding_requests[request_id] = "waiting"
            asyncio.create_task(self.forward_tensor(shard, forward, request_id, self.get_partition_index(offset = 1)))
        else:
          forward = result
          if not is_finished:
            self.outstanding_requests[request_id] = "waiting"
            asyncio.create_task(self.forward_tensor(shard, forward, request_id, self.get_partition_index(offset = 1)))

        if is_finished:
          # End the request span when generation is complete
          if context.request_span:
            context.request_span.set_attribute("total_tokens", len(self.buffered_token_output[request_id][0]))
            context.request_span.end()
            context.request_span = None
          self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
          self.outstanding_requests.pop(request_id)


        return np.array(self.buffered_token_output[request_id][0])
    except Exception as e:
      if request_id in self.outstanding_requests:
        self.outstanding_requests.pop(request_id)
      # End request span on error
      if context and context.request_span:
        context.request_span.set_status(Status(StatusCode.ERROR, str(e)))
        context.request_span.end()
        context.request_span = None
      raise

  async def process_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: Optional[str] = None,
  ) -> None:
    shard = self.get_current_shard(base_shard)
    start_time = time.perf_counter_ns()
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "start_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
        }),
      )
    )
    await self._process_prompt(base_shard, prompt, request_id)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "end_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
        }),
      )
    )
    if DEBUG >= 2: print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=} {elapsed_time_ns=}")

  async def _process_prompt(self, base_shard: Shard, prompt: str, request_id: Optional[str] = None) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
      
    # Create or get trace context
    context = tracer.get_context(request_id)
    if not context:
      # Create new context with request span
      request_span = tracer.tracer.start_span(
        "request",
        attributes={
          "request_id": request_id,
          "prompt": prompt,
          "node_id": self.id,
          "request_type": "process_prompt"
        }
      )
      context = TraceContext(
        request_id=request_id,
        sequence_number=0,
        request_span=request_span,
        current_span=request_span,
        trace_parent=tracer.inject_context(request_span)
      )
      tracer.set_context(request_id, context)
      
    shard = self.get_current_shard(base_shard)
    if DEBUG >= 2: print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=}")

    try:
      if not shard.is_first_layer():
        if DEBUG >= 2: print(f"[{request_id}] forwarding to next shard: {base_shard=} {shard=} {prompt=}")
        self.outstanding_requests[request_id] = "waiting"
        await self.forward_prompt(shard, prompt, request_id, 0)
        return None

      self.outstanding_requests[request_id] = "processing"
      # Add span for prompt inference
      with tracer.start_span(
        "infer_prompt",
        context,
        extra_attributes={
          "prompt_length": len(prompt),
          "shard_layers": f"{shard.start_layer}-{shard.end_layer}"
        }
      ):
        result = await self.inference_engine.infer_prompt(request_id, shard, prompt)
      
      # Add span for prompt result processing
      with tracer.start_span(
        "process_prompt_result",
        context,
        extra_attributes={
          "result_shape": str(result.shape)
        }
      ):
        await self.process_inference_result(shard, result, request_id)
    except Exception as e:
      if context.request_span:
        context.request_span.set_status(Status(StatusCode.ERROR, str(e)))
      raise

  async def enqueue_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray, 
    length: np.ndarray,
    request_id: Optional[str] = None,
    train: bool = False,
  ):
    shard = self.get_current_shard(base_shard)
    if shard.is_first_layer():
      loss = await self.process_example(shard, example, target, length, train, request_id)
      return loss
    else:
      if request_id is None:
        request_id = str(uuid.uuid4())
      self.outstanding_requests[request_id] = "waiting"
      loss = await self.forward_example(shard, example, target, length, train, request_id, 0) 
    return loss

  async def coordinate_save(
    self,
    base_shard: Shard,
    iteration: int,
    destination: str,
  ):
    shard = self.get_current_shard(base_shard)
    model = shard.model_id
    sid = shard.__hash__()
    path = f"{destination}/{model}/{sid}-{iteration}.safetensors"
    self.outstanding_requests[f"{sid}::{iteration}"] = "Checking"
    if model not in self.checkpoints:
      self.checkpoints[model] = {}
    if sid not in self.checkpoints[model]:
      self.checkpoints[model][sid] = []
    if len(self.checkpoints[model][sid]) < 1 or self.checkpoints[model][sid][-1] < iteration:
      print(f"Saving checkpoint to {path}")
      self.outstanding_requests[f"{sid}::{iteration}"] = "Saving"
      import os
      os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
      await self.inference_engine.save_checkpoint(shard, path)
      self.checkpoints[model][sid] = sorted(self.checkpoints[model][sid] + [iteration])
    self.outstanding_requests.pop(f"{sid}::{iteration}")

  async def process_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray, 
    length: np.ndarray,
    train: bool = False,
    request_id: Optional[str] = None,
  ):
    shard = self.get_current_shard(base_shard)
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": f"start_{'train' if train else 'eval'}_example",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "example_size": example.size,
          "example_shape": example.shape,
          "request_id": request_id,
        }),
      )
    )
    start_time = time.perf_counter_ns()
    resp = await self._process_example(shard, example, target, length, train, request_id)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": f"end_{'train' if train else 'eval'}_example",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
        }),
      )
    )
    return resp

  async def _process_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray, 
    length: np.ndarray,
    train: bool = False,
    request_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)
    if DEBUG >= 1: print(f"[{request_id}] process_example: {example.shape=}")
    try:
      target = target.astype(int)
      if train:
        if shard.is_last_layer():
          self.outstanding_requests[request_id] = "training"
          loss, grad = await self.inference_engine.train(request_id, shard, example, target, length)
        else:
          self.outstanding_requests[request_id] = "preprocessing"
          step = await self.inference_engine.infer_tensor(request_id, shard, example)
          self.outstanding_requests[request_id] = "waiting"
          loss, backgrad = await self.forward_example(shard, step, target, length, train, request_id, self.get_partition_index(offset = 1))
          self.outstanding_requests[request_id] = "training"
          partial_loss, grad = await self.inference_engine.train(request_id, shard, example, backgrad, length, loss="back_gradient")
        self.outstanding_requests.pop(request_id)
        if shard.is_first_layer():
          return loss
        else:
          return loss, grad
      else:
        if shard.is_last_layer():
          self.outstanding_requests[request_id] = "evaluating"
          loss = await self.inference_engine.evaluate(request_id, shard, example, target, length)
        else:
          self.outstanding_requests[request_id] = "preprocessing"
          step = await self.inference_engine.infer_tensor(request_id, shard, example)
          self.outstanding_requests[request_id] = "waiting"
          loss = await self.forward_example(shard, step, target, length, train, request_id, self.get_partition_index(offset = 1))
        self.outstanding_requests.pop(request_id)
        return loss
    except Exception as e:
      self.outstanding_requests.pop(request_id)
      print(f"Error processing example for shard {shard}: {e}")
      traceback.print_exc()
      return None
        
  async def process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
  ):
    context = tracer.get_context(request_id)
    if not context:
      context = TraceContext(request_id=request_id or str(uuid.uuid4()), sequence_number=0)
      tracer.set_context(request_id, context)

    try:
      self.outstanding_requests[request_id] = "processing"
      with tracer.start_span(
        f"process_tensor.{self.get_partition_index()}",
        context,
        extra_attributes={
          "partition_index": self.get_partition_index(),
          "node_id": self.id,
          "start_layer": base_shard.start_layer,
          "end_layer": base_shard.end_layer,
          "tensor_shape": str(tensor.shape)
        }
      ):
        # Add span for tensor inference
        with tracer.start_span(
          "infer_tensor",
          context,
          extra_attributes={
            "input_shape": str(tensor.shape),
            "shard_layers": f"{base_shard.start_layer}-{base_shard.end_layer}"
          }
        ):
          result = await self.inference_engine.infer_tensor(request_id, base_shard, tensor)
        
        # Add span for inference result processing
        with tracer.start_span(
          "process_result",
          context,
          extra_attributes={
            "result_shape": str(result.shape)
          }
        ):
          await self.process_inference_result(base_shard, result, request_id)
    except Exception as e:
      if request_id in self.outstanding_requests:
        self.outstanding_requests.pop(request_id)
      if context and context.request_span:
        context.request_span.set_status(Status(StatusCode.ERROR, str(e)))
      print(f"Error processing tensor for shard {base_shard}: {e}")
      traceback.print_exc()
      raise

  async def forward_example(
    self,
    base_shard: Shard,
    step: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    train: bool,
    request_id: str,
    target_index: int,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    target_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"computed target from: {base_shard} {target_index}, {self.topology}. target shard: {target_shard}")
    target_peer = next((p for p in self.peers if p.id() == target_id), None)
    if not target_peer:
      raise ValueError(f"peer for {target_index} not found")
    if DEBUG >= 1: print(f"sending example to {target_peer.id()}: {step} => {target} ({length})")
    resp = await target_peer.send_example(target_shard, step, target, length, request_id=request_id, train=train)
    return resp

  async def forward_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: str,
    target_index: int,
  ) -> None:
    context = tracer.get_context(request_id)
    if not context:
      context = TraceContext(request_id=request_id, sequence_number=0)
      tracer.set_context(request_id, context)

    with tracer.start_span(
      "forward_prompt",
      context,
      extra_attributes={
        "source_node": self.id,
        "target_index": target_index,
        "prompt": prompt
      }
    ) as span:
      if DEBUG >= 1: print(f"target partition index: {target_index}")
      target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
      next_shard = self.get_current_shard(base_shard, target_index)
      span.set_attribute("target_node", target_id)
      
      # Get trace context for propagation
      trace_parent = tracer.inject_context(span)
      
      if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. next shard: {next_shard}")
      if target_id == self.id:
        # Update local context with trace parent
        context.trace_parent = trace_parent
        await self.process_prompt(next_shard, prompt, request_id)
      else:
        target_peer = next((p for p in self.peers if p.id() == target_id), None)
        if not target_peer:
          raise ValueError(f"Peer for {target_index} not found")
        if DEBUG >= 1: print(f"Sending prompt to {target_peer.id()}: {prompt}")
        await target_peer.send_prompt(next_shard, prompt, request_id=request_id, sequence_number=context.sequence_number, trace_parent=trace_parent)
  
  async def forward_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: str,
    target_index: int,
  ):
    context = tracer.get_context(request_id)
    if not context:
      context = TraceContext(request_id=request_id, sequence_number=0)
      tracer.set_context(request_id, context)

    with tracer.start_span(
      "forward_tensor",
      context,
      extra_attributes={
        "source_node": self.id,
        "target_index": target_index,
        "tensor_shape": str(tensor.shape)
      }
    ) as span:
      target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
      next_shard = self.get_current_shard(base_shard, target_index)
      span.set_attribute("target_node", target_id)
      
      # Get trace context for propagation
      trace_parent = tracer.inject_context(context.request_span or span)
      
      if target_id == self.id:
        # Update local context with trace parent
        context.trace_parent = trace_parent
        await self.process_tensor(next_shard, tensor, request_id)
      else:
        target_peer = next((p for p in self.peers if p.id() == target_id), None)
        if not target_peer:
          raise ValueError(f"Peer for {target_index} not found")
        
        if DEBUG >= 1: print(f"Sending tensor to {target_peer.id()}: {tensor}")
        await target_peer.send_tensor(next_shard, tensor, request_id=request_id, sequence_number=context.sequence_number, trace_parent=trace_parent)

  def get_partition_index(self, offset: int = 0):
    if not self.partitioning_strategy:
      if DEBUG >= 1: print("No partitioning strategy found. Skipping forward.")
      return None
    partitions = self.partitioning_strategy.partition(self.topology)
    current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
    if current_partition_index is None:
      raise ValueError(f"No current partition found for node: {self.id}")
    return (current_partition_index + offset) % len(partitions)

  def get_current_shard(self, base_shard: Shard, index: Optional[int] = None) -> Shard:
    if index is None:
      index = self.get_partition_index()
    partitions = self.partitioning_strategy.partition(self.topology)
    shards = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id)
    return shards[index]

  async def update_peers(self, wait_for_peers: int = 0) -> bool:
    next_peers = await self.discovery.discover_peers(wait_for_peers)
    current_peer_ids = {peer.id() for peer in self.peers}
    next_peer_ids = {peer.id() for peer in next_peers}
    peers_added = [peer for peer in next_peers if peer.id() not in current_peer_ids]
    peers_removed = [peer for peer in self.peers if peer.id() not in next_peer_ids]
    peers_updated = [peer for peer in next_peers if peer.id() in current_peer_ids and any(p.addr() != peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_unchanged = [peer for peer in next_peers if peer.id() in current_peer_ids and all(p.addr() == peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_to_disconnect = [peer for peer in peers_removed if await peer.is_connected()]
    peers_to_connect = [peer for peer in peers_added + peers_updated + peers_unchanged if not await peer.is_connected()]

    def _pretty(peers: List[PeerHandle]) -> List[str]:
      return [f"{peer.id()}@{peer.addr()}" for peer in peers]

    if DEBUG >= 2:
      print(f"update_peers: added={peers_added} removed={peers_removed} updated={peers_updated} unchanged={peers_unchanged} to_disconnect={peers_to_disconnect} to_connect={peers_to_connect}")

    async def disconnect_with_timeout(peer, timeout=5):
      try:
        await asyncio.wait_for(peer.disconnect(), timeout)
        return True
      except Exception as e:
        print(f"Error disconnecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    async def connect_with_timeout(peer, timeout=5):
      try:
        await asyncio.wait_for(peer.connect(), timeout)
        return True
      except Exception as e:
        print(f"Error connecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    disconnect_results = await asyncio.gather(*(disconnect_with_timeout(peer) for peer in peers_to_disconnect), return_exceptions=True)
    connect_results = await asyncio.gather(*(connect_with_timeout(peer) for peer in peers_to_connect), return_exceptions=True)

    successful_disconnects = [peer for peer, result in zip(peers_to_disconnect, disconnect_results) if result is True]
    failed_disconnects = [peer for peer, result in zip(peers_to_disconnect, disconnect_results) if result is False]
    successful_connects = [peer for peer, result in zip(peers_to_connect, connect_results) if result is True]
    failed_connects = [peer for peer, result in zip(peers_to_connect, connect_results) if result is False]
    if DEBUG >= 1:
      if successful_disconnects: print(f"Successfully disconnected peers: {_pretty(successful_disconnects)}")
      if failed_disconnects: print(f"Failed to disconnect peers: {_pretty(failed_disconnects)}")
      if successful_connects: print(f"Successfully connected peers: {_pretty(successful_connects)}")
      if failed_connects: print(f"Failed to connect peers: {_pretty(failed_connects)}")

    self.peers = next_peers
    return len(peers_added) > 0 or len(peers_removed) > 0 or len(peers_updated) > 0

  async def select_best_inference_engine(self):
    if self.inference_engine.__class__.__name__ == 'DummyInferenceEngine': return
    supported_engines = self.get_supported_inference_engines()
    await self.broadcast_supported_engines(supported_engines)
    if len(self.get_topology_inference_engines()):
      self.inference_engine = get_inference_engine(supported_engines[0], self.shard_downloader)

  async def periodic_topology_collection(self, interval: int):
    while True:
      await asyncio.sleep(interval)
      try:
        did_peers_change = await self.update_peers()
        if DEBUG >= 2: print(f"{did_peers_change=}")
        await self.collect_topology(set())
        if did_peers_change:
          await self.select_best_inference_engine()
      except Exception as e:
        print(f"Error collecting topology: {e}")
        traceback.print_exc()

  async def collect_topology(self, visited: set[str], max_depth: int = 4) -> Topology:
    next_topology = Topology()
    next_topology.update_node(self.id, self.device_capabilities)

    if DEBUG >= 2: print(f"Collecting topology {max_depth=} {visited=}")

    prev_visited = visited.copy()
    visited.add(self.id)
    visited.update(p.id() for p in self.peers)

    for peer in self.peers:
      next_topology.update_node(peer.id(), peer.device_capabilities())
      next_topology.add_edge(self.id, peer.id(), peer.description())

      if peer.id() in prev_visited:
        continue

      if max_depth <= 0:
        if DEBUG >= 2: print("Max depth reached. Skipping...")
        continue

      try:
        other_topology = await asyncio.wait_for(peer.collect_topology(visited, max_depth=max_depth - 1), timeout=5.0)
        if DEBUG >= 2: print(f"Collected topology from: {peer.id()}: {other_topology}")
        next_topology.merge(peer.id(), other_topology)
      except Exception as e:
        print(f"Error collecting topology from {peer.id()}: {e}")
        traceback.print_exc()

    next_topology.active_node_id = self.topology.active_node_id
    self.topology = next_topology
    if self.topology_viz:
      self.topology_viz.update_visualization(self.topology, self.partitioning_strategy.partition(self.topology), self.id)
    return self.topology

  @property
  def on_token(self) -> AsyncCallbackSystem[str, Tuple[str, int, bool]]:
    return self._on_token

  @property
  def on_opaque_status(self) -> AsyncCallbackSystem[str, Tuple[str, str]]:
    return self._on_opaque_status

  def trigger_on_token_callbacks(self, request_id: str, token: int, is_finished: bool) -> None:
    if DEBUG >= 2: print(f"[Node] Triggering token callbacks: {request_id=} {token=} {is_finished=}")
    self.on_token.trigger_all(request_id, token, is_finished)
  
  async def broadcast_new_token(self, request_id: str, token: int, is_finished: bool):
    """Broadcast a new token to all peers."""
    context = tracer.get_context(request_id)
    if context:
      # Handle token in tracer for grouping
      tracer.handle_token(context, token, is_finished)
      # Get current trace context for propagation
      trace_parent = ""
      if context.current_span:
        trace_parent = tracer.inject_context(context.current_span)
    
    tasks = []
    for peer in self.peers:
      tasks.append(
        peer.send_new_token(
          request_id,
          token,
          is_finished,
          context.sequence_number if context else 0,
          trace_parent if context else ""
        )
      )
    await asyncio.gather(*tasks)

  async def broadcast_opaque_status(self, request_id: str, status: str) -> None:
    if DEBUG >= 8: print(f"Broadcasting opaque status: {request_id=} {status=}")

    async def send_status_to_peer(peer):
      try:
        await asyncio.wait_for(peer.send_opaque_status(request_id, status), timeout=15.0)
      except asyncio.TimeoutError:
        print(f"Timeout sending opaque status to {peer.id()}")
      except Exception as e:
        print(f"Error sending opaque status to {peer.id()}: {e}")
        traceback.print_exc()

    await asyncio.gather(*[send_status_to_peer(peer) for peer in self.peers], return_exceptions=True)
    # in the case of opaque status, we also want to receive our own opaque statuses
    # self.on_opaque_status.trigger_all(request_id, status)

  @property
  def current_topology(self) -> Topology:
    return self.topology
