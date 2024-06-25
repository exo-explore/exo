from typing import List, Optional, Callable
import numpy as np
from networking import Discovery, PeerHandle, Server
from inference.inference_engine import InferenceEngine, Shard
from .node import Node
from topology.topology import Topology
from topology.device_capabilities import device_capabilities
from topology.partitioning_strategy import PartitioningStrategy
from topology.partitioning_strategy import Partition

class StandardNode(Node):
    def __init__(self, id: str, server: Server, inference_engine: InferenceEngine, discovery: Discovery, partitioning_strategy: PartitioningStrategy = None, on_token: Callable[[List[int]], None] = None, max_generate_tokens: int = 50):
        self.id = id
        self.inference_engine = inference_engine
        self.server = server
        self.discovery = discovery
        self.partitioning_strategy = partitioning_strategy
        self.peers: List[PeerHandle] = {}
        self.topology: Topology = Topology()
        self.device_capabilities = device_capabilities()
        self.buffered_token_output: List[int] = []
        self.on_token = on_token
        self.max_generate_tokens = max_generate_tokens

    async def start(self, wait_for_peers: int = 0) -> None:
        await self.server.start()
        await self.discovery.start()
        self.peers = await self.discovery.discover_peers(wait_for_peers)
        print(f"Starting with the following peers: {self.peers}")
        print("Connecting to peers...")
        for peer in self.peers:
            await peer.connect()
            print(f"Connected to {peer.id()}")
        await self.collect_topology()
        print(f"Collected topology: {self.topology}")

    async def stop(self) -> None:
        await self.discovery.stop()
        await self.server.stop()

    async def process_prompt(self, shard: Shard, prompt: str) -> Optional[np.ndarray]:
        print("process prompt", shard, prompt)
        result, is_finished = await self.inference_engine.infer_prompt(self.get_current_shard(shard), prompt)

        print(f"result size: {result.size}, is finished: {is_finished}")
        if result.size == 1:
            self.buffered_token_output.append(result.item())
            self.on_token(self.buffered_token_output)

        if not is_finished and len(self.buffered_token_output) < self.max_generate_tokens:
            await self.forward_tensor_to_next_shard(shard, result)

        return np.array(self.buffered_token_output) if self.buffered_token_output else None

    async def process_tensor(self, shard: Shard, tensor: np.ndarray) -> Optional[np.ndarray]:
        result, is_finished = await self.inference_engine.infer_tensor(self.get_current_shard(shard), tensor)

        print(f"result size: {result.size}, is finished: {is_finished}")
        if result.size == 1:
            self.buffered_token_output.append(result.item())
            self.on_token(self.buffered_token_output)

        if not is_finished and len(self.buffered_token_output) < self.max_generate_tokens:
            await self.forward_tensor_to_next_shard(shard, result)

        return np.array(self.buffered_token_output) if self.buffered_token_output else None

    async def forward_tensor_to_next_shard(self, shard: Shard, tensor: np.ndarray) -> None:
        if not self.partitioning_strategy:
            print("No partitioning strategy found. Skipping forward.")
            return

        partitions = self.partitioning_strategy.partition(self.topology)
        current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
        print(f"Current partition index: {current_partition_index}")
        if current_partition_index is not None:
            next_partition_index = (current_partition_index + 1) % len(partitions)
            next_partition: Partition = partitions[next_partition_index]
            print(f"Computed next from: {shard}, {self.topology}. Next partition: {next_partition}")

            if next_partition:
                if next_partition.node_id == self.id:
                    await self.process_tensor(shard, tensor)
                    return

                target_peer = next((p for p in self.peers if p.id() == next_partition.node_id), None)
                if not target_peer:
                    raise ValueError(f"Peer for {next_partition} not found")

                start_layer = int(next_partition.start * shard.n_layers)
                end_layer = int(next_partition.end * shard.n_layers) - 1
                next_shard = Shard(shard.model_id, start_layer, end_layer, shard.n_layers)

                print(f"Sending tensor to {target_peer.id()} for shard: {next_shard}")

                await target_peer.send_tensor(next_shard, tensor)

    def get_current_shard(self, shard: Shard) -> Shard:
        partitions = self.partitioning_strategy.partition(self.topology)
        current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
        if current_partition_index is None:
            raise ValueError(f"No current partition found for node: {self.id}")

        current_partition = partitions[current_partition_index]
        start_layer = int(current_partition.start * shard.n_layers)
        end_layer = int(current_partition.end * shard.n_layers) - 1
        return Shard(shard.model_id, start_layer, end_layer, shard.n_layers)


    async def reset_shard(self, shard: Shard) -> None:
        # Implement shard reset logic
        print(f"Resetting shard: {shard}")
        self.buffered_token_output = []
        await self.inference_engine.reset_shard(self.get_current_shard(shard))

    async def collect_topology(self, max_depth: int = 4) -> Topology:
        self.topology.update_node(self.id, self.device_capabilities)

        for peer in self.peers:
            self.topology.update_node(peer.id(), peer.device_capabilities())
            self.topology.add_edge(self.id, peer.id())

            if max_depth > 0:
                other_topology = await peer.collect_topology(max_depth = max_depth - 1)
                print(f"Collected topology from: {peer.id()}: {other_topology}")
                self.topology.merge(other_topology)

        return self.topology
