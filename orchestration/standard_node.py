from typing import List, Optional
import numpy as np
from networking import Discovery, PeerHandle, Server
from inference.inference_engine import InferenceEngine, Shard
from .node import Node
from topology.topology import Topology
from topology.device_capabilities import device_capabilities
from topology.partitioning_strategy import PartitioningStrategy
from topology.partitioning_strategy import Partition

class StandardNode(Node):
    def __init__(self, id: str, server: Server, inference_engine: InferenceEngine, discovery: Discovery, partitioning_strategy: PartitioningStrategy = None):
        self.id = id
        self.inference_engine = inference_engine
        self.server = server
        self.discovery = discovery
        self.partitioning_strategy = partitioning_strategy
        self.peers: List[PeerHandle] = {}
        self.topology: Topology = Topology()
        self.device_capabilities = device_capabilities()

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

    async def process_prompt(self, shard: Shard, prompt: str) -> Optional[np.array]:
        print("Process prompt", shard, prompt)
        result = await self.inference_engine.infer_prompt(shard, prompt)
        print(f"Got result from prompt: {prompt}. Result: {result}")

        await self.forward_tensor_to_next_shard(shard, result)

        return result

    async def process_tensor(self, shard: Shard, tensor: np.ndarray) -> None:
        print("Process tensor", shard, tensor)
        result = await self.inference_engine.infer_tensor(shard, tensor)
        print(f"Got result from tensor: {len(tensor)}. Result: {result}")

        await self.forward_tensor_to_next_shard(shard, result)

        return result

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
                target_peer = next((p for p in self.peers if p.id() == next_partition.node_id), None)
                if not target_peer:
                    raise ValueError(f"Peer for {next_partition} not found")

                start_layer = int(next_partition.start * shard.n_layers)
                end_layer = int(next_partition.end * shard.n_layers) - 1
                next_shard = Shard(shard.model_id, start_layer, end_layer, shard.n_layers)

                print(f"Sending tensor to {target_peer.id()} for shard: {next_shard}")

                await target_peer.send_tensor(next_shard, tensor)

    async def reset_shard(self, shard: Shard) -> None:
        # Implement shard reset logic
        print(f"Resetting shard: {shard}")
        await self.inference_engine.reset_shard(shard)

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
