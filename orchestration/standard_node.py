from typing import List, Optional
import numpy as np
from networking import Discovery, PeerHandle, Server
from inference.inference_engine import InferenceEngine, Shard
from .node import Node

class StandardNode(Node):
    def __init__(self, id: str, server: Server, inference_engine: InferenceEngine, discovery: Discovery):
        self.id = id
        self.inference_engine = inference_engine
        self.server = server
        self.discovery = discovery
        self.peers: List[PeerHandle] = {}
        self.ring_order: List[str] = []

    async def start(self, wait_for_peers: int = 0) -> None:
        await self.server.start()
        await self.discovery.start()
        self.peers = await self.discovery.discover_peers(wait_for_peers)
        print(f"Starting with the following peers: {self.peers}")
        print("Connecting to peers...")
        for peer in self.peers:
            await peer.connect()
            print(f"Connected to {peer.id()}")

    async def stop(self) -> None:
        await self.discovery.stop()
        await self.server.stop()

    async def process_prompt(self, shard: Shard, prompt: str, target: Optional[str] = None) -> Optional[np.array]:
        print("Process prompt", shard, prompt, target)
        result = await self.inference_engine.infer_prompt(shard, prompt)
        # Implement prompt processing logic
        print(f"Got result from prompt: {prompt}. Result: {result}")
        # You might want to initiate inference here
        if target:
            target_peer = next((p for p in self.peers if p.id() == target), None)
            if not target_peer:
                raise ValueError(f"Peer {target} not found")

            await target_peer.send_tensor(result)

        return result

    async def process_tensor(self, shard: Shard, tensor: np.ndarray, target: Optional[str] = None) -> None:
        print("Process tensor", shard, tensor)
        result = await self.inference_engine.infer_shard(shard, tensor)
        # Implement prompt processing logic
        print(f"Got result from prompt: {len(tensor)}. Result: {result}")

        if target:
            target_peer = next((p for p in self.peers if p.id() == target), None)
            if not target_peer:
                raise ValueError(f"Peer {target} not found")

            await target_peer.send_tensor(result)

        return result

    async def reset_shard(self, shard: Shard) -> None:
        # Implement shard reset logic
        print(f"Resetting shard: {shard}")
        await self.inference_engine.reset_shard(shard)
