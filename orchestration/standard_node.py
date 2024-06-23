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

    async def start(self) -> None:
        await self.server.start()
        await self.discovery.start()
        self.peers = await self.discovery.discover_peers()
        print(f"Starting with the following peers: {self.peers}")
        print("Connecting to peers...")
        for peer in self.peers:
            await peer.connect()
            print(f"Connected to {peer.id()}")

    async def stop(self) -> None:
        await self.discovery.stop()
        await self.server.stop()

    async def process_tensor(self, tensor: np.ndarray, target: Optional[str] = None) -> None:
        result = await self.inference_engine.process_shard(tensor)

        if target:
            if not filter(lambda p: p.id() == target, self.peers):
                raise ValueError(f"Peer {target} not found")

            await self.peers[target].send_tensor(result)

    async def process_prompt(self, prompt: str) -> None:
        # Implement prompt processing logic
        print(f"Processing prompt: {prompt}")
        # You might want to initiate inference here

    async def reset_shard(self, shard: Shard) -> None:
        # Implement shard reset logic
        print(f"Resetting shard: {shard}")
        await self.inference_engine.reset_shard(shard)
