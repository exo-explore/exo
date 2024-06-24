from abc import ABC, abstractmethod
from typing import List
from inference.shard import Shard
from networking.peer_handle import PeerHandle
from .topology import Topology

class PartitioningStrategy(ABC):
    @abstractmethod
    def next_shard(self, current_shard: Shard, topology: Topology, node_stats: dict) -> Shard:
        pass
