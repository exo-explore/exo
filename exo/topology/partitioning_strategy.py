from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from exo.inference.shard import Shard
from exo.networking.peer_handle import PeerHandle
from .topology import Topology

# Partitions shard-space into pieces of contiguous shards, represented by floating point range [start, end) between 0 and 1
@dataclass
class Partition:
    node_id: str
    start: float
    end: float

class PartitioningStrategy(ABC):
    def node_id(self) -> str:
        pass

class PartitioningStrategy(ABC):
    @abstractmethod
    def partition(self, topology: Topology) -> List[Partition]:
        pass
