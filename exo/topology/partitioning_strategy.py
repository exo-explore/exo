from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from .topology import Topology

# Partitions shard-space into pieces of contiguous shards, represented by floating point range [start, end) between 0 and 1
@dataclass
class Partition:
    node_id: str
    start: float
    end: float

class PartitioningStrategy(ABC):
    @abstractmethod
    def partition(self, topology: Topology) -> List[Partition]:
        pass
