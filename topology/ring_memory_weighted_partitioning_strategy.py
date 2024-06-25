from typing import List
from .partitioning_strategy import PartitioningStrategy
from inference.shard import Shard
from .topology import Topology
from .partitioning_strategy import Partition

class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
    def partition(self, topology: Topology) -> List[Partition]:
        nodes = list(topology.all_nodes())
        nodes.sort(key=lambda x: x[0])
        total_memory = sum(node[1].memory for node in nodes)
        partitions = []
        start = 0
        for node in nodes:
            end = start + (node[1].memory / total_memory)
            partitions.append(Partition(node[0], start, end))
            start = end
        return partitions
