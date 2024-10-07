from typing import List
from .partitioning_strategy import PartitioningStrategy, Partition
from .topology import Topology

class AvailableMemoryWeightedPartitioningStrategy(PartitioningStrategy):
    def partition(self, topology: Topology) -> List[Partition]:
        nodes = list(topology.all_nodes())
        # Filter out nodes with zero available memory
        nodes = [(node_id, cap) for node_id, cap in nodes if cap.available_memory > 0]

        if not nodes:
            # Handle edge case: No nodes with available memory
            return [Partition(node_id=topology.active_node_id or self.id, start=0.0, end=1.0)]

        total_available_memory = sum(cap.available_memory for _, cap in nodes)
        partitions = []
        start = 0.0
        for node_id, cap in nodes:
            proportion = cap.available_memory / total_available_memory
            end = start + proportion
            partitions.append(Partition(node_id=node_id, start=start, end=end))
            start = end
        # Ensure the last partition ends at 1.0
        partitions[-1].end = 1.0
        return partitions
