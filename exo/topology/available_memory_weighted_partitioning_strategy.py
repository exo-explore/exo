from exo.topology.partitioning_strategy import PartitioningStrategy
from exo.topology.topology import Topology
from exo.topology.partitioning_strategy import Partition

class AvailableMemoryWeightedPartitioningStrategy(PartitioningStrategy):
    def partition(self, topology: Topology) -> list[Partition]:
        nodes = list(topology.nodes.items())
        
        # Filter out nodes with zero available memory
        nodes = [(node_id, cap) for node_id, cap in nodes if cap.memory_available > 0]
        
        if not nodes:
            # If all nodes have zero available memory, return a single partition
            return [Partition(node_id=list(topology.nodes.keys())[0], start=0.0, end=1.0)]
        
        total_memory = sum(cap.memory_available for _, cap in nodes)
        partitions = []
        current_start = 0.0
        
        for node_id, cap in nodes:
            partition_size = cap.memory_available / total_memory
            current_end = current_start + partition_size
            partitions.append(Partition(node_id=node_id, start=current_start, end=current_end))
            current_start = current_end
        
        return partitions