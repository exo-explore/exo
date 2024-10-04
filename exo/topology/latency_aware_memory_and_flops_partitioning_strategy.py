from typing import List
from exo.topology.partitioning_strategy import PartitioningStrategy, Partition
from exo.topology.topology import Topology

class LatencyAwareMemoryAndFlopsPartitioningStrategy(PartitioningStrategy):
    def partition(self, topology: Topology) -> List[Partition]:
        # Extract nodes and their capabilities
        nodes = list(topology.all_nodes())
        
        if not nodes:
            return []
        
        # Normalize memory and FLOPS
        max_memory = max(node[1].memory for node in nodes)
        max_flops = max(node[1].flops.fp32 for node in nodes)
        
        # Normalize latency and throughput
        max_latency = max(latency for _, neighbors in topology.peer_graph.items() for _, (latency, _) in neighbors.items()) if topology.peer_graph else 1
        max_throughput = max(throughput for _, neighbors in topology.peer_graph.items() for _, (_, throughput) in neighbors.items()) if topology.peer_graph else 1
        
        def normalize(value, max_value):
            return value / max_value if max_value != 0 else 0
        
        def combined_metric(node):
            # Normalize memory and FLOPS
            normalized_memory = normalize(node[1].memory, max_memory)
            normalized_flops = normalize(node[1].flops.fp32, max_flops)
            
            # Normalize latency and throughput
            neighbors = topology.get_neighbors(node[0])
            normalized_latency = normalize(max(latency for _, (latency, _) in neighbors.items()), max_latency) if neighbors else 0
            normalized_throughput = normalize(max(throughput for _, (_, throughput) in neighbors.items()), max_throughput) if neighbors else 0
            
            # Combine normalized values with weights
            # Example: Weight memory 80%, FLOPS 10%, latency 5%, throughput 5%
            weight_memory = 0.8
            weight_flops = 0.1
            weight_latency = 0.05
            weight_throughput = 0.05
            return (
                weight_memory * normalized_memory +
                weight_flops * normalized_flops +
                weight_latency * normalized_latency +
                weight_throughput * normalized_throughput
            )
        
        # Sort nodes based on the combined metric in descending order
        nodes.sort(key=combined_metric, reverse=True)
        
        # Calculate total combined metric
        total_combined_metric = sum(combined_metric(node) for node in nodes)
        
        # Initialize partitions
        partitions = []
        start = 0
        
        # Create partitions based on the combined metric
        for node in nodes:
            end = round(start + (combined_metric(node) / total_combined_metric), 5)
            partitions.append(Partition(node[0], start, end))
            start = end
        
        return partitions