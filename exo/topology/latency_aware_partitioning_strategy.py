from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition

class LatencyAwarePartitioningStrategy(PartitioningStrategy):
    def partition(self, topology: Topology) -> List[Partition]:
        nodes = list(topology.all_nodes())
        node_scores = []

        for node_id, capabilities in nodes:
            neighbors = topology.get_neighbors(node_id)
            # Compute average latency to neighbors; handle division by zero
            if neighbors:
                avg_latency = sum(neighbors.values()) / len(neighbors)
            else:
                avg_latency = float('inf')  # Assign high latency if no neighbors

            # Compute score based on memory and latency
            # Higher memory and lower latency should result in a higher score
            # Avoid division by zero by adding a small epsilon
            epsilon = 1e-6
            latency_factor = 1 / (avg_latency + epsilon)
            score = capabilities.memory * latency_factor
            node_scores.append((node_id, score))

        # Normalize scores to sum to 1
        total_score = sum(score for node_id, score in node_scores)
        partitions = []
        start = 0.0
        for node_id, score in node_scores:
            proportion = score / total_score if total_score > 0 else 0
            end = start + proportion
            partitions.append(Partition(node_id, start, end))
            start = end

        return partitions
