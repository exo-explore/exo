from .partitioning_strategy import PartitioningStrategy
from inference.shard import Shard
from .topology import Topology

class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy):
    def next_shard(self, current_shard: Shard, topology: Topology, node_stats: dict) -> Shard:
        # Get all nodes from the topology and include the current node
        nodes = list(topology.all_nodes())
        nodes.append((self.id, None, node_stats))

        # Sort nodes by their IDs
        nodes.sort(key=lambda x: x[0])

        # Calculate the total memory of all nodes
        total_memory = sum(node[2]['memory'] for node in nodes)

        # Calculate the number of layers to assign to each node proportional to its memory
        layers_per_node = {node[0]: (node[2]['memory'] / total_memory) * current_shard.n_layers for node in nodes}

        # Find the successor node
        node_ids = [node[0] for node in nodes]
        current_index = node_ids.index(self.id)
        successor_index = (current_index + 1) % len(node_ids)
        successor_id = node_ids[successor_index]

        # Return the Shard calculated for the successor
        return Shard(successor_id, layers_per_node[successor_id])
