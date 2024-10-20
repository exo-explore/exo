from typing import List, Dict
from exo.topology.partitioning_strategy import PartitioningStrategy, Partition
from exo.topology.topology import Topology
from exo.inference.shard import Shard

class AdaptiveFlopsPartitioningStrategy(PartitioningStrategy):
    def __init__(self, ema_alpha: float = 0.2):
        self.node_performance: Dict[str, float] = {}
        self.total_flops: float = 0
        self.ema_alpha = ema_alpha

    def partition(self, topology: Topology) -> List[Partition]:
        nodes = list(topology.all_nodes())
        self.total_flops = sum(node[1].flops.fp16 for node in nodes)
        
        partitions = []
        start = 0
        total_performance = sum(self.node_performance.get(node[0], node[1].flops.fp16) for node in nodes)
        
        for node_id, capabilities in nodes:
            if node_id not in self.node_performance:
                # Use FLOPS as initial performance estimate
                performance = capabilities.flops.fp16
            else:
                performance = self.node_performance[node_id]
            
            end = start + (performance / total_performance)
            partitions.append(Partition(node_id, start, min(end, 1.0)))
            start = end

        return partitions

    def update_node_performance(self, node_id: str, processing_time: float, shard: Shard):
        shard_size = shard.end_layer - shard.start_layer + 1
        current_performance = shard_size / processing_time

        if node_id in self.node_performance:
            # EMA
            self.node_performance[node_id] = (self.ema_alpha * current_performance + 
                                              (1 - self.ema_alpha) * self.node_performance[node_id])
        else:
            # First Measurement
            self.node_performance[node_id] = current_performance