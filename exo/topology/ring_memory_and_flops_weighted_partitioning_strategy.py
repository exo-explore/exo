from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition


class RingMemoryAndFlopsWeightedPartitioningStrategy(PartitioningStrategy):
  def __init__(self, memory_weight: float = 0.5, flops_weight: float = 0.5):
    self.memory_weight = memory_weight
    self.flops_weight = flops_weight

  def partition(self, topology: Topology) -> List[Partition]:
    nodes = list(topology.all_nodes())
    max_flops = max(node[1].flops for node in nodes)
    max_memory = max(node[1].memory for node in nodes)
    def compute_score(node):
      return self.flops_weight * node[1].flops/max_flops + self.memory_weight * node[1].memory/max_memory
    nodes.sort(key=compute_score, reverse=True)
    total_score = sum(compute_score(node) for node in nodes)
    partitions = []
    start = 0
    for node in nodes:
      end = round(start + compute_score(node)/total_score, 5)
      partitions.append(Partition(node[0], start, end))
      start = end
    return partitions
