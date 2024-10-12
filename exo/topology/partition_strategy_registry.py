
from exo.topology.ring_memory_and_flops_weighted_partitioning_strategy import RingMemoryAndFlopsWeightedPartitioningStrategy
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy

partition_strategy_registry = {
  "ring-memory-weighted": RingMemoryWeightedPartitioningStrategy,
  "ring-memory-and-flops-weighted": RingMemoryAndFlopsWeightedPartitioningStrategy,
}
