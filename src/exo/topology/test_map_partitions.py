import unittest
from typing import List
from exo.topology.partitioning_strategy import Partition, map_partitions_to_shards
from exo.inference.shard import Shard


class TestRingMemoryWeightedPartitioningStrategy(unittest.TestCase):
  def test_map_partitions_to_shards(self):
    partitions = [
      Partition("node1", 0.0, 0.42857),
      Partition("node2", 0.42857, 0.71428),
      Partition("node3", 0.71428, 0.99999),
    ]
    shards = map_partitions_to_shards(partitions, 32, "model")
    self.assertEqual(
      shards,
      [
        Shard("model", 0, 12, 32),
        Shard("model", 13, 21, 32),
        Shard("model", 22, 31, 32),
      ],
    )

    partitions = [
      Partition("node1", 0.0, 0.1),
      Partition("node2", 0.1, 0.2),
      Partition("node3", 0.2, 1.0),
    ]
    shards = map_partitions_to_shards(partitions, 32, "model")
    self.assertEqual(
      shards,
      [
        Shard("model", 0, 2, 32),
        Shard("model", 3, 5, 32),
        Shard("model", 6, 31, 32),
      ],
    )

    partitions = [
      Partition("node1", 0.0, 1.0),
    ]
    shards = map_partitions_to_shards(partitions, 32, "model")
    self.assertEqual(
      shards,
      [
        Shard("model", 0, 31, 32),
      ],
    )

    partitions = []
    shards = map_partitions_to_shards(partitions, 32, "model")
    self.assertEqual(shards, [])

  def test_broken_map_partitions_to_shards(self):
    # this was an old broken implementation that sometimes had rounding errors!
    def _broken_map_partitions_to_shards(partitions: List[Partition], num_layers, model_id: str):
      shards = []
      for i, partition in enumerate(partitions):
        start_layer = int(partition.start*num_layers)
        end_layer = int(partition.end*num_layers) - 1
        shards.append(Shard(model_id, start_layer, end_layer, num_layers))
      return shards

    partitions = [
      Partition("node1", 0.0, 0.42857),
      Partition("node2", 0.42857, 0.71428),
      Partition("node3", 0.71428, 0.99999),
    ]
    shards = _broken_map_partitions_to_shards(partitions, 32, "model")
    self.assertEqual(
      shards,
      [
        Shard("model", 0, 12, 32),
        Shard("model", 13, 21, 32),
        Shard("model", 22, 30, 32),
      ],
    )


if __name__ == "__main__":
  unittest.main()
