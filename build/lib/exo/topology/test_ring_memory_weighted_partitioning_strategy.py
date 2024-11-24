import unittest
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.partitioning_strategy import Partition


class TestRingMemoryWeightedPartitioningStrategy(unittest.TestCase):
  def test_partition(self):
    # triangle
    # node1 -> node2 -> node3 -> node1
    topology = Topology()
    topology.update_node(
      "node1",
      DeviceCapabilities(model="test1", chip="test1", memory=3000, flops=DeviceFlops(fp32=0, fp16=0, int8=0)),
    )
    topology.update_node(
      "node2",
      DeviceCapabilities(model="test2", chip="test2", memory=1000, flops=DeviceFlops(fp32=0, fp16=0, int8=0)),
    )
    topology.update_node(
      "node3",
      DeviceCapabilities(model="test3", chip="test3", memory=6000, flops=DeviceFlops(fp32=0, fp16=0, int8=0)),
    )
    topology.add_edge("node1", "node2")
    topology.add_edge("node2", "node3")
    topology.add_edge("node3", "node1")
    topology.add_edge("node1", "node3")

    strategy = RingMemoryWeightedPartitioningStrategy()
    partitions = strategy.partition(topology)

    self.assertEqual(len(partitions), 3)
    self.assertEqual(
      partitions,
      [
        Partition("node3", 0.0, 0.6),
        Partition("node1", 0.6, 0.9),
        Partition("node2", 0.9, 1.0),
      ],
    )

  def test_partition_rounding(self):
    # triangle
    # node1 -> node2 -> node3 -> node1
    topology = Topology()
    topology.update_node(
      "node1",
      DeviceCapabilities(
        model="MacBook Pro",
        chip="test1",
        memory=128*1024*1024*1024,
        flops=DeviceFlops(fp32=0, fp16=0, int8=0),
      ),
    )
    topology.update_node(
      "node2",
      DeviceCapabilities(
        model="Mac Studio",
        chip="test2",
        memory=192*1024*1024*1024,
        flops=DeviceFlops(fp32=0, fp16=0, int8=0),
      ),
    )
    topology.update_node(
      "node3",
      DeviceCapabilities(
        model="MacBook Pro",
        chip="test3",
        memory=128*1024*1024*1024,
        flops=DeviceFlops(fp32=0, fp16=0, int8=0),
      ),
    )

    strategy = RingMemoryWeightedPartitioningStrategy()
    partitions = strategy.partition(topology)

    self.assertEqual(len(partitions), 3)
    self.assertEqual(
      partitions,
      [
        Partition("node3", 0.0, 0.42857),
        Partition("node1", 0.6, 0.9),
        Partition("node2", 0.9, 1.0),
      ],
    )


if __name__ == "__main__":
  unittest.main()
