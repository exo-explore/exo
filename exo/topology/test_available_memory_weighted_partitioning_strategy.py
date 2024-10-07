# In your tests directory, create a test file, e.g., test_available_memory_weighted_partitioning_strategy.py

import unittest
from exo.topology.available_memory_weighted_partitioning_strategy import AvailableMemoryWeightedPartitioningStrategy
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.topology import Topology

class TestAvailableMemoryWeightedPartitioningStrategy(unittest.TestCase):
    def test_single_node(self):
        # Setup topology with a single node
        topology = Topology()
        topology.update_node('node1', DeviceCapabilities(
            model='TestModel',
            chip='TestChip',
            memory=16000,
            memory_available=8000,
            flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0)
        ))
        strategy = AvailableMemoryWeightedPartitioningStrategy()
        partitions = strategy.partition(topology)
        self.assertEqual(len(partitions), 1)
        self.assertEqual(partitions[0].start, 0.0)
        self.assertEqual(partitions[0].end, 1.0)

    def test_zero_available_memory(self):
        # Setup topology with nodes having zero available memory
        topology = Topology()
        topology.update_node('node1', DeviceCapabilities(
            model='TestModel',
            chip='TestChip',
            memory=16000,
            memory_available=0,
            flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0)
        ))
        strategy = AvailableMemoryWeightedPartitioningStrategy()
        partitions = strategy.partition(topology)
        self.assertEqual(len(partitions), 1)
        self.assertEqual(partitions[0].start, 0.0)
        self.assertEqual(partitions[0].end, 1.0)

    def test_varying_memory_sizes(self):
        # Setup topology with nodes having varying available memory
        topology = Topology()
        topology.update_node('node1', DeviceCapabilities(
            model='Model1',
            chip='Chip1',
            memory=16000,
            memory_available=8000,
            flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0)
        ))
        topology.update_node('node2', DeviceCapabilities(
            model='Model2',
            chip='Chip2',
            memory=32000,
            memory_available=16000,
            flops=DeviceFlops(fp32=2.0, fp16=4.0, int8=8.0)
        ))
        strategy = AvailableMemoryWeightedPartitioningStrategy()
        partitions = strategy.partition(topology)
        self.assertEqual(len(partitions), 2)
        self.assertAlmostEqual(partitions[0].end, 0.3333, places=4)
        self.assertEqual(partitions[1].end, 1.0)

if __name__ == '__main__':
    unittest.main()
