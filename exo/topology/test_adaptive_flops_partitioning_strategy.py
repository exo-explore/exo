import unittest
from exo.topology.adaptive_flops_partitioning_strategy import AdaptiveFlopsPartitioningStrategy
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.inference.shard import Shard

class TestAdaptiveFlopsPartitioningStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = AdaptiveFlopsPartitioningStrategy(ema_alpha=0.5)
        self.topology = Topology()

    def test_initial_partition_based_on_flops(self):
        self.topology.update_node(
            "node1",
            DeviceCapabilities(model="test1", chip="test1", memory=3000, flops=DeviceFlops(fp32=0, fp16=100, int8=0))
        )
        self.topology.update_node(
            "node2",
            DeviceCapabilities(model="test2", chip="test2", memory=1000, flops=DeviceFlops(fp32=0, fp16=200, int8=0))
        )
        
        partitions = self.strategy.partition(self.topology)
        
        self.assertEqual(len(partitions), 2)
        self.assertAlmostEqual(partitions[0].start, 0.0)
        self.assertAlmostEqual(partitions[0].end, 1/3)
        self.assertAlmostEqual(partitions[1].start, 1/3)
        self.assertAlmostEqual(partitions[1].end, 1.0)

    def test_partition_after_performance_update(self):
        self.topology.update_node(
            "node1",
            DeviceCapabilities(model="test1", chip="test1", memory=3000, flops=DeviceFlops(fp32=0, fp16=100, int8=0))
        )
        self.topology.update_node(
            "node2",
            DeviceCapabilities(model="test2", chip="test2", memory=1000, flops=DeviceFlops(fp32=0, fp16=100, int8=0))
        )
        
        # Initial partition
        initial_partitions = self.strategy.partition(self.topology)
        
        # Update performance for node1 (significantly better performance)
        self.strategy.update_node_performance("node1", 0.1, Shard(model_id="test", start_layer=0, end_layer=49, n_layers=100))
        
        # New partition after update
        updated_partitions = self.strategy.partition(self.topology)
        
        self.assertNotEqual(initial_partitions[0].end, updated_partitions[0].end)
        self.assertGreater(updated_partitions[0].end, 0.5)  # node1 should now have a larger partition

    def test_ema_smoothing(self):
        self.topology.update_node(
            "node1",
            DeviceCapabilities(model="test1", chip="test1", memory=3000, flops=DeviceFlops(fp32=0, fp16=100, int8=0))
        )
        
        # First update
        self.strategy.update_node_performance("node1", 1.0, Shard(model_id="test", start_layer=0, end_layer=49, n_layers=100))
        first_performance = self.strategy.node_performance["node1"]
        
        # Second update with worse performance
        self.strategy.update_node_performance("node1", 2.0, Shard(model_id="test", start_layer=0, end_layer=49, n_layers=100))
        second_performance = self.strategy.node_performance["node1"]
        
        # Check that performance decreased but not to half due to EMA
        self.assertLess(second_performance, first_performance)
        self.assertGreater(second_performance, first_performance / 2)

    def test_adding_new_node(self):
        self.topology.update_node(
            "node1",
            DeviceCapabilities(model="test1", chip="test1", memory=3000, flops=DeviceFlops(fp32=0, fp16=100, int8=0))
        )
        initial_partitions = self.strategy.partition(self.topology)
        
        self.topology.update_node(
            "node2",
            DeviceCapabilities(model="test2", chip="test2", memory=1000, flops=DeviceFlops(fp32=0, fp16=100, int8=0))
        )
        updated_partitions = self.strategy.partition(self.topology)
        
        self.assertEqual(len(initial_partitions), 1)
        self.assertEqual(len(updated_partitions), 2)
        self.assertAlmostEqual(updated_partitions[0].end, 0.5)
        self.assertAlmostEqual(updated_partitions[1].start, 0.5)

    def test_node_removal(self):
        self.topology.update_node(
            "node1",
            DeviceCapabilities(model="test1", chip="test1", memory=3000, flops=DeviceFlops(fp32=0, fp16=100, int8=0))
        )
        self.topology.update_node(
            "node2",
            DeviceCapabilities(model="test2", chip="test2", memory=1000, flops=DeviceFlops(fp32=0, fp16=100, int8=0))
        )
        initial_partitions = self.strategy.partition(self.topology)
        
        # Create a new topology with only one node to simulate removal
        new_topology = Topology()
        new_topology.update_node(
            "node1",
            DeviceCapabilities(model="test1", chip="test1", memory=3000, flops=DeviceFlops(fp32=0, fp16=100, int8=0))
        )
        updated_partitions = self.strategy.partition(new_topology)
        
        self.assertEqual(len(initial_partitions), 2)
        self.assertEqual(len(updated_partitions), 1)
        self.assertAlmostEqual(updated_partitions[0].end, 1.0)

if __name__ == '__main__':
    unittest.main()