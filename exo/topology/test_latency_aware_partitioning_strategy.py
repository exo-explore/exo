import unittest
from exo.topology.latency_aware_partitioning_strategy import LatencyAwarePartitioningStrategy
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.partitioning_strategy import Partition


class TestLatencyAwarePartitioningStrategy(unittest.TestCase):
    def test_partition(self):
        # Create a topology with three nodes
        topology = Topology()
        topology.update_node(
            "node1",
            DeviceCapabilities(model="Node1", chip="Chip1", memory=4000, flops=DeviceFlops(fp32=0, fp16=0, int8=0)),
        )
        topology.update_node(
            "node2",
            DeviceCapabilities(model="Node2", chip="Chip2", memory=2000, flops=DeviceFlops(fp32=0, fp16=0, int8=0)),
        )
        topology.update_node(
            "node3",
            DeviceCapabilities(model="Node3", chip="Chip3", memory=8000, flops=DeviceFlops(fp32=0, fp16=0, int8=0)),
        )

        # Add edges with latencies
        # node1 <-> node2 with latency 10ms
        # node2 <-> node3 with latency 20ms
        # node3 <-> node1 with latency 30ms
        topology.add_edge("node1", "node2", latency=10)
        topology.add_edge("node2", "node3", latency=20)
        topology.add_edge("node3", "node1", latency=30)

        strategy = LatencyAwarePartitioningStrategy()
        partitions = strategy.partition(topology)

        # Expected scores calculation
        # Node1:
        # Average latency = (10 + 30) / 2 = 20ms
        # Latency factor = 1 / (20 + epsilon)
        # Score = 4000 * Latency factor = 4000 / 20 = 200

        # Node2:
        # Average latency = (10 + 20) / 2 = 15ms
        # Latency factor = 1 / (15 + epsilon)
        # Score = 2000 / 15 ≈ 133.333

        # Node3:
        # Average latency = (20 + 30) / 2 = 25ms
        # Latency factor = 1 / (25 + epsilon)
        # Score = 8000 / 25 = 320

        # Total score = 200 + 133.333 + 320 = 653.333

        # Compute proportions
        total_score = 200 + 133.333 + 320
        node1_proportion = 200 / total_score
        node2_proportion = 133.333 / total_score
        node3_proportion = 320 / total_score

        # Expected partitions
        expected_partitions = [
            Partition("node1", 0.0, node1_proportion),
            Partition("node2", node1_proportion, node1_proportion + node2_proportion),
            Partition("node3", node1_proportion + node2_proportion, 1.0),
        ]

        # Assert that the number of partitions is correct
        self.assertEqual(len(partitions), 3)

        # Helper function to compare partitions within a tolerance
        def partitions_almost_equal(p1, p2, tol=1e-4):
            return (
                p1.node_id == p2.node_id and
                abs(p1.start - p2.start) <= tol and
                abs(p1.end - p2.end) <= tol
            )

        # Compare each expected partition with the actual partition
        for expected, actual in zip(expected_partitions, partitions):
            self.assertTrue(
                partitions_almost_equal(expected, actual),
                f"Expected {expected}, got {actual}"
            )


if __name__ == "__main__":
    unittest.main()
