import unittest
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops

class TestTopology(unittest.TestCase):

    def test_update_node(self):
        topology = Topology()
        node_id = "node1"
        capabilities = DeviceCapabilities(model="test1", chip="test1", memory=3000, flops=DeviceFlops(fp32=1000, fp16=0, int8=0))
        topology.update_node(node_id, capabilities)
        self.assertEqual(topology.get_node(node_id), capabilities)

    def test_add_edge(self):
        topology = Topology()
        node1_id = "node1"
        node2_id = "node2"
        latency = 10.0
        throughput = 100.0
        topology.add_edge(node1_id, node2_id, latency, throughput)
        neighbors = topology.get_neighbors(node1_id)
        self.assertEqual(neighbors[node2_id], (latency, throughput))
        neighbors = topology.get_neighbors(node2_id)
        self.assertEqual(neighbors[node1_id], (latency, throughput))

    def test_get_neighbors(self):
        topology = Topology()
        node1_id = "node1"
        node2_id = "node2"
        latency = 10.0
        throughput = 100.0
        topology.add_edge(node1_id, node2_id, latency, throughput)
        neighbors = topology.get_neighbors(node1_id)
        self.assertEqual(neighbors, {node2_id: (latency, throughput)})

    def test_all_edges(self):
        topology = Topology()
        node1_id = "node1"
        node2_id = "node2"
        node3_id = "node3"
        latency1 = 10.0
        throughput1 = 100.0
        latency2 = 20.0
        throughput2 = 200.0
        topology.add_edge(node1_id, node2_id, latency1, throughput1)
        topology.add_edge(node2_id, node3_id, latency2, throughput2)
        edges = topology.all_edges()
        expected_edges = [
            (node1_id, node2_id, latency1, throughput1),
            (node2_id, node1_id, latency1, throughput1),
            (node2_id, node3_id, latency2, throughput2),
            (node3_id, node2_id, latency2, throughput2)
        ]
        self.assertEqual(sorted(edges), sorted(expected_edges))

    def test_merge(self):
        topology1 = Topology()
        topology2 = Topology()
        node1_id = "node1"
        node2_id = "node2"
        node3_id = "node3"
        latency1 = 10.0
        throughput1 = 100.0
        latency2 = 20.0
        throughput2 = 200.0
        topology1.update_node(node1_id, DeviceCapabilities(model="test1", chip="test1", memory=3000, flops=DeviceFlops(fp32=1000, fp16=0, int8=0)))
        topology2.update_node(node2_id, DeviceCapabilities(model="test2", chip="test2", memory=2000, flops=DeviceFlops(fp32=500, fp16=0, int8=0)))
        topology2.update_node(node3_id, DeviceCapabilities(model="test3", chip="test3", memory=1000, flops=DeviceFlops(fp32=250, fp16=0, int8=0)))
        topology2.add_edge(node2_id, node3_id, latency2, throughput2)
        topology1.merge(topology2)
        self.assertEqual(topology1.get_node(node1_id).memory, 3000)
        self.assertEqual(topology1.get_node(node2_id).memory, 2000)
        self.assertEqual(topology1.get_node(node3_id).memory, 1000)
        neighbors = topology1.get_neighbors(node2_id)
        self.assertEqual(neighbors[node3_id], (latency2, throughput2))


if __name__ == "__main__":
    unittest.main()