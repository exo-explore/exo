import asyncio
import unittest
from exo.viz.topology_viz import TopologyViz
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.partitioning_strategy import Partition
from exo.helpers import AsyncCallbackSystem

class TestNodeViz(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.topology = Topology()
        self.topology.update_node("node1", DeviceCapabilities(model="ModelA", chip="ChipA", memory=8*1024, flops=DeviceFlops(fp32=1.0,fp16=2.0,int8=4.0)))
        self.topology.update_node("node2", DeviceCapabilities(model="ModelB", chip="ChipB", memory=16*1024, flops=DeviceFlops(fp32=2.0,fp16=4.0,int8=8.0)))
        self.topology.update_node("node3", DeviceCapabilities(model="ModelC", chip="ChipC", memory=32*1024, flops=DeviceFlops(fp32=4.0,fp16=8.0,int8=16.0)))
        self.topology.update_node("node4", DeviceCapabilities(model="ModelD", chip="ChipD", memory=64*1024, flops=DeviceFlops(fp32=8.0,fp16=16.0,int8=32.0)))

        self.top_viz = TopologyViz()
        await asyncio.sleep(2)  # Simulate running for a short time

    async def test_layout_generation(self):
        self.top_viz._generate_layout()
        self.top_viz.refresh()
        import time
        time.sleep(2)
        self.top_viz.update_visualization(self.topology, [
            Partition("node1", 0, 0.2),
            Partition("node4", 0.2, 0.4),
            Partition("node2", 0.4, 0.8),
            Partition("node3", 0.8, 1),
        ])
        time.sleep(2)
        self.topology.active_node_id = "node3"
        self.top_viz.update_visualization(self.topology, [
            Partition("node1", 0, 0.3),
            Partition("node2", 0.3, 0.7),
            Partition("node4", 0.7, 0.9),
            Partition("node3", 0.9, 1),
        ])
        time.sleep(2)


if __name__ == "__main__":
    unittest.main()
