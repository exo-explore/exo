import unittest

from exo.networking.manual.network_topology_config import NetworkTopology

root_path = "./exo/networking/manual/test_data/"


class TestNetworkTopologyConfig(unittest.TestCase):
  def test_from_path_invalid_path(self):
    with self.assertRaises(FileNotFoundError) as e:
      NetworkTopology.from_path("invalid_path")
    self.assertEqual(str(e.exception), "Config file not found at invalid_path")

  def test_from_path_invalid_json(self):
    with self.assertRaises(ValueError) as e:
      NetworkTopology.from_path(root_path + "invalid_json.json")
    self.assertIn("Error validating network topology config from", str(e.exception))
    self.assertIn("1 validation error for NetworkTopology\n  Invalid JSON: EOF while parsing a value at line 1 column 0", str(e.exception))

  def test_from_path_invalid_config(self):
    with self.assertRaises(ValueError) as e:
      NetworkTopology.from_path(root_path + "invalid_config.json")
    self.assertIn("Error validating network topology config from", str(e.exception))
    self.assertIn("port\n  Field required", str(e.exception))

  def test_from_path_valid(self):
    config = NetworkTopology.from_path(root_path + "test_config.json")

    self.assertEqual(config.peers["node1"].port, 50051)
    self.assertEqual(config.peers["node1"].device_capabilities.model, "Unknown Model")
    self.assertEqual(config.peers["node1"].address, "localhost")
    self.assertEqual(config.peers["node1"].device_capabilities.chip, "Unknown Chip")
    self.assertEqual(config.peers["node1"].device_capabilities.memory, 0)
    self.assertEqual(config.peers["node1"].device_capabilities.flops.fp32, 0)
    self.assertEqual(config.peers["node1"].device_capabilities.flops.fp16, 0)
    self.assertEqual(config.peers["node1"].device_capabilities.flops.int8, 0)

    self.assertEqual(config.peers["node2"].port, 50052)
    self.assertEqual(config.peers["node2"].device_capabilities.model, "Unknown Model")
    self.assertEqual(config.peers["node2"].address, "localhost")
    self.assertEqual(config.peers["node2"].device_capabilities.chip, "Unknown Chip")
    self.assertEqual(config.peers["node2"].device_capabilities.memory, 0)
    self.assertEqual(config.peers["node2"].device_capabilities.flops.fp32, 0)
    self.assertEqual(config.peers["node2"].device_capabilities.flops.fp16, 0)
    self.assertEqual(config.peers["node2"].device_capabilities.flops.int8, 0)


if __name__ == "__main__":
  unittest.main()
