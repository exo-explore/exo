from typing import Dict
from pydantic import BaseModel, ValidationError

from exo.topology.device_capabilities import DeviceCapabilities


class PeerConfig(BaseModel):
  address: str
  port: int
  device_capabilities: DeviceCapabilities


class NetworkTopology(BaseModel):
  """Configuration of the network. A collection outlining all nodes in the network, including the node this is running from."""

  peers: Dict[str, PeerConfig]
  """
  node_id to PeerConfig. The node_id is used to identify the peer in the discovery process. The node that this is running from should be included in this dict.
  
  Example configuration file format:
  ```json
  {
    "peers": {
      "node1": {
        "address": "192.168.1.10",
        "port": 50051,
        "device_capabilities": {
          "model": "Mac Studio",
          "chip": "Apple M3 Ultra",
          "memory": 524288,
          "flops": {
            "fp32": 108.52,
            "fp16": 54.26,
            "int8": 217.04
          }
        }
      },
      "node2": {
        "address": "192.168.1.20",
        "port": 50051,
        "device_capabilities": {
          "model": "Desktop PC",
          "chip": "NVIDIA GEFORCE GTX 1080 TI",
          "memory": 11264,
          "flops": {
            "fp32": 11.34,
            "fp16": 0.177,
            "int8": 45.36
          }
        }
      }
    }
  }
  ```
  
  When running with manual discovery, make sure your node_id matches one of the keys in this peers dictionary.
  """
  @classmethod
  def from_path(cls, path: str) -> "NetworkTopology":
    try:
      with open(path, "r") as f:
        config_data = f.read()
    except FileNotFoundError as e:
      raise FileNotFoundError(f"Config file not found at {path}") from e

    try:
      return cls.model_validate_json(config_data)
    except ValidationError as e:
      raise ValueError(f"Error validating network topology config from {path}: {e}") from e
