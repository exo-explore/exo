from typing import Dict
from dataclasses import dataclass

import json

from exo.topology.device_capabilities import DeviceCapabilities


@dataclass
class PeerConfig:
  address: str
  port: int
  device_capabilities: DeviceCapabilities


@dataclass
class NetworkTopology:
  """Configuration of the network. A collection outlining all nodes in the network, including the node this is running from."""

  peers: Dict[str, PeerConfig]
  """
  node_id to PeerConfig. The node_id is used to identify the peer in the discovery process. The node that this is running from should be included in this dict.
  """

  @classmethod
  def from_path(cls, path: str) -> "NetworkTopology":
    try:
      with open(path, "r") as f:
        config = json.load(f)
    except FileNotFoundError:
      raise FileNotFoundError(f"Config file not found at {path}")
    except json.JSONDecodeError as e:
      raise json.JSONDecodeError(f"Error decoding JSON data from {path}: {e}", e.doc, e.pos)

    try:
      peers = {}
      for node_id, peer_data in config["peers"].items():
        device_capabilities = DeviceCapabilities(**peer_data["device_capabilities"])
        peer_config = PeerConfig(address=peer_data["address"], port=peer_data["port"], device_capabilities=device_capabilities)
        peers[node_id] = peer_config

      networking_config = cls(peers=peers)
    except KeyError as e:
      raise KeyError(f"Missing required key in config file: {e}")
    except TypeError as e:
      raise TypeError(f"Error parsing networking config from {path}: {e}")

    return networking_config
