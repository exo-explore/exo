import asyncio
import json
import base64
from typing import Optional, List
import numpy as np

from exo.networking.peer_handle import PeerHandle
from exo.topology.topology import Topology
from exo.inference.shard import Shard
from exo.topology.device_capabilities import DeviceCapabilities
from .radio_transport import RadioTransport
from exo.helpers import DEBUG


class RadioPeerHandle(PeerHandle):
  def __init__(self, _id: str, addr: str, desc: str, device_capabilities: DeviceCapabilities, transport: RadioTransport):
    self._id = _id
    self._addr = addr
    self._desc = desc
    self._device_capabilities = device_capabilities
    self.transport = transport
    self._connected = False

  def id(self) -> str:
    return self._id

  def addr(self) -> str:
    return self._addr

  def description(self) -> str:
    return self._desc

  def device_capabilities(self) -> DeviceCapabilities:
    return self._device_capabilities

  async def connect(self) -> None:
    await self.transport.open()
    self._connected = True

  async def is_connected(self) -> bool:
    return self._connected

  async def disconnect(self) -> None:
    await self.transport.close()
    self._connected = False

  async def health_check(self) -> bool:
    try:
      await self.transport.send(self._addr, json.dumps({"type": "health"}).encode())
      # Best-effort: we don't wait for a reply on radio, assume OK when send succeeds
      return True
    except Exception:
      return False

  async def send_prompt(self, shard: Shard, prompt: str, inference_state: dict | None = None, request_id: str | None = None):
    msg = {
      "type": "prompt",
      "shard": shard.to_dict(),
      "prompt": prompt,
      "request_id": request_id,
      "inference_state": inference_state,
    }
    await self.transport.send(self._addr, json.dumps(msg).encode())

  async def send_tensor(self, shard: Shard, tensor: np.ndarray, inference_state: dict | None = None, request_id: str | None = None):
    msg = {
      "type": "tensor",
      "shard": shard.to_dict(),
      "tensor": {
        "data": base64.b64encode(tensor.tobytes()).decode(),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype)
      },
      "request_id": request_id,
      "inference_state": inference_state,
    }
    await self.transport.send(self._addr, json.dumps(msg).encode())
    # In this minimal implementation, we do not await a response
    return None

  async def send_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    await self.transport.send(self._addr, json.dumps({
      "type": "result",
      "request_id": request_id,
      "result": result,
      "is_finished": is_finished,
    }).encode())

  async def collect_topology(self, visited: set[str], max_depth: int) -> Topology:
    await self.transport.send(self._addr, json.dumps({
      "type": "topology",
      "visited": list(visited),
      "max_depth": max_depth,
    }).encode())
    # Minimal: return empty topology (radio is best-effort here)
    return Topology()
