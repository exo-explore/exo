import asyncio
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class RadioTransport(ABC):
  """Abstract radio transport for sending small framed messages over a shared medium.
  Implementations may use real radios (LoRa, AX.25/KISS) or simulated loopback.
  """
  @abstractmethod
  async def open(self) -> None:
    pass

  @abstractmethod
  async def close(self) -> None:
    pass

  @abstractmethod
  async def send(self, dest: str, payload: bytes) -> None:
    pass

  @abstractmethod
  async def recv(self, timeout: Optional[float] = None) -> Optional[tuple[str, bytes]]:
    """Receive next frame. Returns (source, payload) or None on timeout."""
    pass


class LoopbackHub:
  """In-memory hub to simulate a shared radio channel among addresses."""
  def __init__(self):
    self.queues: Dict[str, asyncio.Queue[tuple[str, bytes]]] = {}

  def register(self, addr: str):
    if addr not in self.queues:
      self.queues[addr] = asyncio.Queue()

  async def broadcast(self, src: str, dest: str, payload: bytes):
    # Unicast if dest known, otherwise drop (no true broadcast to keep simple)
    if dest in self.queues:
      await self.queues[dest].put((src, payload))

  async def recv(self, addr: str, timeout: Optional[float] = None) -> Optional[tuple[str, bytes]]:
    q = self.queues.get(addr)
    if not q:
      return None
    try:
      if timeout is None:
        return await q.get()
      else:
        return await asyncio.wait_for(q.get(), timeout)
    except asyncio.TimeoutError:
      return None


class LoopbackRadioTransport(RadioTransport):
  def __init__(self, hub: LoopbackHub, addr: str):
    self.hub = hub
    self.addr = addr
    self.hub.register(addr)
    self._open = False

  async def open(self) -> None:
    self._open = True

  async def close(self) -> None:
    self._open = False

  async def send(self, dest: str, payload: bytes) -> None:
    if not self._open:
      raise RuntimeError("transport not open")
    await self.hub.broadcast(self.addr, dest, payload)

  async def recv(self, timeout: Optional[float] = None) -> Optional[tuple[str, bytes]]:
    if not self._open:
      raise RuntimeError("transport not open")
    return await self.hub.recv(self.addr, timeout)
