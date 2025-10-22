import asyncio
import json
import base64
from typing import Optional, List
import numpy as np

from exo.networking.server import Server
from exo.orchestration import Node
from exo.inference.shard import Shard
from .radio_transport import RadioTransport
from exo.helpers import DEBUG


class RadioServer(Server):
  """Minimal radio server that speaks a tiny JSON message protocol over a RadioTransport.

  Messages:
    {"type":"health"}
    {"type":"prompt","shard":{...},"prompt":str,"request_id":str,"inference_state":{...}}
    {"type":"tensor","shard":{...},"tensor":{"data":base64,"shape":[],"dtype":str},"request_id":str,"inference_state":{...}}
    {"type":"topology","visited":[],"max_depth":int}
    {"type":"result","request_id":str,"result":[],"is_finished":bool}
    {"type":"opaque","request_id":str,"status":str}

  Replies mirror request with {"ok":true,...} and may include tensors.
  """
  def __init__(self, node: Node, transport: RadioTransport, addr: str):
    self.node = node
    self.transport = transport
    self.addr = addr
    self._task: Optional[asyncio.Task] = None

  async def start(self) -> None:
    await self.transport.open()
    self._task = asyncio.create_task(self._serve())
    if DEBUG >= 1: print(f"RadioServer listening at {self.addr}")

  async def stop(self) -> None:
    if self._task: self._task.cancel()
    await self.transport.close()

  async def _serve(self):
    while True:
      msg = await self.transport.recv()
      if msg is None: continue
      src, payload = msg
      try:
        data = json.loads(payload.decode("utf-8"))
      except Exception:
        continue
      t = data.get("type")
      if t == "health":
        await self.transport.send(src, json.dumps({"ok": True, "type": "health"}).encode())
      elif t == "prompt":
        shard = Shard(**data["shard"])
        prompt = data["prompt"]
        request_id = data.get("request_id")
        inf_state = data.get("inference_state")
        await self.node.process_prompt(shard, prompt, request_id=request_id, inference_state=inf_state)
        # No immediate tensor reply, results are pushed via result messages
        await self.transport.send(src, json.dumps({"ok": True, "type": "prompt", "request_id": request_id}).encode())
      elif t == "tensor":
        shard = Shard(**data["shard"])
        tinfo = data["tensor"]
        tensor = np.frombuffer(base64.b64decode(tinfo["data"]), dtype=np.dtype(tinfo["dtype"]))
        tensor = tensor.reshape(tinfo["shape"]) if tinfo.get("shape") else tensor
        request_id = data.get("request_id")
        inf_state = data.get("inference_state")
        result = await self.node.process_tensor(shard, tensor, request_id=request_id, inference_state=inf_state)
        reply = {"ok": True, "type": "tensor", "request_id": request_id}
        if isinstance(result, np.ndarray):
          reply["tensor"] = {
            "data": base64.b64encode(result.tobytes()).decode(),
            "shape": list(result.shape),
            "dtype": str(result.dtype)
          }
        await self.transport.send(src, json.dumps(reply).encode())
      elif t == "topology":
        visited = set(data.get("visited", []))
        max_depth = int(data.get("max_depth", 2))
        topo = await self.node.collect_topology(visited, max_depth)
        await self.transport.send(src, json.dumps({"ok": True, "type": "topology", "topology": topo.to_json()}).encode())
      elif t == "result":
        # Incoming result from other node
        await self.node.on_token.trigger_all(data["request_id"], data.get("result", []), data.get("is_finished", False))
      elif t == "opaque":
        await self.node.on_opaque_status.trigger_all(data.get("request_id"), data.get("status"))
      # else: ignore
