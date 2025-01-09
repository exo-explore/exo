import json
import asyncio
import aiohttp
import re
from typing import Dict, Any, Tuple, List, Optional
from exo.helpers import DEBUG_DISCOVERY
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from datetime import datetime, timezone


class Device:
  def __init__(self, device_id: str, name: str, addresses: List[str], last_seen: Optional[datetime] = None):
    self.device_id = device_id
    self.name = name
    self.addresses = addresses
    self.last_seen = last_seen

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> 'Device':
    return cls(device_id=data.get('id', ''), name=data.get('name', ''), addresses=data.get('addresses', []), last_seen=cls.parse_datetime(data.get('lastSeen')))

  @staticmethod
  def parse_datetime(date_string: Optional[str]) -> Optional[datetime]:
    if not date_string:
      return None
    return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


async def get_device_id() -> str:
  try:
    process = await asyncio.create_subprocess_exec('tailscale', 'status', '--json', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
      raise Exception(f"Command failed with exit code {process.returncode}: {stderr.decode().strip()}.")
    if DEBUG_DISCOVERY >= 4: print(f"tailscale status: {stdout.decode()}")
    data = json.loads(stdout.decode())
    return data['Self']['ID']
  except Exception as e:
    raise Exception(f"{str(e)} Do you have the tailscale cli installed? See: https://tailscale.com/kb/1080/cli")


async def update_device_attributes(device_id: str, api_key: str, node_id: str, node_port: int, device_capabilities: DeviceCapabilities):
  async with aiohttp.ClientSession() as session:
    base_url = f"https://api.tailscale.com/api/v2/device/{device_id}/attributes"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

    attributes = {
      "custom:exo_node_id": node_id.replace('-', '_'), "custom:exo_node_port": node_port, "custom:exo_device_capability_chip": sanitize_attribute(device_capabilities.chip),
      "custom:exo_device_capability_model": sanitize_attribute(device_capabilities.model), "custom:exo_device_capability_memory": str(device_capabilities.memory),
      "custom:exo_device_capability_flops_fp16": str(device_capabilities.flops.fp16), "custom:exo_device_capability_flops_fp32": str(device_capabilities.flops.fp32),
      "custom:exo_device_capability_flops_int8": str(device_capabilities.flops.int8)
    }

    for attr_name, attr_value in attributes.items():
      url = f"{base_url}/{attr_name}"
      data = {"value": str(attr_value).replace(' ', '_')}  # Ensure all values are strings for JSON
      async with session.post(url, headers=headers, json=data) as response:
        if response.status == 200:
          if DEBUG_DISCOVERY >= 1: print(f"Updated device posture attribute {attr_name} for device {device_id}")
        else:
          print(f"Failed to update device posture attribute {attr_name}: {response.status} {await response.text()}")


async def get_device_attributes(device_id: str, api_key: str) -> Tuple[str, int, DeviceCapabilities]:
  async with aiohttp.ClientSession() as session:
    url = f"https://api.tailscale.com/api/v2/device/{device_id}/attributes"
    headers = {'Authorization': f'Bearer {api_key}'}
    async with session.get(url, headers=headers) as response:
      if response.status == 200:
        data = await response.json()
        attributes = data.get("attributes", {})
        node_id = attributes.get("custom:exo_node_id", "").replace('_', '-')
        node_port = int(attributes.get("custom:exo_node_port", 0))
        device_capabilities = DeviceCapabilities(
          model=attributes.get("custom:exo_device_capability_model", "").replace('_', ' '),
          chip=attributes.get("custom:exo_device_capability_chip", "").replace('_', ' '),
          memory=int(attributes.get("custom:exo_device_capability_memory", 0)),
          flops=DeviceFlops(
            fp16=float(attributes.get("custom:exo_device_capability_flops_fp16", 0)),
            fp32=float(attributes.get("custom:exo_device_capability_flops_fp32", 0)),
            int8=float(attributes.get("custom:exo_device_capability_flops_int8", 0))
          )
        )
        return node_id, node_port, device_capabilities
      else:
        print(f"Failed to fetch posture attributes for {device_id}: {response.status}")
        return "", 0, DeviceCapabilities(model="", chip="", memory=0, flops=DeviceFlops(fp16=0, fp32=0, int8=0))


def parse_device_attributes(data: Dict[str, str]) -> Dict[str, Any]:
  result = {}
  prefix = "custom:exo_"
  for key, value in data.items():
    if key.startswith(prefix):
      attr_name = key.replace(prefix, "")
      if attr_name in ["node_id", "node_port", "device_capability_chip", "device_capability_model"]:
        result[attr_name] = value.replace('_', ' ')
      elif attr_name in ["device_capability_memory", "device_capability_flops_fp16", "device_capability_flops_fp32", "device_capability_flops_int8"]:
        result[attr_name] = float(value)
  return result


def sanitize_attribute(value: str) -> str:
  # Replace invalid characters with underscores
  sanitized_value = re.sub(r'[^a-zA-Z0-9_.]', '_', value)
  # Truncate to 50 characters
  return sanitized_value[:50]


async def get_tailscale_devices(api_key: str, tailnet: str) -> Dict[str, Device]:
  async with aiohttp.ClientSession() as session:
    url = f"https://api.tailscale.com/api/v2/tailnet/{tailnet}/devices"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with session.get(url, headers=headers) as response:
      response.raise_for_status()
      data = await response.json()

      devices = {}
      for device_data in data.get("devices", []):
        print("Device data: ", device_data)
        device = Device.from_dict(device_data)
        devices[device.name] = device

      return devices
