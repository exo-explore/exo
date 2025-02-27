"""Helpers for interacting with Headscale API."""
import json
import asyncio
import aiohttp
import re
from typing import Dict, Any, Tuple, List, Optional
from exo.helpers import DEBUG_DISCOVERY
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops


class Device:
    """Represents a device in the Headscale network."""
    
    def __init__(self, device_id: str, name: str, addresses: List[str], online: bool = False):
        self.device_id = device_id
        self.name = name
        self.addresses = addresses
        self.online = online

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Device':
        """Create a Device instance from a dictionary."""
        node_data = data.get('node', data)  # Handle nested 'node' structure
        return cls(
            device_id=str(node_data.get('id', '')),
            name=node_data.get('name', ''),
            addresses=node_data.get('ipAddresses', []),
            online=node_data.get('online', False),
        )


async def get_device_id() -> str:
    """Get the current device ID from tailscale cli."""
    try:
        process = await asyncio.create_subprocess_exec(
            'tailscale', 'status', '--json', 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"Command failed with exit code {process.returncode}: {stderr.decode().strip()}.")
        if DEBUG_DISCOVERY >= 4: print(f"tailscale status: {stdout.decode()}")
        data = json.loads(stdout.decode())
        return data['Self']['ID']
    except Exception as e:
        raise Exception(f"{str(e)} Do you have the tailscale cli installed? See: https://tailscale.com/kb/1080/cli")


def sanitize_attribute(value: str) -> str:
    """Sanitize an attribute value for use in tags."""
    # Replace invalid characters with underscores
    sanitized_value = re.sub(r'[^a-zA-Z0-9_.]', '_', value)
    # Truncate to 50 characters
    return sanitized_value[:50]


async def update_device_attributes(api_base_url: str, device_id: str, api_key: str, node_id: str, 
                                  node_port: int, device_capabilities: DeviceCapabilities) -> bool:
    """Update the device attributes in Headscale using the tags API."""
    print(f"Starting update_device_attributes for device {device_id}")
    print(f"API base URL: {api_base_url}")
    print(f"Node ID: {node_id}, Node Port: {node_port}")
    
    async with aiohttp.ClientSession() as session:
        base_url = f"{api_base_url}/api/v1/node/{device_id}"
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

        try:
            # First try to get existing node data
            print(f"Fetching existing node data from: {base_url}")
            async with session.get(base_url, headers=headers) as response:
                status = response.status
                print(f"GET request status: {status}")
                
                if status == 200:
                    data = await response.json()
                    node_data = data.get('node', data)  # Handle nested 'node' structure
                    
                    print(f"Node name: {node_data.get('name', 'unknown')}")
                    existing_tags = node_data.get("forcedTags", [])
                    print(f"Existing tags: {existing_tags}")
                    
                    # Filter out existing exo tags
                    existing_tags = [tag for tag in existing_tags if not tag.startswith("tag:exo_")]
                    
                    # Add our new exo tags - using the required "tag:" prefix and ensuring lowercase
                    attributes = {
                        "node_id": node_id.replace('-', '_'),
                        "node_port": str(node_port),
                        "device_capability_chip": sanitize_attribute(device_capabilities.chip),
                        "device_capability_model": sanitize_attribute(device_capabilities.model),
                        "device_capability_memory": str(device_capabilities.memory),
                        "device_capability_flops_fp16": str(device_capabilities.flops.fp16),
                        "device_capability_flops_fp32": str(device_capabilities.flops.fp32),
                        "device_capability_flops_int8": str(device_capabilities.flops.int8)
                    }
                    
                    new_tags = existing_tags + [f"tag:exo_{k.lower()}={v}".lower() for k, v in attributes.items()]
                    print(f"New tags: {new_tags}")
                    
                    # Update node with new tags
                    update_url = f"{api_base_url}/api/v1/node/{device_id}/tags"
                    print(f"Sending update to: {update_url}")
                    
                    async with session.post(update_url, headers=headers, json={"tags": new_tags}) as response:
                        status = response.status
                        response_text = await response.text()
                        print(f"POST request status: {status}")
                        print(f"Response text: {response_text}")
                        
                        if status == 200:
                            print(f"Successfully updated device tags for device {device_id}")
                            return True
                        else:
                            print(f"Failed to update device tags: {status} - {response_text}")
                else:
                    response_text = await response.text()
                    print(f"Failed to get node data: {status} - {response_text}")
        except Exception as e:
            print(f"Error updating device tags: {e}")
            import traceback
            print(traceback.format_exc())
        
        return False


async def get_device_attributes(api_base_url: str, device_id: str, api_key: str) -> Tuple[Optional[str], Optional[int], Optional[DeviceCapabilities]]:
    """Get the device attributes from Headscale using the tags API."""
    async with aiohttp.ClientSession() as session:
        base_url = f"{api_base_url}/api/v1/node/{device_id}"
        headers = {'Authorization': f'Bearer {api_key}'}
        
        try:
            async with session.get(base_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    node_data = data.get('node', data)  # Handle nested 'node' structure
                    
                    tags = node_data.get("forcedTags", [])
                    
                    attributes = {}
                    for tag in tags:
                        if tag.startswith("tag:exo_"):
                            key, value = tag[8:].split("=", 1)  # Remove "tag:exo_" prefix and split on first "="
                            attributes[key] = value
                    
                    if not attributes:
                        return None, None, None
                    
                    node_id = attributes.get("node_id", "").replace('_', '-')
                    node_port = int(attributes.get("node_port", 0))
                    device_capabilities = DeviceCapabilities(
                        model=attributes.get("device_capability_model", "").replace('_', ' '),
                        chip=attributes.get("device_capability_chip", "").replace('_', ' '),
                        memory=int(attributes.get("device_capability_memory", 0)),
                        flops=DeviceFlops(
                            fp16=float(attributes.get("device_capability_flops_fp16", 0)),
                            fp32=float(attributes.get("device_capability_flops_fp32", 0)),
                            int8=float(attributes.get("device_capability_flops_int8", 0))
                        )
                    )
                    return node_id, node_port, device_capabilities
                else:
                    print(f"Failed to get node data: {response.status} - {await response.text()}")
        except Exception as e:
            print(f"Error getting device tags: {e}")
        
        return None, None, None


async def get_headscale_devices(api_base_url: str, api_key: str) -> Dict[str, Device]:
    """Get the list of devices from Headscale."""
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {api_key}"}
        url = f"{api_base_url}/api/v1/node"
        
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    nodes_data = data.get('nodes', data)
                    
                    devices = {}
                    for node_data in nodes_data:
                        # Handle nested structure - access node data whether directly or in 'node' key
                        node = node_data.get('node', node_data)
                        
                        if DEBUG_DISCOVERY >= 4:
                            print(f"Raw node data: {node}")
                            
                        # Create device with proper online status from API
                        device = Device(
                            device_id=str(node.get('id', '')),
                            name=node.get('name', ''),
                            addresses=node.get('ipAddresses', []),
                            online=node.get('online', False),
                        )
                        
                        if DEBUG_DISCOVERY >= 3:
                            print(f"Parsed device: {device.name}, online: {device.online}")
                            
                        devices[device.name] = device
                    return devices
                else:
                    if DEBUG_DISCOVERY >= 1:
                        error_text = await response.text()
                        print(f"Failed to get devices: {response.status} - {error_text}")
        except Exception as e:
            if DEBUG_DISCOVERY >= 1:
                print(f"Error getting devices from Headscale API: {e}")
    
    # Return empty dict if we couldn't get devices
    return {}
