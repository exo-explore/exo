"""Headscale-based peer discovery for Exo."""
import asyncio
import time
import traceback
from typing import List, Dict, Callable, Tuple, Optional
from exo.networking.discovery import Discovery
from exo.networking.peer_handle import PeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.helpers import DEBUG, DEBUG_DISCOVERY
from .headscale_helpers import get_device_id, update_device_attributes, get_device_attributes, get_headscale_devices, Device


class HeadscaleDiscovery(Discovery):
    """Discovery implementation for Headscale VPN."""
    
    def __init__(
        self,
        node_id: str,
        node_port: int,
        create_peer_handle: Callable[[str, str, str, DeviceCapabilities], PeerHandle],
        discovery_interval: int = 5,
        discovery_timeout: int = 30,
        update_interval: int = 15,
        device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
        headscale_api_key: str = None,
        headscale_api_base_url: str = None,
        allowed_node_ids: List[str] = None,
    ):
        """Initialize Headscale discovery.
        
        Args:
            node_id: ID of this node
            node_port: Port this node is listening on
            create_peer_handle: Function to create a peer handle
            discovery_interval: How often to run discovery (seconds)
            discovery_timeout: How long to wait for a peer to respond (seconds)
            update_interval: How often to update device attributes (seconds)
            device_capabilities: Capabilities of this device
            headscale_api_key: API key for Headscale
            headscale_api_base_url: Base URL for Headscale API
            allowed_node_ids: List of node IDs allowed to connect
        """
        self.node_id = node_id
        self.node_port = node_port
        self.create_peer_handle = create_peer_handle
        self.discovery_interval = discovery_interval
        self.discovery_timeout = discovery_timeout
        self.update_interval = update_interval
        self.device_capabilities = device_capabilities
        # Store peers as: peer_id -> (PeerHandle, last_health_check_time)
        self.known_peers: Dict[str, Tuple[PeerHandle, float]] = {}
        self.discovery_task = None
        self.cleanup_task = None
        self.update_task = None
        self.headscale_api_key = headscale_api_key
        self.headscale_api_base_url = headscale_api_base_url
        self.allowed_node_ids = allowed_node_ids
        self._device_id = None

    async def start(self):
        """Start the discovery process."""
        if not self.headscale_api_key or not self.headscale_api_base_url:
            print("Headscale API key or base URL not provided, Headscale discovery will not work")
            return

        self.device_capabilities = await device_capabilities()
        self.discovery_task = asyncio.create_task(self.task_discover_peers())
        self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())
        self.update_task = asyncio.create_task(self.task_update_device_posture_attributes())

    async def task_update_device_posture_attributes(self):
        """Periodically update the device attributes in Headscale."""
        while True:
            try:
                await self.update_device_posture_attributes()
                if DEBUG_DISCOVERY >= 2:
                    print(f"Updated device posture attributes")
            except Exception as e:
                print(f"Error updating device posture attributes: {e}")
                print(traceback.format_exc())
            finally:
                await asyncio.sleep(self.update_interval)

    async def get_device_id(self):
        """Get the device ID from tailscale status."""
        if self._device_id:
            return self._device_id
        self._device_id = await get_device_id()
        return self._device_id

    def _get_tailscale_status(self):
        """Get the current tailscale status data.
        
        Returns:
            dict: The tailscale status data as a dictionary
        """
        try:
            import subprocess
            import json
            result = subprocess.run(['tailscale', 'status', '--json'], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to get tailscale status: {result.stderr}")
                return {}
            return json.loads(result.stdout)
        except Exception as e:
            print(f"Error getting tailscale status: {e}")
            return {}

    async def update_device_posture_attributes(self):
        """Update the device attributes in Headscale."""
        try:
            device_id = await self.get_device_id()
            print(f"Attempting to update attributes for device ID: {device_id}")
            
            success = await update_device_attributes(
                self.headscale_api_base_url,
                device_id,
                self.headscale_api_key,
                self.node_id,
                self.node_port,
                self.device_capabilities
            )
            
            if success:
                print(f"Successfully updated attributes for device {device_id}")
            else:
                print(f"Failed to update attributes for device {device_id}. Check API key and URL.")
                
            return success
        except Exception as e:
            print(f"Error in update_device_posture_attributes: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    async def task_discover_peers(self):
        """Periodically discover peers in the Headscale network."""
        while True:
            try:
                devices: dict[str, Device] = await get_headscale_devices(
                    self.headscale_api_base_url,
                    self.headscale_api_key
                )
                current_time = time.time()

                if not devices and DEBUG_DISCOVERY >= 1:
                    print("No devices found from Headscale API. Check API key and URL.")
                    await asyncio.sleep(self.discovery_interval)
                    continue

                # Define active devices based solely on the online status from the API
                active_devices = {
                    name: device for name, device in devices.items() 
                    if device.online
                }

                if DEBUG_DISCOVERY >= 2: 
                    print(f"Active headscale devices: {len(active_devices)}/{len(devices)}")
                    if DEBUG_DISCOVERY >= 3 and devices:
                        print("Online status of headscale devices:", 
                              [(name, f"online: {device.online}") for name, device in devices.items()])

                for device in active_devices.values():
                    if device.name == self.node_id:
                        continue  # Skip self
                        
                    if not device.addresses:
                        if DEBUG_DISCOVERY >= 2: print(f"Device {device.name} has no addresses, skipping")
                        continue
                        
                    peer_host = device.addresses[0]
                    peer_attrs = await get_device_attributes(
                        self.headscale_api_base_url,
                        device.device_id,
                        self.headscale_api_key
                    )
                    
                    if peer_attrs[0] is None:
                        if DEBUG_DISCOVERY >= 4:
                            print(f"{device.device_id} does not have exo attributes. skipping.")
                        continue
                        
                    peer_id, peer_port, device_capabilities = peer_attrs

                    if self.allowed_node_ids and peer_id not in self.allowed_node_ids:
                        if DEBUG_DISCOVERY >= 2:
                            print(f"Ignoring peer {peer_id} as it's not in the allowed node IDs list")
                        continue

                    if peer_id not in self.known_peers or self.known_peers[peer_id][0].addr() != f"{peer_host}:{peer_port}":
                        new_peer_handle = self.create_peer_handle(
                            peer_id, 
                            f"{peer_host}:{peer_port}", 
                            "HS",  # Use HS for Headscale to distinguish from TS for Tailscale
                            device_capabilities
                        )
                        if not await new_peer_handle.health_check():
                            if DEBUG >= 1:
                                print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Skipping.")
                            continue

                        if DEBUG >= 1:
                            print(f"Adding {peer_id=} at {peer_host}:{peer_port}. Replace existing peer_id: {peer_id in self.known_peers}")
                        self.known_peers[peer_id] = (
                            new_peer_handle,
                            current_time,
                        )
                    else:
                        if not await self.known_peers[peer_id][0].health_check():
                            if DEBUG >= 1:
                                print(f"Peer {peer_id} at {peer_host}:{peer_port} is not healthy. Removing.")
                            if peer_id in self.known_peers:
                                del self.known_peers[peer_id]
                            continue
                        self.known_peers[peer_id] = (
                            self.known_peers[peer_id][0], 
                            current_time
                        )

            except Exception as e:
                print(f"Error in discover peers: {e}")
                print(traceback.format_exc())
            finally:
                await asyncio.sleep(self.discovery_interval)

    async def stop(self):
        """Stop the discovery process."""
        if self.discovery_task:
            self.discovery_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.update_task:
            self.update_task.cancel()
        if self.discovery_task or self.cleanup_task or self.update_task:
            await asyncio.gather(
                self.discovery_task, 
                self.cleanup_task, 
                self.update_task, 
                return_exceptions=True
            )

    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        """Discover peers in the network.
        
        Args:
            wait_for_peers: Number of peers to wait for before returning
            
        Returns:
            List of peer handles
        """
        if not self.headscale_api_key or not self.headscale_api_base_url:
            print("Headscale API key or base URL not provided, cannot discover peers")
            return []

        if wait_for_peers > 0:
            while len(self.known_peers) < wait_for_peers:
                if DEBUG_DISCOVERY >= 2:
                    print(f"Current peers: {len(self.known_peers)}/{wait_for_peers}. Waiting for more peers...")
                await asyncio.sleep(0.1)
        return [peer_handle for peer_handle, _ in self.known_peers.values()]

    async def task_cleanup_peers(self):
        """Periodically clean up stale peers."""
        while True:
            try:
                current_time = time.time()
                peers_to_remove = []

                peer_ids = list(self.known_peers.keys())
                results = await asyncio.gather(
                    *[self.check_peer(peer_id, current_time) for peer_id in peer_ids], 
                    return_exceptions=True
                )

                for peer_id, should_remove in zip(peer_ids, results):
                    if should_remove:
                        peers_to_remove.append(peer_id)

                if DEBUG_DISCOVERY >= 2:
                    print(
                        "Peer statuses:", {
                            peer_handle.id(): f"is_connected={await peer_handle.is_connected()}, health_check={await peer_handle.health_check()}, connected_at={connected_at}"
                            for peer_handle, connected_at in self.known_peers.values()
                        }
                    )

                for peer_id in peers_to_remove:
                    if peer_id in self.known_peers:
                        del self.known_peers[peer_id]
                        if DEBUG_DISCOVERY >= 2:
                            print(f"Removed peer {peer_id} due to inactivity or failed health check.")
            except Exception as e:
                print(f"Error in cleanup peers: {e}")
                print(traceback.format_exc())
            finally:
                await asyncio.sleep(self.discovery_interval)

    async def check_peer(self, peer_id: str, current_time: float) -> bool:
        """Check if a peer should be removed.
        
        Args:
            peer_id: The peer ID to check
            current_time: The current time
            
        Returns:
            True if the peer should be removed, False otherwise
        """
        if peer_id not in self.known_peers:
            return True

        peer_handle, connected_at = self.known_peers[peer_id]
        
        # If we haven't seen this peer in a while, check it's still healthy
        if current_time - connected_at > self.discovery_timeout:
            # Check if it's still connected
            is_connected = await peer_handle.is_connected()
            if not is_connected:
                # Try a health check
                is_healthy = await peer_handle.health_check()
                if not is_healthy:
                    return True

        return False
