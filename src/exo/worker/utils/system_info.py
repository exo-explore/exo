import asyncio
import re
import sys
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from exo.shared.types.profiling import NetworkInterfaceInfo


class SystemInfo(BaseModel):
    model_id: str
    chip_id: str
    memory: int
    network_interfaces: list[NetworkInterfaceInfo] = Field(default_factory=list)


async def get_mac_friendly_name_async() -> str | None:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    if sys.platform != 'darwin': # 'darwin' is the platform name for macOS
        print("This function is designed for macOS only.")
        return None

    try:
        # asyncio.create_subprocess_exec allows running external commands asynchronously.
        # stdout=asyncio.subprocess.PIPE captures standard output.
        # stderr=asyncio.subprocess.PIPE captures standard error.
        process = await asyncio.create_subprocess_exec(
            'scutil', '--get', 'ComputerName',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # process.communicate() reads all data from stdout and stderr
        # and waits for the process to terminate.
        # It returns a tuple (stdout_data, stderr_data).
        stdout_data, stderr_data = await process.communicate()

        # Check the return code of the process
        if process.returncode == 0:
            if stdout_data:
                # Decode from bytes to string and strip whitespace
                friendly_name = stdout_data.decode().strip()
                return friendly_name
            else:
                # Should not happen if returncode is 0, but good to check
                print("scutil command succeeded but produced no output.")
                return None
        else:
            # If there was an error, print the stderr output
            error_message = stderr_data.decode().strip() if stderr_data else "Unknown error"
            print(f"Error executing scutil (return code {process.returncode}): {error_message}")
            return None

    except FileNotFoundError:
        # This would happen if scutil is somehow not found, highly unlikely on a Mac.
        print("Error: 'scutil' command not found. Are you sure this is macOS?")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

async def get_network_interface_info_async() -> List[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    if sys.platform != 'darwin':
        return []

    interfaces_info: List[NetworkInterfaceInfo] = []
    device_to_type_map: Dict[str, str] = {}

    async def _run_cmd_async(command_parts: List[str]) -> Optional[str]:
        # Helper to run a command and return its stdout, or None on error.
        try:
            process = await asyncio.create_subprocess_exec(
                *command_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout_data, stderr_data = await process.communicate()
            if process.returncode == 0:
                # Use 'utf-8' and replace errors for robustness
                return stdout_data.decode('utf-8', errors='replace').strip()
            else:
                error_message = stderr_data.decode('utf-8', errors='replace').strip() if stderr_data else "Unknown error"
                print(f"Error executing {' '.join(command_parts)} (code {process.returncode}): {error_message}")
                return None
        except FileNotFoundError:
            print(f"Error: Command '{command_parts[0]}' not found. Ensure it's in PATH.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred running {' '.join(command_parts)}: {e}")
            return None

    # 1. Get hardware port types from networksetup
    networksetup_output = await _run_cmd_async(['networksetup', '-listallhardwareports'])
    if networksetup_output:
        current_hardware_port_type_raw: Optional[str] = None
        for line in networksetup_output.splitlines():
            line_stripped = line.strip()
            if line_stripped.startswith("Hardware Port:"):
                current_hardware_port_type_raw = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.startswith("Device:") and current_hardware_port_type_raw:
                device_name = line_stripped.split(":", 1)[1].strip()
                if device_name and device_name != "N/A":
                    if "Thunderbolt" in current_hardware_port_type_raw:
                        device_to_type_map[device_name] = 'thunderbolt'
                    elif "Wi-Fi" in current_hardware_port_type_raw or "AirPort" in current_hardware_port_type_raw:
                        device_to_type_map[device_name] = 'wifi'
                    elif "Ethernet" in current_hardware_port_type_raw or \
                         "LAN" in current_hardware_port_type_raw:
                        device_to_type_map[device_name] = 'ethernet'
                current_hardware_port_type_raw = None # Reset for the next block

    # 2. Get interface names and IP addresses from ifconfig
    ifconfig_output = await _run_cmd_async(['ifconfig'])
    if ifconfig_output:
        current_if_name: Optional[str] = None
        # Regex for interface name (e.g., en0:, utun0:, tailscale0.)
        interface_header_pattern = re.compile(r'^([a-zA-Z0-9\._-]+):')
        # Regex for IPv4 address (inet)
        inet_pattern = re.compile(r'^\s+inet\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
        # Regex for IPv6 address (inet6)
        inet6_pattern = re.compile(r'^\s+inet6\s+([0-9a-fA-F:]+(?:%[a-zA-Z0-9._-]+)?)')

        def _add_interface_entry(if_name: str, ip_addr: str):
            _if_type = device_to_type_map.get(if_name)
            if not _if_type: # Infer type if not found via networksetup
                if if_name.startswith(("utun", "wg", "ppp")) or "tailscale" in if_name:
                    _if_type = 'vpn'
                elif if_name.startswith("bridge"):
                    _if_type = 'virtual' # For non-Thunderbolt bridges (e.g., Docker)
                else:
                    _if_type = 'other'
            
            interfaces_info.append(NetworkInterfaceInfo(
                name=if_name,
                ip_address=ip_addr,
                type=_if_type
            ))

        for line in ifconfig_output.splitlines():
            header_match = interface_header_pattern.match(line)
            if header_match:
                potential_if_name = header_match.group(1)
                if potential_if_name == "lo0": # Skip loopback interface
                    current_if_name = None
                else:
                    current_if_name = potential_if_name
                continue

            if current_if_name:
                inet_m = inet_pattern.match(line)
                if inet_m:
                    ipv4_address = inet_m.group(1)
                    _add_interface_entry(current_if_name, ipv4_address) # Add all IPv4, including APIPA
                    continue

                inet6_m = inet6_pattern.match(line)
                if inet6_m:
                    ipv6_address = inet6_m.group(1)
                    # No specific filtering for IPv6 link-local (e.g., fe80::) for now.
                    _add_interface_entry(current_if_name, ipv6_address)
    
    return interfaces_info

async def get_mac_system_info_async() -> SystemInfo:
    """Get Mac system information using system_profiler."""
    model_id_val = "Unknown Model"
    chip_id_val = "Unknown Chip"
    memory_val = 0
    network_interfaces_info_list: List[NetworkInterfaceInfo] = []

    try:
        process = await asyncio.create_subprocess_exec(
            "system_profiler", "SPHardwareDataType",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout_data, stderr_data = await process.communicate()
        if process.returncode == 0:
            if stdout_data:
                output = stdout_data.decode().strip()
                model_line = next((line for line in output.split("\n") if "Model Name" in line), None)
                model_id_val = model_line.split(": ")[1] if model_line else "Unknown Model"

                chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
                chip_id_val = chip_line.split(": ")[1] if chip_line else "Unknown Chip"

                memory_line = next((line for line in output.split("\n") if "Memory" in line), None)
                memory_str = memory_line.split(": ")[1] if memory_line else "0 GB" # Default to "0 GB"
                memory_units = memory_str.split()
                if len(memory_units) == 2:
                    try:
                        memory_value_int = int(memory_units[0])
                        if memory_units[1] == "GB":
                            memory_val = memory_value_int * 1024 # Assuming MB
                        elif memory_units[1] == "MB":
                             memory_val = memory_value_int
                        else: # TB? Unlikely for typical memory, handle gracefully
                            memory_val = memory_value_int # Store as is, let consumer decide unit or log
                            print(f"Warning: Unknown memory unit {memory_units[1]}")
                    except ValueError:
                        print(f"Warning: Could not parse memory value {memory_units[0]}")
                        memory_val = 0

            else:
                print("system_profiler command succeeded but produced no output for hardware.")
        else:
            error_message = stderr_data.decode().strip() if stderr_data else "Unknown error"
            print(f"Error executing system_profiler (return code {process.returncode}): {error_message}")
    except Exception as e:
        print(f"Error getting Mac hardware info: {e}")

    # Call the new function to get network info
    try:
        network_interfaces_info_list = await get_network_interface_info_async()
    except Exception as e:
        print(f"Error getting Mac network interface info: {e}")
        network_interfaces_info_list = []


    return SystemInfo(
        model_id=model_id_val,
        chip_id=chip_id_val,
        memory=memory_val,
        network_interfaces=network_interfaces_info_list
    )
