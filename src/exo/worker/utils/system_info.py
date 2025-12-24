import socket
import subprocess
import sys
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.profiling import NetworkInterfaceInfo


def is_android() -> bool:
    """Check if running on Android/Termux."""
    return "termux" in sys.prefix.lower() or "android" in sys.prefix.lower()


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    # TODO: better non mac support
    if sys.platform != "darwin":  # 'darwin' is the platform name for macOS
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


def _parse_ifconfig_output(output: str) -> list[NetworkInterfaceInfo]:
    """Parse ifconfig output to extract interface names and IPv4 addresses."""
    interfaces: list[NetworkInterfaceInfo] = []
    current_iface: str | None = None
    
    for line in output.split('\n'):
        # Interface line starts without whitespace (e.g., "wlan0: flags=...")
        if line and not line[0].isspace() and ':' in line:
            current_iface = line.split(':')[0].strip()
        # IP address line contains "inet " (IPv4)
        elif current_iface and 'inet ' in line and 'inet6' not in line:
            parts = line.strip().split()
            for i, part in enumerate(parts):
                if part == 'inet' and i + 1 < len(parts):
                    ip = parts[i + 1]
                    # Some formats have "inet addr:x.x.x.x"
                    if ip.startswith('addr:'):
                        ip = ip[5:]
                    interfaces.append(NetworkInterfaceInfo(name=current_iface, ip_address=ip))
                    break
    
    return interfaces


def _get_ifconfig_interfaces() -> list[NetworkInterfaceInfo]:
    """Get network interfaces using ifconfig command (fallback for Termux/Android)."""
    try:
        result = subprocess.run(
            ['ifconfig'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return _parse_ifconfig_output(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return []


def get_external_ip_for_rpc() -> str | None:
    """
    Get the best external IP address for RPC/distributed inference.
    
    Specifically designed for Android/Termux where we need the wlan0 IP
    that other devices on the same network can reach.
    
    Priority:
    1. wlan0 interface (WiFi on Android)
    2. Any interface with a private network IP (10.x, 192.168.x, 172.16-31.x)
    3. Any non-loopback IPv4 address
    
    Returns None if no suitable IP is found.
    """
    interfaces = get_network_interfaces()
    
    # Priority 1: Look for wlan0 with a valid IP
    for iface in interfaces:
        if iface.name == 'wlan0' and _is_valid_external_ip(iface.ip_address):
            print(f"[EXO] Using wlan0 IP for RPC: {iface.ip_address}")
            return iface.ip_address
    
    # Priority 2: Look for any private network IP
    for iface in interfaces:
        ip = iface.ip_address
        if _is_preferred_private_ip(ip):
            print(f"[EXO] Using {iface.name} IP for RPC: {ip}")
            return ip
    
    # Priority 3: Any valid external IP
    for iface in interfaces:
        if _is_valid_external_ip(iface.ip_address):
            print(f"[EXO] Using {iface.name} IP for RPC (fallback): {iface.ip_address}")
            return iface.ip_address
    
    return None


def _is_valid_external_ip(ip: str) -> bool:
    """Check if an IP address is a valid external (non-loopback) address."""
    if not ip:
        return False
    if ip in ("127.0.0.1", "::1", "localhost", "0.0.0.0"):
        return False
    if ip.startswith("127."):
        return False
    if ":" in ip:  # Skip IPv6
        return False
    return True


def _is_preferred_private_ip(ip: str) -> bool:
    """Check if IP is on a preferred private network (10.x.x.x, 192.168.x.x, 172.16-31.x.x)."""
    if not _is_valid_external_ip(ip):
        return False
    if ip.startswith("10."):
        return True
    if ip.startswith("192.168."):
        return True
    if ip.startswith("172."):
        try:
            second_octet = int(ip.split(".")[1])
            if 16 <= second_octet <= 31:
                return True
        except (ValueError, IndexError):
            pass
    return False


def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information.
    
    On Android/Termux: Uses ifconfig FIRST (most reliable for getting wlan0 IP).
    On other platforms: Uses psutil as primary source.
    
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []

    # On Android/Termux, prioritize ifconfig (psutil often fails or returns wrong IPs)
    if is_android():
        ifconfig_interfaces = _get_ifconfig_interfaces()
        if ifconfig_interfaces:
            interfaces_info.extend(ifconfig_interfaces)
            # Log what we found for debugging
            for iface in ifconfig_interfaces:
                if not iface.ip_address.startswith('127.'):
                    print(f"[EXO] Found network interface: {iface.name} = {iface.ip_address}")
        
        # If ifconfig found useful IPs, return them (don't mix with psutil)
        has_useful_ip = any(
            not info.ip_address.startswith('127.') and ':' not in info.ip_address
            for info in interfaces_info
        )
        if has_useful_ip:
            return interfaces_info

    # Primary for non-Android or fallback: use psutil
    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(name=iface, ip_address=service.address)
                    )
                case _:
                    pass

    # Check if we got any useful IPs (not just loopback)
    has_useful_ip = any(
        not info.ip_address.startswith('127.') and ':' not in info.ip_address
        for info in interfaces_info
    )
    
    # Final fallback: use ifconfig if psutil didn't find useful IPs
    if not has_useful_ip:
        ifconfig_interfaces = _get_ifconfig_interfaces()
        if ifconfig_interfaces:
            existing_names = {info.name for info in interfaces_info}
            for iface in ifconfig_interfaces:
                if iface.name not in existing_names or not iface.ip_address.startswith('127.'):
                    interfaces_info.append(iface)

    return interfaces_info


async def get_model_and_chip() -> tuple[str, str]:
    """Get Mac system information using system_profiler."""
    model = "Unknown Model"
    chip = "Unknown Chip"

    # TODO: better non mac support
    if sys.platform != "darwin":
        return (model, chip)

    try:
        process = await run_process(
            [
                "system_profiler",
                "SPHardwareDataType",
            ]
        )
    except CalledProcessError:
        return (model, chip)

    # less interested in errors here because this value should be hard coded
    output = process.stdout.decode().strip()

    model_line = next(
        (line for line in output.split("\n") if "Model Name" in line), None
    )
    model = model_line.split(": ")[1] if model_line else "Unknown Model"

    chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
    chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"

    return (model, chip)
