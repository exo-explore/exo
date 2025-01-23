import os
import sys
import asyncio
from typing import Callable, TypeVar, Optional, Dict, Generic, Tuple, List
import socket
import random
import platform
import psutil
import uuid
from scapy.all import get_if_addr, get_if_list
import re
import subprocess
from pathlib import Path
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
import traceback

DEBUG = int(os.getenv("DEBUG", default="0"))
DEBUG_DISCOVERY = int(os.getenv("DEBUG_DISCOVERY", default="0"))
VERSION = "0.0.1"

exo_text = r"""
  _____  _____  
 / _ \ \/ / _ \ 
|  __/>  < (_) |
 \___/_/\_\___/ 
    """

# Single shared thread pool for subprocess operations
subprocess_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="subprocess_worker")


def get_system_info():
  if psutil.MACOS:
    if platform.machine() == "arm64":
      return "Apple Silicon Mac"
    if platform.machine() in ["x86_64", "i386"]:
      return "Intel Mac"
    return "Unknown Mac architecture"
  if psutil.LINUX:
    return "Linux"
  return "Non-Mac, non-Linux system"


def find_available_port(host: str = "", min_port: int = 49152, max_port: int = 65535) -> int:
  used_ports_file = os.path.join(tempfile.gettempdir(), "exo_used_ports")

  def read_used_ports():
    if os.path.exists(used_ports_file):
      with open(used_ports_file, "r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]
    return []

  def write_used_port(port, used_ports):
    with open(used_ports_file, "w") as f:
      print(used_ports[-19:])
      for p in used_ports[-19:] + [port]:
        f.write(f"{p}\n")

  used_ports = read_used_ports()
  available_ports = set(range(min_port, max_port + 1)) - set(used_ports)

  while available_ports:
    port = random.choice(list(available_ports))
    if DEBUG >= 2: print(f"Trying to find available port {port=}")
    try:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
      write_used_port(port, used_ports)
      return port
    except socket.error:
      available_ports.remove(port)

  raise RuntimeError("No available ports in the specified range")


def print_exo():
  print(exo_text)


def print_yellow_exo():
  yellow = "\033[93m"  # ANSI escape code for yellow
  reset = "\033[0m"  # ANSI escape code to reset color
  print(f"{yellow}{exo_text}{reset}")


def terminal_link(uri, label=None):
  if label is None:
    label = uri
  parameters = ""

  # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST
  escape_mask = "\033]8;{};{}\033\\{}\033]8;;\033\\"

  return escape_mask.format(parameters, uri, label)


T = TypeVar("T")
K = TypeVar("K")


class AsyncCallback(Generic[T]):
  def __init__(self) -> None:
    self.condition: asyncio.Condition = asyncio.Condition()
    self.result: Optional[Tuple[T, ...]] = None
    self.observers: list[Callable[..., None]] = []

  async def wait(self, check_condition: Callable[..., bool], timeout: Optional[float] = None) -> Tuple[T, ...]:
    async with self.condition:
      await asyncio.wait_for(self.condition.wait_for(lambda: self.result is not None and check_condition(*self.result)), timeout)
      assert self.result is not None  # for type checking
      return self.result

  def on_next(self, callback: Callable[..., None]) -> None:
    self.observers.append(callback)

  def set(self, *args: T) -> None:
    self.result = args
    for observer in self.observers:
      observer(*args)
    asyncio.create_task(self.notify())

  async def notify(self) -> None:
    async with self.condition:
      self.condition.notify_all()


class AsyncCallbackSystem(Generic[K, T]):
  def __init__(self) -> None:
    self.callbacks: Dict[K, AsyncCallback[T]] = {}

  def register(self, name: K) -> AsyncCallback[T]:
    if name not in self.callbacks:
      self.callbacks[name] = AsyncCallback[T]()
    return self.callbacks[name]

  def deregister(self, name: K) -> None:
    if name in self.callbacks:
      del self.callbacks[name]

  def trigger(self, name: K, *args: T) -> None:
    if name in self.callbacks:
      self.callbacks[name].set(*args)

  def trigger_all(self, *args: T) -> None:
    for callback in self.callbacks.values():
      callback.set(*args)


K = TypeVar('K', bound=str)
V = TypeVar('V')


class PrefixDict(Generic[K, V]):
  def __init__(self):
    self.items: Dict[K, V] = {}

  def add(self, key: K, value: V) -> None:
    self.items[key] = value

  def find_prefix(self, argument: str) -> List[Tuple[K, V]]:
    return [(key, value) for key, value in self.items.items() if argument.startswith(key)]

  def find_longest_prefix(self, argument: str) -> Optional[Tuple[K, V]]:
    matches = self.find_prefix(argument)
    if len(matches) == 0:
      return None

    return max(matches, key=lambda x: len(x[0]))


def is_valid_uuid(val):
  try:
    uuid.UUID(str(val))
    return True
  except ValueError:
    return False


def get_or_create_node_id():
  NODE_ID_FILE = Path(tempfile.gettempdir())/".exo_node_id"
  try:
    if NODE_ID_FILE.is_file():
      with open(NODE_ID_FILE, "r") as f:
        stored_id = f.read().strip()
      if is_valid_uuid(stored_id):
        if DEBUG >= 2: print(f"Retrieved existing node ID: {stored_id}")
        return stored_id
      else:
        if DEBUG >= 2: print("Stored ID is not a valid UUID. Generating a new one.")

    new_id = str(uuid.uuid4())
    with open(NODE_ID_FILE, "w") as f:
      f.write(new_id)

    if DEBUG >= 2: print(f"Generated and stored new node ID: {new_id}")
    return new_id
  except IOError as e:
    if DEBUG >= 2: print(f"IO error creating node_id: {e}")
    return str(uuid.uuid4())
  except Exception as e:
    if DEBUG >= 2: print(f"Unexpected error creating node_id: {e}")
    return str(uuid.uuid4())


def pretty_print_bytes(size_in_bytes: int) -> str:
  if size_in_bytes < 1024:
    return f"{size_in_bytes} B"
  elif size_in_bytes < 1024**2:
    return f"{size_in_bytes / 1024:.2f} KB"
  elif size_in_bytes < 1024**3:
    return f"{size_in_bytes / (1024 ** 2):.2f} MB"
  elif size_in_bytes < 1024**4:
    return f"{size_in_bytes / (1024 ** 3):.2f} GB"
  else:
    return f"{size_in_bytes / (1024 ** 4):.2f} TB"


def pretty_print_bytes_per_second(bytes_per_second: int) -> str:
  if bytes_per_second < 1024:
    return f"{bytes_per_second} B/s"
  elif bytes_per_second < 1024**2:
    return f"{bytes_per_second / 1024:.2f} KB/s"
  elif bytes_per_second < 1024**3:
    return f"{bytes_per_second / (1024 ** 2):.2f} MB/s"
  elif bytes_per_second < 1024**4:
    return f"{bytes_per_second / (1024 ** 3):.2f} GB/s"
  else:
    return f"{bytes_per_second / (1024 ** 4):.2f} TB/s"


def get_all_ip_addresses_and_interfaces():
    ip_addresses = []
    for interface in get_if_list():
      try:
        ip = get_if_addr(interface)
        if ip.startswith("0.0."): continue
        simplified_interface = re.sub(r'^\\Device\\NPF_', '', interface)
        ip_addresses.append((ip, simplified_interface))
      except:
        if DEBUG >= 1: print(f"Failed to get IP address for interface {interface}")
        if DEBUG >= 1: traceback.print_exc()
    if not ip_addresses:
      if DEBUG >= 1: print("Failed to get any IP addresses. Defaulting to localhost.")
      return [("localhost", "lo")]
    return list(set(ip_addresses))



async def get_macos_interface_type(ifname: str) -> Optional[Tuple[int, str]]:
  try:
    # Use the shared subprocess_pool
    output = await asyncio.get_running_loop().run_in_executor(
      subprocess_pool, lambda: subprocess.run(['system_profiler', 'SPNetworkDataType', '-json'], capture_output=True, text=True, close_fds=True).stdout
    )

    data = json.loads(output)

    for interface in data.get('SPNetworkDataType', []):
      if interface.get('interface') == ifname:
        hardware = interface.get('hardware', '').lower()
        type_name = interface.get('type', '').lower()
        name = interface.get('_name', '').lower()

        if 'thunderbolt' in name:
          return (5, "Thunderbolt")
        if hardware == 'ethernet' or type_name == 'ethernet':
          if 'usb' in name:
            return (4, "Ethernet [USB]")
          return (4, "Ethernet")
        if hardware == 'airport' or type_name == 'airport' or 'wi-fi' in name:
          return (3, "WiFi")
        if type_name == 'vpn':
          return (1, "External Virtual")

  except Exception as e:
    if DEBUG >= 2: print(f"Error detecting macOS interface type: {e}")

  return None


async def get_interface_priority_and_type(ifname: str) -> Tuple[int, str]:
  # On macOS, try to get interface type using networksetup
  if psutil.MACOS:
    macos_type = await get_macos_interface_type(ifname)
    if macos_type is not None: return macos_type

  # Local container/virtual interfaces
  if (ifname.startswith(('docker', 'br-', 'veth', 'cni', 'flannel', 'calico', 'weave')) or 'bridge' in ifname):
    return (7, "Container Virtual")

  # Loopback interface
  if ifname.startswith('lo'):
    return (6, "Loopback")

  # Traditional detection for non-macOS systems or fallback
  if ifname.startswith(('tb', 'nx', 'ten')):
    return (5, "Thunderbolt")

  # Regular ethernet detection
  if ifname.startswith(('eth', 'en')) and not ifname.startswith(('en1', 'en0')):
    return (4, "Ethernet")

  # WiFi detection
  if ifname.startswith(('wlan', 'wifi', 'wl')) or ifname in ['en0', 'en1']:
    return (3, "WiFi")

  # Non-local virtual interfaces (VPNs, tunnels)
  if ifname.startswith(('tun', 'tap', 'vtun', 'utun', 'gif', 'stf', 'awdl', 'llw')):
    return (1, "External Virtual")

  # Other physical interfaces
  return (2, "Other")


async def shutdown(signal, loop, server):
  """Gracefully shutdown the server and close the asyncio loop."""
  print(f"Received exit signal {signal.name}...")
  print("Thank you for using exo.")
  print_yellow_exo()
  server_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
  [task.cancel() for task in server_tasks]
  print(f"Cancelling {len(server_tasks)} outstanding tasks")
  await asyncio.gather(*server_tasks, return_exceptions=True)
  await server.stop()


def is_frozen():
  return getattr(sys, 'frozen', False) or os.path.basename(sys.executable) == "exo" \
    or ('Contents/MacOS' in str(os.path.dirname(sys.executable))) \
    or '__nuitka__' in globals() or getattr(sys, '__compiled__', False)

async def get_mac_system_info() -> Tuple[str, str, int]:
    """Get Mac system information using system_profiler."""
    try:
        output = await asyncio.get_running_loop().run_in_executor(
            subprocess_pool,
            lambda: subprocess.check_output(["system_profiler", "SPHardwareDataType"]).decode("utf-8")
        )
        
        model_line = next((line for line in output.split("\n") if "Model Name" in line), None)
        model_id = model_line.split(": ")[1] if model_line else "Unknown Model"
        
        chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
        chip_id = chip_line.split(": ")[1] if chip_line else "Unknown Chip"
        
        memory_line = next((line for line in output.split("\n") if "Memory" in line), None)
        memory_str = memory_line.split(": ")[1] if memory_line else "Unknown Memory"
        memory_units = memory_str.split()
        memory_value = int(memory_units[0])
        memory = memory_value * 1024 if memory_units[1] == "GB" else memory_value
        
        return model_id, chip_id, memory
    except Exception as e:
        if DEBUG >= 2: print(f"Error getting Mac system info: {e}")
        return "Unknown Model", "Unknown Chip", 0

def get_exo_home() -> Path:
  if psutil.WINDOWS: docs_folder = Path(os.environ["USERPROFILE"])/"Documents"
  else: docs_folder = Path.home()/"Documents"
  if not docs_folder.exists(): docs_folder.mkdir(exist_ok=True)
  exo_folder = docs_folder/"Exo"
  if not exo_folder.exists(): exo_folder.mkdir(exist_ok=True)
  return exo_folder


def get_exo_images_dir() -> Path:
  exo_home = get_exo_home()
  images_dir = exo_home/"Images"
  if not images_dir.exists(): images_dir.mkdir(exist_ok=True)
  return images_dir
