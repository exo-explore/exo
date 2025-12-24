"""
Utility functions for the llama.cpp engine.
"""

import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Any, Final

from exo.shared.types.worker.instances import BoundInstance, LlamaCppInstance
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.llamacpp.constants import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_N_BATCH,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_N_THREADS,
)
from exo.worker.runner.bootstrap import logger


DISTRIBUTED_SERVER_STARTUP_TIMEOUT: Final[int] = 600  # 10 minutes for slow tensor transfer


def is_android() -> bool:
    """Check if running on Android/Termux."""
    return "termux" in sys.prefix.lower() or "android" in sys.prefix.lower()


def use_native_cli() -> bool:
    """Check if we should use native CLI instead of Python bindings."""
    # Use native CLI on Android where Python bindings may segfault
    if os.environ.get("EXO_LLAMA_NATIVE_CLI", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("EXO_LLAMA_PYTHON_BINDINGS", "").lower() in ("1", "true", "yes"):
        return False
    # Default: use native CLI on Android
    return is_android()


def use_server_mode() -> bool:
    """
    Check if we should use llama-server HTTP API instead of subprocess.
    
    Server mode is more reliable on Android because:
    - No TTY/stdin issues
    - Model stays loaded between requests
    - Proper streaming support
    """
    # Explicit override via environment variable
    if os.environ.get("EXO_LLAMA_SERVER", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("EXO_LLAMA_SERVER", "").lower() in ("0", "false", "no"):
        return False
    # Default: use server mode on Android for reliability
    return is_android()


def find_gguf_file(model_path: Path) -> Path | None:
    """
    Find a GGUF file in the model directory.
    Returns the path to the first GGUF file found, or None.
    """
    if model_path.is_file() and model_path.suffix == ".gguf":
        return model_path

    if model_path.is_dir():
        gguf_files = list(model_path.glob("*.gguf"))
        if gguf_files:
            # Prefer quantized versions for memory efficiency
            for preferred in ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0"]:
                for f in gguf_files:
                    if preferred in f.name:
                        return f
            return gguf_files[0]

    return None


def get_gguf_path_for_instance(bound_instance: BoundInstance) -> Path:
    """Get the GGUF file path for an instance."""
    shard_metadata = bound_instance.bound_shard
    model_id = str(shard_metadata.model_meta.model_id)
    model_path = build_model_path(model_id)
    gguf_path = find_gguf_file(model_path)
    
    if gguf_path is None:
        raise FileNotFoundError(
            f"No GGUF file found in {model_path}. "
            "Please ensure you're using a GGUF-format model for llama.cpp."
        )
    
    return gguf_path


def initialize_llamacpp(
    bound_instance: BoundInstance,
) -> Any:
    """
    Initialize a llama.cpp model for inference.
    Returns the loaded Llama model instance, or a native CLI wrapper on Android.
    """
    shard_metadata = bound_instance.bound_shard
    model_id = str(shard_metadata.model_meta.model_id)

    logger.info(f"Initializing llama.cpp for model: {model_id}")

    gguf_path = get_gguf_path_for_instance(bound_instance)
    logger.info(f"Loading GGUF model from: {gguf_path}")

    # Configure threading based on environment or auto-detect
    n_threads = int(os.environ.get("LLAMA_N_THREADS", DEFAULT_N_THREADS))
    if n_threads == 0:
        n_threads = os.cpu_count() or 4

    n_ctx = int(os.environ.get("LLAMA_N_CTX", DEFAULT_CONTEXT_SIZE))

    # Use native CLI on Android to avoid Python binding segfaults
    if use_native_cli():
        logger.info("Using NATIVE llama-cli (Python bindings disabled on Android)")
        from exo.worker.engines.llamacpp.native_cli import NativeLlamaCpp
        
        return NativeLlamaCpp(
            model_path=str(gguf_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
        )

    # Use Python bindings on desktop
    logger.info("Using Python llama-cpp bindings")
    from llama_cpp import Llama
    
    n_gpu_layers = int(os.environ.get("LLAMA_N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS))
    n_batch = int(os.environ.get("LLAMA_N_BATCH", DEFAULT_N_BATCH))

    logger.info(
        f"llama.cpp config: threads={n_threads}, gpu_layers={n_gpu_layers}, "
        f"ctx={n_ctx}, batch={n_batch}"
    )

    logger.info(f">>> STARTING Llama() constructor for {gguf_path}")
    logger.info(f">>> File size: {gguf_path.stat().st_size / (1024*1024):.2f} MB")
    
    import time
    load_start = time.time()
    
    use_mmap = not is_android()
    logger.info(f">>> use_mmap={use_mmap}")
    
    model = Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        use_mmap=use_mmap,
        verbose=True,
    )
    
    load_time = time.time() - load_start
    logger.info(f">>> FINISHED Llama() constructor in {load_time:.2f} seconds")

    return model


def get_model_info(model: Any) -> dict[str, object]:
    """
    Get information about the loaded model.
    """
    info: dict[str, object] = {
        "n_ctx": model.n_ctx(),
        "n_vocab": model.n_vocab(),
    }

    if hasattr(model, "metadata") and model.metadata:
        info["metadata"] = dict(model.metadata)

    return info


def is_distributed_instance(bound_instance: BoundInstance) -> bool:
    """Check if this is a distributed multi-node instance."""
    instance = bound_instance.instance
    if not isinstance(instance, LlamaCppInstance):
        return False
    return instance.is_distributed


def _is_valid_rpc_ip(ip: str) -> bool:
    """Check if an IP is valid for RPC connections (not localhost/loopback)."""
    if not ip:
        return False
    if ip in ("127.0.0.1", "::1", "localhost", "0.0.0.0"):
        return False
    if ip.startswith("127."):
        return False
    return True


def _get_current_node_ips() -> set[str]:
    """Get all IP addresses of the current node for self-connection detection.
    
    Uses ifconfig on Android/Termux (ip command doesn't work there).
    """
    import socket

    local_ips: set[str] = {"127.0.0.1", "localhost", "::1"}
    try:
        hostname = socket.gethostname()
        try:
            addrs = socket.getaddrinfo(hostname, None, socket.AF_INET)
            for addr in addrs:
                local_ips.add(addr[4][0])
        except socket.gaierror:
            pass

        # Try ifconfig first (works on Android/Termux)
        ifconfig_ips = _get_ips_from_ifconfig()
        if ifconfig_ips:
            local_ips.update(ifconfig_ips)
        else:
            # Fall back to ip command
            try:
                result = subprocess.run(
                    ["ip", "-4", "addr", "show"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    import re
                    ips = re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", result.stdout)
                    local_ips.update(ips)
            except Exception:
                pass
    except Exception:
        pass

    return local_ips


def _get_ips_from_ifconfig() -> list[str]:
    """Get IP addresses using ifconfig command (works on Android/Termux)."""
    ips: list[str] = []
    try:
        result = subprocess.run(
            ["ifconfig"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'inet ' in line and 'inet6' not in line:
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if part == 'inet' and i + 1 < len(parts):
                            ip = parts[i + 1]
                            if ip.startswith('addr:'):
                                ip = ip[5:]
                            ips.append(ip)
                            break
    except Exception:
        pass
    return ips


def build_rpc_address_list(bound_instance: BoundInstance) -> str:
    """
    Build the RPC address list for llama.cpp --rpc flag.

    Format: "ip1:port1,ip2:port2,..."
    Only includes worker nodes (device_rank > 0).
    Hosts are ordered by device_rank in the instance.

    Validates that IPs are external (not localhost) and logs warnings
    for any configuration issues. Also detects if the master node's own IP
    is incorrectly included in the worker list (a sign of cycle ordering bugs).
    """
    instance = bound_instance.instance
    if not isinstance(instance, LlamaCppInstance):
        return ""

    shard_assignments = instance.shard_assignments
    rpc_addresses: list[str] = []
    invalid_workers: list[str] = []

    master_device_rank = bound_instance.bound_shard.device_rank
    current_node_ips = _get_current_node_ips()

    logger.info(f"Building RPC address list for {len(instance.hosts)} hosts")
    logger.info(f"Instance hosts: {[(h.ip, h.port) for h in instance.hosts]}")
    logger.info(f"RPC ports: {instance.rpc_ports}")
    logger.info(f"Tensor split: {instance.tensor_split}")
    logger.info(f"Current node device_rank: {master_device_rank}")
    logger.info(f"Current node IPs: {current_node_ips}")

    for node_id, runner_id in shard_assignments.node_to_runner.items():
        shard = shard_assignments.runner_to_shard.get(runner_id)
        if shard is None:
            continue

        if shard.device_rank == 0:
            logger.debug(f"Skipping master node {node_id} (rank 0)")
            continue

        rpc_port = instance.rpc_ports.get(node_id, 0)
        if rpc_port == 0:
            logger.warning(f"No RPC port for node {node_id} (rank {shard.device_rank})")
            invalid_workers.append(f"{node_id}:no_port")
            continue

        host_index = shard.device_rank
        if host_index < len(instance.hosts):
            host = instance.hosts[host_index]

            if not _is_valid_rpc_ip(host.ip):
                logger.error(
                    f"Invalid IP for worker rank {shard.device_rank}: '{host.ip}'. "
                    "Distributed inference requires external network IPs, not localhost. "
                    "Check network topology and ensure devices are on the same VLAN/WiFi."
                )
                invalid_workers.append(f"{node_id}:{host.ip}")
                continue

            if master_device_rank == 0 and host.ip in current_node_ips:
                logger.error(
                    f"CYCLE ORDERING BUG DETECTED: Worker IP {host.ip} matches this master node's IP! "
                    f"This node (device_rank={master_device_rank}) should not connect to itself. "
                    "Skipping this address to prevent self-connection. "
                    "This indicates a mismatch between cycle ordering and device_rank assignment."
                )
                invalid_workers.append(f"{node_id}:self_ip")
                continue

            rpc_address = f"{host.ip}:{rpc_port}"
            logger.info(f"Worker rank {shard.device_rank} ({node_id}): {rpc_address}")
            rpc_addresses.append(rpc_address)
        else:
            logger.warning(
                f"No host found for device_rank {shard.device_rank} "
                f"(only {len(instance.hosts)} hosts in instance)"
            )
            invalid_workers.append(f"{node_id}:no_host")

    if invalid_workers:
        logger.warning(
            f"Some workers have invalid configurations: {invalid_workers}. "
            "These workers will not participate in distributed inference."
        )

    result = ",".join(rpc_addresses)
    
    if result:
        logger.info(f"Built RPC address list: {result}")
    else:
        logger.error(
            "RPC address list is empty! No valid worker IPs found. "
            "Distributed inference will fail. Check that:\n"
            "  1. Devices are on the same network (same VLAN/WiFi)\n"
            "  2. Client isolation is disabled on the network\n"
            "  3. Devices have discovered each other with real IPs (not localhost)"
        )
    
    return result


def build_tensor_split_string(bound_instance: BoundInstance) -> str:
    """
    Build the tensor split string for llama.cpp --tensor-split flag.

    Format: "ratio1,ratio2,ratio3,..."
    """
    instance = bound_instance.instance
    if not isinstance(instance, LlamaCppInstance):
        return ""

    if not instance.tensor_split:
        return ""

    return ",".join(str(ratio) for ratio in instance.tensor_split)


def find_llama_server() -> Path | None:
    """Find the llama-server binary."""
    search_paths = [
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-server",
        Path("/usr/local/bin/llama-server"),
        Path("/usr/bin/llama-server"),
    ]

    for path in search_paths:
        if path.exists() and os.access(path, os.X_OK):
            return path

    return None


def verify_llama_server_rpc_support(server_path: Path) -> bool:
    """
    Verify that llama-server was built with RPC support (-DGGML_RPC=ON).

    Returns True if the binary supports --rpc flag, False otherwise.
    """
    try:
        result = subprocess.run(
            [str(server_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Check if --rpc appears in the help output
        return "--rpc" in result.stdout or "--rpc" in result.stderr
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as exc:
        logger.warning(f"Failed to check llama-server RPC support: {exc}")
        return False


RPC_BUILD_INSTRUCTIONS = """
llama-server was not built with RPC support. To enable distributed inference:

  cd ~/llama.cpp
  rm -rf build
  cmake -B build -DGGML_RPC=ON -DBUILD_SHARED_LIBS=ON
  cmake --build build --target llama-server rpc-server

Then restart EXO. The installer script (scripts/install_exo-termux.sh) now
builds with RPC enabled by default.
""".strip()


def get_lib_path() -> str:
    """Get LD_LIBRARY_PATH for llama.cpp libraries."""
    lib_dirs = [
        Path.home() / "llama.cpp" / "build" / "bin",
        Path.home() / "llama.cpp" / "build" / "lib",
    ]
    return ":".join(str(directory) for directory in lib_dirs if directory.exists())


def is_rpc_port_responding(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if an RPC server is responding on the given host:port."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((host, port))
            return True
    except socket.timeout:
        return False
    except OSError as e:
        # Log connection errors with error code for debugging
        errno = getattr(e, 'errno', None)
        logger.warning(f"Cannot connect to {host}:{port} - errno={errno}: {e}")
        return False


def precheck_rpc_connectivity(rpc_addresses: str) -> dict[str, bool]:
    """
    Pre-check connectivity to all RPC workers without waiting.
    
    Returns a dict mapping each address to its reachability status.
    This is useful for quick diagnostics before starting the full wait loop.
    """
    if not rpc_addresses:
        return {}
    
    addresses = [addr.strip() for addr in rpc_addresses.split(",") if addr.strip()]
    results: dict[str, bool] = {}
    
    logger.info("Pre-checking RPC worker connectivity...")
    
    for addr in addresses:
        try:
            host, port_str = addr.split(":")
            port = int(port_str)
            reachable = is_rpc_port_responding(host, port, timeout=3.0)
            results[addr] = reachable
            status = "REACHABLE" if reachable else "UNREACHABLE"
            logger.info(f"  {addr}: {status}")
        except (ValueError, IndexError):
            logger.warning(f"  {addr}: INVALID FORMAT")
            results[addr] = False
    
    reachable_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    if reachable_count == total_count:
        logger.info(f"Pre-check: All {total_count} workers reachable")
    else:
        logger.warning(
            f"Pre-check: Only {reachable_count}/{total_count} workers reachable. "
            "Distributed inference may fail or be delayed."
        )
    
    return results


def wait_for_rpc_workers(rpc_addresses: str, timeout: int = 120) -> bool:
    """
    Wait for all RPC worker servers to be available.

    Polls each worker's RPC port until they're all responding or timeout is reached.
    Returns True if all workers are ready, False if timeout occurred.
    """
    if not rpc_addresses:
        return True

    addresses = [addr.strip() for addr in rpc_addresses.split(",") if addr.strip()]
    if not addresses:
        return True

    logger.info(f"Checking connectivity to {len(addresses)} RPC worker(s)...")
    
    # Track which workers we've seen become ready
    workers_ready: set[str] = set()

    start_time = time.time()
    last_log_time = start_time
    
    while time.time() - start_time < timeout:
        not_ready: list[str] = []

        for addr in addresses:
            if addr in workers_ready:
                continue
                
            try:
                host, port_str = addr.split(":")
                port = int(port_str)
                if is_rpc_port_responding(host, port):
                    if addr not in workers_ready:
                        logger.info(f"  Worker {addr}: READY")
                        workers_ready.add(addr)
                else:
                    not_ready.append(addr)
            except (ValueError, IndexError):
                logger.warning(f"Invalid RPC address format: {addr}")
                not_ready.append(addr)

        if len(workers_ready) == len(addresses):
            elapsed = int(time.time() - start_time)
            logger.info(f"All {len(addresses)} RPC workers ready after {elapsed}s")
            return True

        elapsed = int(time.time() - start_time)
        now = time.time()
        
        # Log progress every 10 seconds
        if now - last_log_time >= 10:
            ready_count = len(workers_ready)
            total_count = len(addresses)
            logger.info(
                f"Waiting for workers: {ready_count}/{total_count} ready, "
                f"pending: {not_ready} ({elapsed}s / {timeout}s)"
            )
            last_log_time = now
            
        time.sleep(2)

    # Log final status
    elapsed = int(time.time() - start_time)
    missing = [addr for addr in addresses if addr not in workers_ready]
    logger.error(
        f"Timeout after {elapsed}s waiting for RPC workers. "
        f"Ready: {list(workers_ready)}, Missing: {missing}"
    )
    return False


class DistributedLlamaServer:
    """
    Manages a llama-server instance for distributed inference (master node).

    The master node runs llama-server with --rpc flag to connect to worker
    RPC servers and distribute tensor operations across nodes.
    """

    def __init__(
        self,
        model_path: str,
        rpc_addresses: str,
        tensor_split: str,
        port: int = 8080,
    ) -> None:
        self.model_path = model_path
        self.rpc_addresses = rpc_addresses
        self.tensor_split = tensor_split
        self.port = port
        self.process: subprocess.Popen[bytes] | None = None
        self.server_path = find_llama_server()
        self.lib_path = get_lib_path()

    def start(self, max_retries: int = 3) -> bool:
        """Start the distributed llama-server.
        
        Waits for all RPC workers to be available before starting,
        and retries on failure with exponential backoff.
        """
        if self.server_path is None:
            logger.error("llama-server not found. Build with: cmake --build build --target llama-server")
            return False

        logger.info(f"Using llama-server binary: {self.server_path}")

        # Verify llama-server has RPC support before attempting distributed inference
        if self.rpc_addresses:
            has_rpc = verify_llama_server_rpc_support(self.server_path)
            logger.info(f"llama-server RPC support: {has_rpc}")
            if not has_rpc:
                logger.error(RPC_BUILD_INSTRUCTIONS)
                return False

        # Pre-check: verify we can reach all workers before starting
        if self.rpc_addresses:
            logger.info("=" * 60)
            logger.info("DISTRIBUTED INFERENCE STARTUP")
            logger.info("=" * 60)
            logger.info(f"Model: {self.model_path}")
            logger.info(f"RPC workers: {self.rpc_addresses}")
            logger.info(f"Tensor split: {self.tensor_split}")
            logger.info("=" * 60)
            
            # Log individual worker addresses
            addresses = [addr.strip() for addr in self.rpc_addresses.split(",") if addr.strip()]
            for i, addr in enumerate(addresses):
                logger.info(f"  Worker {i + 1}: {addr}")
            
            logger.info("Waiting for worker RPC servers to be ready...")
            # Longer timeout for Android WiFi (180 seconds)
            if not wait_for_rpc_workers(self.rpc_addresses, timeout=180):
                logger.error(
                    "Worker RPC servers not available. Troubleshooting:\n"
                    "  1. Ensure all workers have started their rpc-server\n"
                    "  2. Check that devices are on the same network/VLAN\n"
                    "  3. Verify client isolation is disabled on the WiFi\n"
                    "  4. Test connectivity manually: nc -zv <worker-ip> <port>"
                )
                return False
            logger.info("All workers are ready!")

        # Verify model file exists before starting
        model_file = Path(self.model_path)
        if not model_file.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False
        logger.info(f"Model file: {self.model_path} ({model_file.stat().st_size / (1024*1024):.1f} MB)")

        command = [
            str(self.server_path),
            "-m", self.model_path,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "-c", "2048",
            "--verbose",
        ]

        if self.rpc_addresses:
            command.extend(["--rpc", self.rpc_addresses])

        if self.tensor_split:
            command.extend(["--tensor-split", self.tensor_split])

        # Always disable mmap for distributed inference (critical for Android/Termux)
        command.append("--no-mmap")

        env = os.environ.copy()
        if self.lib_path:
            env["LD_LIBRARY_PATH"] = self.lib_path
        
        # Enable detailed debug logging for RPC and llama.cpp
        env["GGML_RPC_DEBUG"] = "1"
        env["LLAMA_LOG_VERBOSITY"] = "0"
        env["LLAMA_LOG_TIMESTAMPS"] = "1"

        logger.info(f"Starting llama-server with distributed inference...")
        logger.info(f"Full command: {' '.join(command)}")

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff: 5s, 10s, 20s...
                    backoff = 5 * (2 ** (attempt - 1))
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries} (waiting {backoff}s)...")
                    time.sleep(backoff)
                    
                    # Re-check workers before retry
                    if self.rpc_addresses:
                        logger.info("Re-checking worker availability...")
                        if not wait_for_rpc_workers(self.rpc_addresses, timeout=60):
                            logger.warning("Workers still not ready, retrying anyway...")

                self.process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                self._start_log_threads()

                start_time = time.time()
                last_log_time = start_time
                
                while time.time() - start_time < DISTRIBUTED_SERVER_STARTUP_TIMEOUT:
                    if self.process.poll() is not None:
                        stderr_output = ""
                        if self.process.stderr:
                            stderr_output = self.process.stderr.read().decode()
                        logger.warning(f"llama-server died (attempt {attempt + 1}): {stderr_output[:500]}")
                        self.process = None
                        break

                    if self._is_healthy():
                        elapsed = int(time.time() - start_time)
                        logger.info("=" * 60)
                        logger.info(f"DISTRIBUTED LLAMA-SERVER READY (took {elapsed}s)")
                        logger.info(f"Listening on: http://127.0.0.1:{self.port}")
                        logger.info("=" * 60)
                        return True

                    # Log progress every 30 seconds
                    now = time.time()
                    if now - last_log_time >= 30:
                        elapsed = int(now - start_time)
                        logger.info(f"Still loading model... ({elapsed}s / {DISTRIBUTED_SERVER_STARTUP_TIMEOUT}s)")
                        last_log_time = now
                    
                    time.sleep(1)

                if self.process is not None:
                    elapsed = int(time.time() - start_time)
                    logger.warning(f"Server startup timeout after {elapsed}s (attempt {attempt + 1})")
                    self.stop()

            except Exception as error:
                logger.warning(f"Failed to start llama-server (attempt {attempt + 1}): {error}")
                import traceback
                logger.debug(traceback.format_exc())

        logger.error(
            f"Server failed to start after {max_retries} attempts. "
            "Check the logs above for specific errors."
        )
        return False

    def _is_healthy(self) -> bool:
        """Check if the server is responding and ready to serve requests.

        Require HTTP /health to return 200. Do NOT treat TCP-open or 503 as ready.
        """
        import requests

        try:
            response = requests.get(
                f"http://127.0.0.1:{self.port}/health",
                timeout=5,
            )
            status = response.status_code
            if status == 200:
                logger.info("Health check: Server is ready (200 OK)")
                return True
            if status == 503:
                # Log the actual response body to understand what's happening
                try:
                    body = response.json()
                    error_msg = body.get("error", {}).get("message", "Unknown")
                    logger.warning(f"Health check 503: {error_msg} (full: {body})")
                except Exception:
                    logger.warning(f"Health check 503: {response.text[:200]}")
                return False
            logger.debug(f"Server health check returned {status}")
            return False
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            logger.debug("Server health check timed out")
            return False
        except Exception as error:
            logger.debug(f"Server health check error: {error}")
            return False

    def _start_log_threads(self) -> None:
        """Stream stdout/stderr from llama-server for diagnostics."""
        if self.process is None:
            return

        def _reader(stream: Any, label: str) -> None:
            if stream is None:
                return
            for line in iter(stream.readline, b""):
                text = line.decode(errors="replace").strip()
                if text:
                    logger.info(f"[llama-server:{label}] {text}")

        threading.Thread(target=_reader, args=(self.process.stdout, "stdout"), daemon=True).start()
        threading.Thread(target=_reader, args=(self.process.stderr, "stderr"), daemon=True).start()

    def stop(self) -> None:
        """Stop the llama-server."""
        if self.process is None:
            return

        logger.info("Stopping distributed llama-server...")
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

        self.process = None


def initialize_llamacpp_distributed(
    bound_instance: BoundInstance,
) -> DistributedLlamaServer:
    """
    Initialize llama.cpp for distributed inference (master node only).

    This starts a llama-server with --rpc flag to connect to worker RPC servers.
    Returns the DistributedLlamaServer instance for making inference requests.
    """
    gguf_path = get_gguf_path_for_instance(bound_instance)
    rpc_addresses = build_rpc_address_list(bound_instance)
    tensor_split = build_tensor_split_string(bound_instance)

    logger.info(f"Initializing distributed llama.cpp: model={gguf_path}")
    logger.info(f"RPC workers: {rpc_addresses}")
    logger.info(f"Tensor split: {tensor_split}")

    server = DistributedLlamaServer(
        model_path=str(gguf_path),
        rpc_addresses=rpc_addresses,
        tensor_split=tensor_split,
    )

    if not server.start():
        raise RuntimeError("Failed to start distributed llama-server")

    return server

