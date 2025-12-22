"""
RPC Server Manager for distributed llama.cpp inference.

Manages an rpc-server process on worker nodes (device_rank > 0) to enable
distributed tensor operations across multiple devices.

The master node (device_rank == 0) connects to worker RPC servers via the
--rpc flag and distributes layers using --tensor-split.
"""

import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Final

from loguru import logger


RPC_SERVER_STARTUP_TIMEOUT: Final[int] = 30
DEFAULT_RPC_PORT: Final[int] = 50052
RPC_BASE_PORT: Final[int] = 50052


def find_rpc_server() -> Path | None:
    """Find the rpc-server binary."""
    search_paths = [
        Path.home() / "llama.cpp" / "build" / "bin" / "rpc-server",
        Path("/usr/local/bin/rpc-server"),
        Path("/usr/bin/rpc-server"),
    ]

    for path in search_paths:
        if path.exists() and os.access(path, os.X_OK):
            return path

    return None


def get_lib_path() -> str:
    """Get LD_LIBRARY_PATH for llama.cpp libraries."""
    lib_dirs = [
        Path.home() / "llama.cpp" / "build" / "bin",
        Path.home() / "llama.cpp" / "build" / "lib",
    ]
    return ":".join(str(directory) for directory in lib_dirs if directory.exists())


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def is_rpc_server_responding(port: int, host: str = "127.0.0.1") -> bool:
    """Check if RPC server is responding on the given port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        try:
            sock.connect((host, port))
            return True
        except (OSError, socket.timeout):
            return False


class RpcServerManager:
    """
    Manages an rpc-server instance for distributed inference.

    Worker nodes (device_rank > 0) run this to expose their compute
    to the master node via TCP.
    """

    _instance: "RpcServerManager | None" = None

    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.port: int = DEFAULT_RPC_PORT
        self.host: str = "0.0.0.0"
        self.server_path: Path | None = find_rpc_server()
        self.lib_path: str = get_lib_path()

        if self.server_path is None:
            logger.warning(
                "rpc-server not found. Distributed inference will not work. "
                "Build with: cd ~/llama.cpp && cmake -B build -DGGML_RPC=ON && "
                "cmake --build build --target rpc-server"
            )

    @classmethod
    def get_instance(cls) -> "RpcServerManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_running(self) -> bool:
        """Check if RPC server is running and responding."""
        if self.process is None:
            return False

        if self.process.poll() is not None:
            self.process = None
            return False

        return is_rpc_server_responding(self.port, "127.0.0.1")

    def start(self, port: int, host: str = "0.0.0.0") -> bool:
        """
        Start the rpc-server on the specified port.

        Returns True if server started successfully.
        """
        if self.server_path is None:
            logger.error("rpc-server binary not found")
            return False

        self.port = port
        self.host = host

        if self.is_running():
            logger.info(f"RPC server already running on port {self.port}")
            return True

        if self.process is not None:
            logger.info("Stopping existing RPC server...")
            self.stop()

        if not is_port_available(port, host):
            logger.warning(f"Port {port} not available, attempting to free it...")
            self._kill_existing_server()
            time.sleep(1)

            if not is_port_available(port, host):
                logger.error(f"Port {port} still not available")
                return False

        command = [
            str(self.server_path),
            "--host", host,
            "--port", str(port),
        ]

        env = os.environ.copy()
        if self.lib_path:
            env["LD_LIBRARY_PATH"] = self.lib_path

        logger.info(f"Starting rpc-server: {' '.join(command)}")

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            start_time = time.time()
            while time.time() - start_time < RPC_SERVER_STARTUP_TIMEOUT:
                if self.process.poll() is not None:
                    stderr_output = ""
                    if self.process.stderr:
                        stderr_output = self.process.stderr.read().decode()
                    logger.error(f"rpc-server died during startup: {stderr_output[:500]}")
                    self.process = None
                    return False

                if is_rpc_server_responding(port, "127.0.0.1"):
                    logger.info(f"rpc-server started successfully on {host}:{port}")
                    # Also log the external IPs this server should be reachable on
                    self._log_network_info(port)
                    return True

                time.sleep(0.5)
                logger.debug(f"Waiting for rpc-server... ({int(time.time() - start_time)}s)")

            logger.error(f"rpc-server failed to start within {RPC_SERVER_STARTUP_TIMEOUT}s")
            self.stop()
            return False

        except Exception as error:
            logger.error(f"Failed to start rpc-server: {error}")
            return False

    def stop(self) -> None:
        """Stop the rpc-server."""
        if self.process is None:
            return

        logger.info("Stopping rpc-server...")

        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("rpc-server didn't terminate gracefully, killing...")
            self.process.kill()
            self.process.wait()

        self.process = None
        logger.info("rpc-server stopped")

    def _kill_existing_server(self) -> None:
        """Try to kill any existing rpc-server on the port."""
        try:
            os.system(f"pkill -f 'rpc-server.*--port {self.port}'")
        except Exception:
            pass

    def _log_network_info(self, port: int) -> None:
        """Log network interface information for debugging connectivity."""
        try:
            import socket
            hostname = socket.gethostname()
            # Get all IP addresses for this host
            try:
                addrs = socket.getaddrinfo(hostname, None, socket.AF_INET)
                ips = list(set(addr[4][0] for addr in addrs))
                logger.info(f"RPC server reachable on: {', '.join(f'{ip}:{port}' for ip in ips)}")
            except socket.gaierror:
                pass

            # Also try to get interface IPs via netifaces-like approach
            try:
                import subprocess
                result = subprocess.run(
                    ["ip", "-4", "addr", "show"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    import re
                    ips = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', result.stdout)
                    external_ips = [ip for ip in ips if not ip.startswith('127.')]
                    if external_ips:
                        logger.info(f"External IPs: {', '.join(external_ips)}")
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Failed to get network info: {e}")


def assign_rpc_port(device_rank: int) -> int:
    """
    Assign an RPC port for a device based on its rank.

    Master (rank 0) doesn't need an RPC port.
    Workers (rank > 0) get sequential ports starting from RPC_BASE_PORT.
    """
    if device_rank == 0:
        return 0
    return RPC_BASE_PORT + device_rank - 1

