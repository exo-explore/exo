"""
Utility functions for the llama.cpp engine.
"""

import os
import subprocess
import sys
import time
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


DISTRIBUTED_SERVER_STARTUP_TIMEOUT: Final[int] = 120


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


def build_rpc_address_list(bound_instance: BoundInstance) -> str:
    """
    Build the RPC address list for llama.cpp --rpc flag.

    Format: "ip1:port1,ip2:port2,..."
    Only includes worker nodes (device_rank > 0).
    """
    instance = bound_instance.instance
    if not isinstance(instance, LlamaCppInstance):
        return ""

    shard_assignments = instance.shard_assignments
    rpc_addresses: list[str] = []

    for node_id, runner_id in shard_assignments.node_to_runner.items():
        shard = shard_assignments.runner_to_shard.get(runner_id)
        if shard is None:
            continue

        if shard.device_rank == 0:
            continue

        rpc_port = instance.rpc_ports.get(node_id, 0)
        if rpc_port == 0:
            continue

        for host in instance.hosts:
            if host.ip and host.ip != "0.0.0.0":
                rpc_addresses.append(f"{host.ip}:{rpc_port}")
                break

    return ",".join(rpc_addresses)


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


def get_lib_path() -> str:
    """Get LD_LIBRARY_PATH for llama.cpp libraries."""
    lib_dirs = [
        Path.home() / "llama.cpp" / "build" / "bin",
        Path.home() / "llama.cpp" / "build" / "lib",
    ]
    return ":".join(str(directory) for directory in lib_dirs if directory.exists())


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

    def start(self) -> bool:
        """Start the distributed llama-server."""
        if self.server_path is None:
            logger.error("llama-server not found")
            return False

        command = [
            str(self.server_path),
            "-m", self.model_path,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "-c", str(DEFAULT_CONTEXT_SIZE),
            "-t", str(os.cpu_count() or 4),
        ]

        if self.rpc_addresses:
            command.extend(["--rpc", self.rpc_addresses])

        if self.tensor_split:
            command.extend(["--tensor-split", self.tensor_split])

        if is_android():
            command.append("--no-mmap")

        env = os.environ.copy()
        if self.lib_path:
            env["LD_LIBRARY_PATH"] = self.lib_path

        logger.info(f"Starting distributed llama-server: {' '.join(command[:10])}...")
        logger.info(f"RPC addresses: {self.rpc_addresses}")
        logger.info(f"Tensor split: {self.tensor_split}")

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            start_time = time.time()
            while time.time() - start_time < DISTRIBUTED_SERVER_STARTUP_TIMEOUT:
                if self.process.poll() is not None:
                    stderr_output = ""
                    if self.process.stderr:
                        stderr_output = self.process.stderr.read().decode()
                    logger.error(f"Distributed llama-server died: {stderr_output[:500]}")
                    self.process = None
                    return False

                if self._is_healthy():
                    logger.info(f"Distributed llama-server started on port {self.port}")
                    return True

                time.sleep(1)
                logger.debug(f"Waiting for server... ({int(time.time() - start_time)}s)")

            logger.error(f"Server failed to start within {DISTRIBUTED_SERVER_STARTUP_TIMEOUT}s")
            self.stop()
            return False

        except Exception as error:
            logger.error(f"Failed to start distributed llama-server: {error}")
            return False

    def _is_healthy(self) -> bool:
        """Check if the server is responding."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            try:
                sock.connect(("127.0.0.1", self.port))
                return True
            except (OSError, socket.timeout):
                return False

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

