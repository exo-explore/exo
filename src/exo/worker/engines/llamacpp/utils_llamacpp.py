"""Utilities for the llama-cpp-python inference backend.

This module handles model discovery/download, RPC server binary location,
and GPU-layer split calculation for distributed inference.
"""

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from exo.shared.constants import EXO_DEFAULT_MODELS_DIR
from exo.shared.types.worker.instances import LlamaCppRpcInstance
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import ShardMetadata

if TYPE_CHECKING:
    from llama_cpp import Llama


def find_gguf_model_path(shard: ShardMetadata) -> Path | None:
    """Return the local path to the GGUF file for this shard's model card.

    Searches ``EXO_DEFAULT_MODELS_DIR`` for a directory whose name encodes the
    gguf_repo_id (hyphens replacing slashes) and contains the expected filename.
    Returns ``None`` when the file has not been downloaded yet.
    """
    card = shard.model_card
    if card.gguf_repo_id is None or card.gguf_filename is None:
        return None

    # Convention: repo stored as owner--reponame (mirrors the MLX naming)
    dir_name = card.gguf_repo_id.replace("/", "--")
    candidate = EXO_DEFAULT_MODELS_DIR / dir_name / card.gguf_filename
    if candidate.exists():
        return candidate
    return None


def find_rpc_server_binary() -> str | None:
    """Locate the ``llama-rpc-server`` binary.

    Checks, in order:
    1. ``PATH`` (e.g. when llama.cpp is installed system-wide)
    2. The ``bin/`` directory next to the installed ``llama_cpp`` package
       (populated when building llama-cpp-python from source with
       ``LLAMA_BUILD_SERVER=on``)
    """
    binary_names = ["llama-rpc-server", "rpc-server"]

    # 1. System PATH
    for name in binary_names:
        found = shutil.which(name)
        if found is not None:
            return found

    # 2. Alongside the llama_cpp package
    try:
        import llama_cpp

        pkg_dir = Path(llama_cpp.__file__).parent
        candidates = [
            pkg_dir / name
            for name in binary_names
        ] + [
            pkg_dir / "bin" / name
            for name in binary_names
        ] + [
            pkg_dir.parent / "bin" / name
            for name in binary_names
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    except ImportError:
        pass

    return None


def start_rpc_server(port: int) -> subprocess.Popen[bytes]:
    """Spawn a ``llama-rpc-server`` subprocess on *port*.

    Raises ``RuntimeError`` if the binary cannot be found.
    """
    binary = find_rpc_server_binary()
    if binary is None:
        raise RuntimeError(
            "llama-rpc-server binary not found. "
            "Install llama.cpp from source with LLAMA_BUILD_SERVER=on, "
            "or build llama-cpp-python from source."
        )
    logger.info(f"Starting llama-rpc-server on port {port} ({binary})")
    proc = subprocess.Popen(
        [binary, "--host", "0.0.0.0", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def calculate_gpu_layer_split(
    instance: LlamaCppRpcInstance,
    bound_runner_id: RunnerId,
    shard: ShardMetadata,
) -> int:
    """Return the number of GPU layers this runner should hold locally.

    For rank 0 in a multi-node setup:
    - Equal split between rank 0 and RPC workers (simple initial heuristic).
    For all other ranks:
    - 0 (they serve as RPC workers and hold layers remotely from rank 0's
      perspective; the actual GPU loading is handled by the RPC server binary).

    The caller (rank 0) passes the total to ``Llama(n_gpu_layers=...)`` while
    the RPC workers are listed in ``rpc_servers``.
    """
    if instance.n_gpu_layers_per_runner:
        # Use master-computed split if available
        val = instance.n_gpu_layers_per_runner.get(bound_runner_id)
        if val is not None:
            return val

    # Fallback: rank 0 gets all layers (safe for single-device or when RPC
    # workers handle their own layers independently)
    if shard.device_rank == 0:
        return shard.n_layers  # request full offload; driver will cap to VRAM
    return 0


def load_llamacpp_model(
    model_path: Path,
    instance: LlamaCppRpcInstance,
    bound_runner_id: RunnerId,
    shard: ShardMetadata,
) -> "Llama":
    """Load the model via llama-cpp-python.

    Rank 0 connects to any RPC servers declared in ``instance.rpc_addresses``
    and requests GPU offloading according to the pre-computed layer split.
    Other ranks return ``None`` because their GPU compute is provided by the
    ``llama-rpc-server`` subprocess started during ``ConnectToGroup``.
    """
    from llama_cpp import Llama

    n_gpu_layers = calculate_gpu_layer_split(instance, bound_runner_id, shard)

    rpc_servers_str: str | None = None
    if shard.device_rank == 0 and instance.rpc_addresses:
        rpc_servers_str = ",".join(instance.rpc_addresses.values())
        logger.info(f"Connecting to RPC servers: {rpc_servers_str}")

    logger.info(
        f"Loading GGUF model from {model_path} "
        f"(n_gpu_layers={n_gpu_layers}, rpc_servers={rpc_servers_str!r})"
    )

    kwargs: dict[str, object] = {
        "model_path": str(model_path),
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": 4096,
        "verbose": False,
    }
    if rpc_servers_str:
        kwargs["rpc_servers"] = rpc_servers_str

    return Llama(**kwargs)  # type: ignore[arg-type]
