"""
Utility functions for the llama.cpp engine.
"""

import os
import sys
from pathlib import Path
from typing import Any

from exo.shared.types.worker.instances import BoundInstance
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.llamacpp.constants import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_N_BATCH,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_N_THREADS,
)
from exo.worker.runner.bootstrap import logger


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

