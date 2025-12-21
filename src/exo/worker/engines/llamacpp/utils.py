"""
Utility functions for the llama.cpp engine.
"""

import os
from pathlib import Path

from llama_cpp import Llama

from exo.shared.types.worker.instances import BoundInstance
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.llamacpp.constants import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_N_BATCH,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_N_THREADS,
)
from exo.worker.runner.bootstrap import logger


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


def initialize_llamacpp(
    bound_instance: BoundInstance,
) -> Llama:
    """
    Initialize a llama.cpp model for inference.
    Returns the loaded Llama model instance.
    """
    shard_metadata = bound_instance.bound_shard
    model_id = str(shard_metadata.model_meta.model_id)

    logger.info(f"Initializing llama.cpp for model: {model_id}")

    # Build the model path
    model_path = build_model_path(model_id)

    # Find the GGUF file
    gguf_path = find_gguf_file(model_path)

    if gguf_path is None:
        raise FileNotFoundError(
            f"No GGUF file found in {model_path}. "
            "Please ensure you're using a GGUF-format model for llama.cpp."
        )

    logger.info(f"Loading GGUF model from: {gguf_path}")

    # Configure threading based on environment or auto-detect
    n_threads = int(os.environ.get("LLAMA_N_THREADS", DEFAULT_N_THREADS))
    if n_threads == 0:
        n_threads = os.cpu_count() or 4

    n_gpu_layers = int(os.environ.get("LLAMA_N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS))
    n_ctx = int(os.environ.get("LLAMA_N_CTX", DEFAULT_CONTEXT_SIZE))
    n_batch = int(os.environ.get("LLAMA_N_BATCH", DEFAULT_N_BATCH))

    logger.info(
        f"llama.cpp config: threads={n_threads}, gpu_layers={n_gpu_layers}, "
        f"ctx={n_ctx}, batch={n_batch}"
    )

    # Load the model - verbose=True to see loading progress
    logger.info(f">>> STARTING Llama() constructor for {gguf_path}")
    logger.info(f">>> File size: {gguf_path.stat().st_size / (1024*1024):.2f} MB")
    
    import time
    load_start = time.time()
    
    # Disable mmap on Android/Termux - can cause hangs
    import sys
    use_mmap = "termux" not in sys.prefix.lower() and "android" not in sys.prefix.lower()
    logger.info(f">>> use_mmap={use_mmap} (prefix={sys.prefix})")
    
    model = Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        use_mmap=use_mmap,  # Disable mmap on Android
        verbose=True,
    )
    
    load_time = time.time() - load_start
    logger.info(f">>> FINISHED Llama() constructor in {load_time:.2f} seconds")

    return model


def get_model_info(model: Llama) -> dict[str, object]:
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

