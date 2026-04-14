import ctypes
import os
import resource
import sys
import urllib.request
from pathlib import Path

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, VllmInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.mlx.patches import apply_mlx_patches

logger: "loguru.Logger" = loguru.logger

_TIKTOKEN_BASE_URL = "https://openaipublic.blob.core.windows.net/encodings"
_TIKTOKEN_FILES = ["o200k_base.tiktoken", "cl100k_base.tiktoken"]

_CUDA_HOST_LIBS = ["libcuda.so.1", "libnvidia-ml.so.1", "libnvidia-ptxjitcompiler.so.1"]
_CUDA_HOST_SEARCH_DIRS = [
    Path("/usr/lib/aarch64-linux-gnu"),
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/usr/lib64"),
    Path("/usr/lib"),
    Path("/usr/local/cuda/lib64"),
    Path("/usr/local/cuda/compat"),
]


def _ensure_tiktoken_encodings() -> None:
    if os.environ.get("TIKTOKEN_ENCODINGS_BASE"):
        return
    from exo.shared.constants import EXO_CACHE_HOME

    enc_dir = EXO_CACHE_HOME / "encodings"
    enc_dir.mkdir(parents=True, exist_ok=True)
    for fname in _TIKTOKEN_FILES:
        dest = enc_dir / fname
        if dest.exists():
            continue
        url = f"{_TIKTOKEN_BASE_URL}/{fname}"
        logger.info(f"Downloading {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception:
            logger.warning(f"Failed to download {fname}, harmony encoding may fail")
            return
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(enc_dir)
    logger.info(f"Set TIKTOKEN_ENCODINGS_BASE={enc_dir}")


def _ensure_cuda_libs() -> None:
    if sys.platform != "linux":
        return
    for search_dir in _CUDA_HOST_SEARCH_DIRS:
        driver = search_dir / "libcuda.so.1"
        if not driver.exists():
            continue
        for lib_name in _CUDA_HOST_LIBS:
            lib_path = search_dir / lib_name
            if lib_path.exists():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                    logger.info(f"Loaded CUDA host lib: {lib_path}")
                except OSError:
                    logger.warning(f"Failed to load {lib_path}")
                    raise
        return


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override == "false":
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        if isinstance(bound_instance.instance, VllmInstance):
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            os.environ["VLLM_KV_CACHE_LAYOUT"] = "NHD"
            if "/usr/local/cuda-13.2/compat" not in os.environ.get("LD_LIBRARY_PATH", ""):
                _ensure_cuda_libs()
            _ensure_tiktoken_encodings()
            from exo.worker.runner.llm_inference.runner import Runner, VllmBuilder

            model_id = bound_instance.bound_shard.model_card.model_id
            builder = VllmBuilder(
                model_id=model_id,
                trust_remote_code=bound_instance.bound_shard.model_card.trust_remote_code,
                cancel_receiver=cancel_receiver,
                event_sender=event_sender,
            )
            runner = Runner(
                bound_instance, event_sender, task_receiver, cancel_receiver, builder
            )
            runner.main()
        elif bound_instance.is_image_model:
            from exo.worker.runner.image_models.runner import Runner as ImageRunner

            runner = ImageRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        else:
            from exo.worker.runner.llm_inference.runner import MlxBuilder, Runner
            apply_mlx_patches()

            builder = MlxBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender,
                cancel_receiver=cancel_receiver,
            )

            runner = Runner(
                bound_instance, event_sender, task_receiver, cancel_receiver, builder
            )
            runner.main()

    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
