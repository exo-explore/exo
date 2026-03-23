import ctypes
import os
import resource
import sys
from pathlib import Path

import loguru
from exo_core.constants import EXO_MODELS_DIR
from exo_core.types.instances import BoundInstance, VllmInstance
from exo_core.types.runners import RunnerFailed
from exo_core.types.tasks import Task, TaskId

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender

_CUDA_HOST_LIBS = ["libcuda.so.1", "libnvidia-ml.so.1", "libnvidia-ptxjitcompiler.so.1"]
_CUDA_HOST_SEARCH_DIRS = [
    Path("/usr/lib/aarch64-linux-gnu"),
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/usr/lib64"),
    Path("/usr/lib"),
    Path("/usr/local/cuda/lib64"),
    Path("/usr/local/cuda/compat"),
]


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
                    loguru.logger.info(f"Loaded CUDA host lib: {lib_path}")
                except OSError:
                    loguru.logger.warning(f"Failed to load {lib_path}")
                    raise
        return


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    loguru.logger = _logger
    logger = loguru.logger

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override != "off":
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        if isinstance(bound_instance.instance, VllmInstance):
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            os.environ["VLLM_KV_CACHE_LAYOUT"] = "NHD"
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
            _ensure_cuda_libs()

            from vllm_engine.builder import VllmBuilder
            from .llm_inference.runner import Runner

            builder = VllmBuilder.create(
                bound_instance=bound_instance,
                cancel_receiver=cancel_receiver,
                event_sender=event_sender,
            )
            runner = Runner(
                bound_instance, event_sender, task_receiver, cancel_receiver, builder
            )
            runner.main()
        elif bound_instance.is_image_model:
            from .image_models.runner import Runner

            runner = Runner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        else:
            from .llm_inference.runner import Runner
            from mlx_engine.builder import MlxBuilder

            builder = MlxBuilder.create(
                bound_instance,
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
