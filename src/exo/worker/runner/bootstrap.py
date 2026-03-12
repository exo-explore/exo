import ctypes
import ctypes.util
import os
import resource
import sys
from pathlib import Path

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, VllmInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender

logger: "loguru.Logger" = loguru.logger

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
                    logger.info(f"Loaded CUDA host lib: {lib_path}")
                except OSError as e:
                    logger.warning(f"Failed to load {lib_path}: {e}")
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
    if fast_synch_override != "off":
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        if isinstance(bound_instance.instance, VllmInstance):
            _ensure_cuda_libs()
            from exo.worker.runner.vllm_inference.runner import Runner as VllmRunner

            runner = VllmRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        elif bound_instance.is_image_model:
            from exo.worker.runner.image_models.runner import Runner as ImageRunner

            runner = ImageRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        else:
            from exo.worker.runner.llm_inference.runner import Runner

            runner = Runner(
                bound_instance, event_sender, task_receiver, cancel_receiver
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
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None and torch_mod.distributed.is_initialized():
                torch_mod.distributed.destroy_process_group()
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
