import ctypes
import os
import resource
import sys

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.mlx.patches import apply_mlx_patches

logger: "loguru.Logger" = loguru.logger


# QoS classes from <sys/qos.h>. Apple's documented values for
# pthread_set_qos_class_self_np().
_QOS_CLASSES = {
    "user_interactive": 0x21,
    "user_initiated": 0x19,
    "default": 0x15,
    "utility": 0x11,
    "background": 0x09,
}


def _set_self_qos_class_darwin(qos_class_name: str) -> None:
    """Pin the calling thread's QoS class on macOS via pthread_set_qos_class_self_np.

    On a multi-runner Mac Studio, sibling Python processes can drift to
    different QoS classes under contention; macOS may then deprioritize
    one of them, producing a stratified throughput pattern. Pinning each
    runner to user_initiated at startup keeps them on equal footing.

    No-op on non-Darwin platforms or if the API is unavailable.
    """
    if sys.platform != "darwin":
        return
    qos_value = _QOS_CLASSES.get(qos_class_name.lower())
    if qos_value is None:
        logger.warning(
            f"Unknown EXO_RUNNER_QOS={qos_class_name!r}; "
            f"valid values: {sorted(_QOS_CLASSES.keys())}. Skipping QoS pin."
        )
        return
    try:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)
        # int pthread_set_qos_class_self_np(qos_class_t qos_class, int relative_priority)
        fn = libsystem.pthread_set_qos_class_self_np
        fn.argtypes = [ctypes.c_uint, ctypes.c_int]
        fn.restype = ctypes.c_int
        rc = int(fn(qos_value, 0))  # pyright: ignore[reportAny]
        if rc != 0:
            err = ctypes.get_errno()
            logger.warning(
                f"pthread_set_qos_class_self_np({qos_class_name}) failed: rc={rc} errno={err}"
            )
            return
        logger.info(f"Runner QoS class pinned to {qos_class_name}")
    except OSError as e:
        logger.warning(f"Could not load libSystem to set QoS class: {e}")
    except AttributeError:
        logger.warning(
            "pthread_set_qos_class_self_np unavailable on this libSystem; skipping QoS pin"
        )


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

    # Pin QoS so siblings on the same machine don't drift apart under contention.
    # Defaults to user_initiated on Darwin; set EXO_RUNNER_QOS=off to disable.
    qos_choice = os.environ.get("EXO_RUNNER_QOS", "user_initiated")
    if qos_choice.lower() != "off":
        _set_self_qos_class_darwin(qos_choice)

    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override == "false":
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        if bound_instance.is_image_model:
            from exo.worker.runner.image_models.runner import Runner as ImageRunner

            runner = ImageRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        else:
            from exo.worker.runner.llm_inference.runner import Runner

            apply_mlx_patches()

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
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
