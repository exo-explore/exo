import sys
import time
from typing import TYPE_CHECKING, Any

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ChatCompletion,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
from exo.shared.types.worker.instances import (
    BoundInstance,
    LlamaCppInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerStatus,
    RunnerWaitingForModel,
    RunnerWarmingUp,
)
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import mlx.core as mx
    from llama_cpp import Llama
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from exo.shared.types.tasks import ChatCompletionTaskParams
    from exo.worker.engines.mlx import Model


def is_mlx_instance(instance: MlxRingInstance | MlxJacclInstance | LlamaCppInstance) -> bool:
    """Check if the instance uses MLX backend."""
    return isinstance(instance, (MlxRingInstance, MlxJacclInstance))


def is_llamacpp_instance(instance: MlxRingInstance | MlxJacclInstance | LlamaCppInstance) -> bool:
    """Check if the instance uses llama.cpp backend."""
    return isinstance(instance, LlamaCppInstance)


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
) -> None:
    instance = bound_instance.instance
    runner_id = bound_instance.bound_runner_id
    shard_metadata = bound_instance.bound_shard

    try:
        logger.info("hello from the runner")
        if getattr(shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        setup_start_time = time.time()

        # Determine which backend to use
        use_llamacpp = is_llamacpp_instance(instance)

        if use_llamacpp:
            logger.info("Using llama.cpp backend")
            _run_llamacpp_runner(
                bound_instance, event_sender, task_receiver, runner_id, shard_metadata, setup_start_time
            )
        else:
            logger.info("Using MLX backend")
            _run_mlx_runner(
                bound_instance, event_sender, task_receiver, runner_id, shard_metadata, setup_start_time
            )

    except ClosedResourceError:
        logger.warning("runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(f"Runner {runner_id} crashed with critical exception {e}")
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        event_sender.close()
        task_receiver.close()
        event_sender.join()
        task_receiver.join()
        logger.info("bye from the runner")


def _run_mlx_runner(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    runner_id: Any,
    shard_metadata: Any,
    setup_start_time: float,
) -> None:
    """Run the MLX-based inference loop."""
    from exo.worker.engines.mlx.generator.generate import (
        mlx_generate,
        warmup_inference as mlx_warmup,
    )
    from exo.worker.engines.mlx.utils_mlx import initialize_mlx, mlx_force_oom

    model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    sampler: Callable[[mx.array], mx.array] | None = None

    current_status: RunnerStatus = RunnerWaitingForModel()
    logger.info("runner waiting for model")
    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

    with task_receiver as tasks:
        for task in tasks:
            event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running))
            event_sender.send(TaskAcknowledged(task_id=task.task_id))

            match task:
                case LoadModel() if isinstance(current_status, (RunnerWaitingForModel, RunnerFailed)):
                    current_status = RunnerLoading()
                    logger.info("runner loading")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                    model, tokenizer, sampler = initialize_mlx(bound_instance)

                    current_status = RunnerLoaded()
                    logger.info("runner loaded")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    assert model is not None
                    assert tokenizer is not None
                    assert sampler is not None
                    current_status = RunnerWarmingUp()
                    logger.info("runner warming up")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                    logger.info(f"warming up inference for instance: {bound_instance.instance}")
                    toks = mlx_warmup(model=model, tokenizer=tokenizer, sampler=sampler)
                    logger.info(f"warmed up by generating {toks} tokens")
                    logger.info(f"runner initialized in {time.time() - setup_start_time} seconds")
                    current_status = RunnerReady()
                    logger.info("runner ready")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerReady()))

                case ChatCompletion(task_params=task_params, command_id=command_id) if isinstance(
                    current_status, RunnerReady
                ):
                    assert model is not None
                    assert tokenizer is not None
                    assert sampler is not None
                    logger.info(f"received chat request: {str(task)[:500]}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                    assert task_params.messages[0].content is not None
                    _check_for_debug_prompts_mlx(task_params.messages[0].content, mlx_force_oom)

                    for response in mlx_generate(
                        model=model, tokenizer=tokenizer, sampler=sampler, task=task_params
                    ):
                        if isinstance(response, GenerationResponse) and shard_metadata.device_rank == 0:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=TokenChunk(
                                        idx=response.token,
                                        model=shard_metadata.model_meta.model_id,
                                        text=response.text,
                                        token_id=response.token,
                                        finish_reason=response.finish_reason,
                                    ),
                                )
                            )

                    current_status = RunnerReady()
                    logger.info("runner ready")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerReady()))

                case Shutdown():
                    logger.info("runner shutting down")
                    event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete))
                    break

                case _:
                    raise ValueError("Received task outside of state machine")

            event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete))

    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown()))


def _run_llamacpp_runner(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    runner_id: Any,
    shard_metadata: Any,
    setup_start_time: float,
) -> None:
    """Run the llama.cpp-based inference loop."""
    from exo.worker.engines.llamacpp.utils import (
        initialize_llamacpp,
        use_native_cli,
        get_gguf_path_for_instance,
    )

    model: Any = None
    is_native = use_native_cli()

    current_status: RunnerStatus = RunnerWaitingForModel()
    logger.info("runner waiting for model")
    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

    with task_receiver as tasks:
        for task in tasks:
            event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running))
            event_sender.send(TaskAcknowledged(task_id=task.task_id))

            match task:
                case LoadModel() if isinstance(current_status, (RunnerWaitingForModel, RunnerFailed)):
                    current_status = RunnerLoading()
                    logger.info("runner loading")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                    model = initialize_llamacpp(bound_instance)

                    current_status = RunnerLoaded()
                    logger.info("runner loaded")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    assert model is not None
                    current_status = RunnerWarmingUp()
                    logger.info("runner warming up")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                    logger.info(f"warming up inference for instance: {bound_instance.instance}")
                    
                    if is_native:
                        from exo.worker.engines.llamacpp.native_cli import native_warmup
                        gguf_path = get_gguf_path_for_instance(bound_instance)
                        toks = native_warmup(str(gguf_path))
                    else:
                        from exo.worker.engines.llamacpp.generate import warmup_inference
                        toks = warmup_inference(model=model)
                    
                    logger.info(f"warmed up by generating {toks} tokens")
                    logger.info(f"runner initialized in {time.time() - setup_start_time} seconds")
                    current_status = RunnerReady()
                    logger.info("runner ready")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerReady()))

                case ChatCompletion(task_params=task_params, command_id=command_id) if isinstance(
                    current_status, RunnerReady
                ):
                    assert model is not None
                    logger.info(f"received chat request: {str(task)[:500]}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                    assert task_params.messages[0].content is not None
                    _check_for_debug_prompts_llamacpp(task_params.messages[0].content)

                    if is_native:
                        from exo.worker.engines.llamacpp.native_cli import native_generate
                        gguf_path = get_gguf_path_for_instance(bound_instance)
                        generator = native_generate(str(gguf_path), task_params)
                    else:
                        from exo.worker.engines.llamacpp.generate import llamacpp_generate
                        generator = llamacpp_generate(model=model, task=task_params)

                    for response in generator:
                        if isinstance(response, GenerationResponse) and shard_metadata.device_rank == 0:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=TokenChunk(
                                        idx=response.token,
                                        model=shard_metadata.model_meta.model_id,
                                        text=response.text,
                                        token_id=response.token,
                                        finish_reason=response.finish_reason,
                                    ),
                                )
                            )

                    current_status = RunnerReady()
                    logger.info("runner ready")
                    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerReady()))

                case Shutdown():
                    logger.info("runner shutting down")
                    event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete))
                    break

                case _:
                    raise ValueError("Received task outside of state machine")

            event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete))

    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown()))


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts_mlx(
    prompt: str | ChatCompletionMessageText | list[ChatCompletionMessageText],
    force_oom_fn: Any,
) -> None:
    """Check for debug prompts (MLX version with OOM support)."""
    text = _extract_prompt_text(prompt)
    if text is None:
        return

    if EXO_RUNNER_MUST_FAIL in text:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in text:
        force_oom_fn()
    if EXO_RUNNER_MUST_TIMEOUT in text:
        time.sleep(100)


def _check_for_debug_prompts_llamacpp(
    prompt: str | ChatCompletionMessageText | list[ChatCompletionMessageText],
) -> None:
    """Check for debug prompts (llama.cpp version)."""
    text = _extract_prompt_text(prompt)
    if text is None:
        return

    if EXO_RUNNER_MUST_FAIL in text:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in text:
        logger.warning("OOM simulation not supported on llama.cpp")
    if EXO_RUNNER_MUST_TIMEOUT in text:
        time.sleep(100)


def _extract_prompt_text(
    prompt: str | ChatCompletionMessageText | list[ChatCompletionMessageText],
) -> str | None:
    """Extract text from various prompt formats."""
    if isinstance(prompt, list):
        if len(prompt) == 0:
            logger.debug("Empty message prompt received in debug prompt")
            return None
        prompt = prompt[0]

    if isinstance(prompt, ChatCompletionMessageText):
        return prompt.text

    return prompt
