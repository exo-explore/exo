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
        get_gguf_path_for_instance,
        initialize_llamacpp,
        initialize_llamacpp_distributed,
        is_distributed_instance,
        use_native_cli,
        use_server_mode,
    )

    device_rank = shard_metadata.device_rank
    world_size = shard_metadata.world_size
    is_distributed = is_distributed_instance(bound_instance)

    if is_distributed and world_size > 1:
        if device_rank > 0:
            _run_llamacpp_worker(
                bound_instance, event_sender, task_receiver, runner_id, shard_metadata, setup_start_time
            )
        else:
            _run_llamacpp_master(
                bound_instance, event_sender, task_receiver, runner_id, shard_metadata, setup_start_time
            )
        return

    model: Any = None
    is_native = use_native_cli()
    is_server = use_server_mode()
    
    if is_server:
        logger.info("Using llama-server HTTP mode (most reliable on Android)")

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
                    
                    gguf_path = get_gguf_path_for_instance(bound_instance)
                    
                    if is_server:
                        from exo.worker.engines.llamacpp.llama_server import server_warmup
                        logger.info("Using llama-server for warmup (starts server and loads model)")
                        toks = server_warmup(str(gguf_path))
                    elif is_native:
                        from exo.worker.engines.llamacpp.native_cli import native_warmup
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

                    gguf_path = get_gguf_path_for_instance(bound_instance)
                    
                    if is_server:
                        from exo.worker.engines.llamacpp.llama_server import server_generate
                        logger.info(f"Starting server_generate with path={gguf_path}")
                        generator = server_generate(str(gguf_path), task_params)
                    elif is_native:
                        from exo.worker.engines.llamacpp.native_cli import native_generate
                        logger.info(f"Starting native_generate with path={gguf_path}")
                        generator = native_generate(str(gguf_path), task_params)
                    else:
                        from exo.worker.engines.llamacpp.generate import llamacpp_generate
                        generator = llamacpp_generate(model=model, task=task_params)

                    logger.info("Starting generator iteration...")
                    response_count = 0
                    for response in generator:
                        response_count += 1
                        logger.info(f"Runner got response #{response_count}: text='{response.text[:50] if response.text else ''}', finish={response.finish_reason}")
                        if isinstance(response, GenerationResponse) and shard_metadata.device_rank == 0:
                            logger.info(f"Sending ChunkGenerated event for command {command_id}")
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

                    logger.info(f"Generator loop complete, processed {response_count} responses")
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


def _run_llamacpp_worker(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    runner_id: Any,
    shard_metadata: Any,
    setup_start_time: float,
) -> None:
    """
    Run as a distributed worker node (device_rank > 0).

    Workers run an RPC server that the master node connects to for
    distributed tensor operations. They don't load the model themselves.
    """
    from exo.worker.engines.llamacpp.rpc_server import RpcServerManager

    instance = bound_instance.instance
    if not isinstance(instance, LlamaCppInstance):
        raise ValueError("Expected LlamaCppInstance for distributed worker")

    rpc_port = instance.rpc_ports.get(bound_instance.bound_node_id, 0)
    if rpc_port == 0:
        raise ValueError(f"No RPC port assigned for worker node {bound_instance.bound_node_id}")

    rpc_manager: RpcServerManager | None = None

    current_status: RunnerStatus = RunnerWaitingForModel()
    logger.info(f"Worker node (rank {shard_metadata.device_rank}) waiting for model")
    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

    try:
        with task_receiver as tasks:
            for task in tasks:
                event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running))
                event_sender.send(TaskAcknowledged(task_id=task.task_id))

                match task:
                    case LoadModel() if isinstance(current_status, (RunnerWaitingForModel, RunnerFailed)):
                        current_status = RunnerLoading()
                        logger.info(f"Worker starting RPC server on port {rpc_port}")
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                        rpc_manager = RpcServerManager.get_instance()
                        if not rpc_manager.start(port=rpc_port):
                            raise RuntimeError(f"Failed to start RPC server on port {rpc_port}")

                        current_status = RunnerLoaded()
                        logger.info(f"Worker RPC server running on port {rpc_port}")
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                    case StartWarmup() if isinstance(current_status, RunnerLoaded):
                        current_status = RunnerWarmingUp()
                        logger.info("Worker warming up (RPC server ready)")
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                        logger.info(f"Worker initialized in {time.time() - setup_start_time} seconds")
                        current_status = RunnerReady()
                        logger.info("Worker ready (RPC server accepting connections)")
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerReady()))

                    case ChatCompletion() if isinstance(current_status, RunnerReady):
                        current_status = RunnerRunning()
                        logger.info("Worker processing distributed inference request")
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                        current_status = RunnerReady()
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerReady()))

                    case Shutdown():
                        logger.info("Worker shutting down")
                        event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete))
                        break

                    case _:
                        raise ValueError("Received task outside of state machine")

                event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete))

    finally:
        if rpc_manager is not None:
            rpc_manager.stop()

    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown()))


def _run_llamacpp_master(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    runner_id: Any,
    shard_metadata: Any,
    setup_start_time: float,
) -> None:
    """
    Run as the distributed master node (device_rank == 0).

    The master runs llama-server with --rpc flag to connect to worker
    RPC servers and distribute tensor operations across all nodes.
    """
    from exo.worker.engines.llamacpp.utils import (
        DistributedLlamaServer,
        build_rpc_address_list,
        build_tensor_split_string,
        get_gguf_path_for_instance,
    )

    distributed_server: DistributedLlamaServer | None = None

    current_status: RunnerStatus = RunnerWaitingForModel()
    logger.info(f"Master node (rank 0) waiting for model")
    event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

    try:
        with task_receiver as tasks:
            for task in tasks:
                event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running))
                event_sender.send(TaskAcknowledged(task_id=task.task_id))

                match task:
                    case LoadModel() if isinstance(current_status, (RunnerWaitingForModel, RunnerFailed)):
                        current_status = RunnerLoading()
                        logger.info("Master loading distributed model")
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                        gguf_path = get_gguf_path_for_instance(bound_instance)
                        rpc_addresses = build_rpc_address_list(bound_instance)
                        tensor_split = build_tensor_split_string(bound_instance)

                        logger.info(f"Master connecting to workers: {rpc_addresses}")
                        logger.info(f"Tensor split: {tensor_split}")

                        distributed_server = DistributedLlamaServer(
                            model_path=str(gguf_path),
                            rpc_addresses=rpc_addresses,
                            tensor_split=tensor_split,
                        )

                        if not distributed_server.start():
                            raise RuntimeError("Failed to start distributed llama-server")

                        current_status = RunnerLoaded()
                        logger.info("Master distributed server loaded")
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                    case StartWarmup() if isinstance(current_status, RunnerLoaded):
                        current_status = RunnerWarmingUp()
                        logger.info("Master warming up distributed inference")
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                        logger.info(f"Master initialized in {time.time() - setup_start_time} seconds")
                        current_status = RunnerReady()
                        logger.info("Master ready for distributed inference")
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerReady()))

                    case ChatCompletion(task_params=task_params, command_id=command_id) if isinstance(
                        current_status, RunnerReady
                    ):
                        assert distributed_server is not None
                        logger.info(f"Master received chat request: {str(task)[:500]}")
                        current_status = RunnerRunning()
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status))

                        assert task_params.messages[0].content is not None
                        _check_for_debug_prompts_llamacpp(task_params.messages[0].content)

                        # Use the existing distributed server for generation
                        from exo.worker.engines.llamacpp.llama_server import distributed_generate
                        generator = distributed_generate(distributed_server.port, task_params)

                        response_count = 0
                        for response in generator:
                            response_count += 1
                            if isinstance(response, GenerationResponse):
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

                        logger.info(f"Master generated {response_count} responses")
                        current_status = RunnerReady()
                        event_sender.send(RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerReady()))

                    case Shutdown():
                        logger.info("Master shutting down")
                        event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete))
                        break

                    case _:
                        raise ValueError("Received task outside of state machine")

                event_sender.send(TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete))

    finally:
        if distributed_server is not None:
            distributed_server.stop()

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
