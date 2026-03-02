import resource
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

from exo.shared.models.model_cards import ModelTask
from exo.shared.tokenizer.chat_template import apply_chat_template
from exo.shared.types.chunks import (
    ErrorChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.tinygrad.generator.generate import (
    cleanup_jit_state,
    tinygrad_generate,
    warmup_inference,
)
from exo.worker.engines.tinygrad.utils_tinygrad import (
    initialize_tinygrad,
    load_tinygrad_items,
)
from exo.worker.engines.tinygrad.weight_loader import TransformerWeights
from exo.worker.runner.bootstrap import logger

from .tool_parsers import ToolParser, make_mlx_parser


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    runner_id = bound_instance.bound_runner_id
    shard_metadata = bound_instance.bound_shard

    logger.info("hello from the tinygrad runner")

    setup_start_time = time.time()
    cancelled_tasks = set[TaskId]()

    inference_model: TransformerWeights | None = None
    tokenizer = None
    tool_parser: ToolParser | None = None
    check_for_cancel_every: int | None = None

    current_status: RunnerStatus = RunnerIdle()
    logger.info("tinygrad runner created")
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )
    seen = set[TaskId]()
    with task_receiver as tasks:
        for task in tasks:
            if task.task_id in seen:
                logger.warning("repeat task - potential error")
            seen.add(task.task_id)
            cancelled_tasks.discard(TaskId("CANCEL_CURRENT_TASK"))
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
            )
            match task:
                case LoadModel() if isinstance(current_status, (RunnerIdle, RunnerFailed)):
                    total_layers = shard_metadata.end_layer - shard_metadata.start_layer
                    current_status = RunnerLoading(
                        layers_loaded=0, total_layers=total_layers
                    )
                    logger.info("tinygrad runner loading")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    def on_model_load_timeout() -> None:
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id,
                                runner_status=RunnerFailed(
                                    error_message="Model loading timed out"
                                ),
                            )
                        )
                        time.sleep(0.5)

                    assert (
                        ModelTask.TextGeneration in shard_metadata.model_card.tasks
                    ), f"Incorrect model task(s): {shard_metadata.model_card.tasks}"

                    initialize_tinygrad(bound_instance)

                    inference_model, tokenizer = load_tinygrad_items(  # pyright: ignore[reportAny]
                        bound_instance,
                        None,
                        on_timeout=on_model_load_timeout,
                    )
                    logger.info(
                        f"model has_tool_calling={tokenizer.has_tool_calling} using tokens {tokenizer.tool_call_start}, {tokenizer.tool_call_end}"  # pyright: ignore[reportAny]
                    )
                    if tokenizer.has_tool_calling:  # pyright: ignore[reportAny]
                        assert tokenizer.tool_call_start  # pyright: ignore[reportAny]
                        assert tokenizer.tool_call_end  # pyright: ignore[reportAny]
                        assert tokenizer.tool_parser  # pyright: ignore[reportAny]
                        tool_parser = make_mlx_parser(
                            tokenizer.tool_call_start,  # pyright: ignore[reportAny]
                            tokenizer.tool_call_end,  # pyright: ignore[reportAny]
                            tokenizer.tool_parser,  # pyright: ignore[reportAny]
                        )
                    current_status = RunnerLoaded()
                    logger.info("tinygrad runner loaded")

                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    current_status = RunnerWarmingUp()
                    logger.info("tinygrad runner warming up")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    assert inference_model
                    assert tokenizer

                    t = time.monotonic()
                    toks = warmup_inference(
                        model=inference_model,
                        tokenizer=tokenizer,
                    )
                    logger.info(f"warmed up by generating {toks} tokens")
                    check_for_cancel_every = min(
                        max(1, round(toks / max(time.monotonic() - t, 0.001))), 100
                    )
                    logger.info(
                        f"tinygrad runner checking for cancellation every {check_for_cancel_every} tokens"
                    )
                    logger.info(
                        f"tinygrad runner initialized in {time.time() - setup_start_time} seconds"
                    )
                    current_status = RunnerReady()
                    logger.info("tinygrad runner ready")

                case TextGeneration(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    logger.info(f"received chat request: {task}")
                    current_status = RunnerRunning()
                    logger.info("tinygrad runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))
                    assert inference_model
                    assert tokenizer
                    assert check_for_cancel_every

                    try:
                        _check_for_debug_prompts(task_params)

                        prompt = apply_chat_template(tokenizer, task_params)

                        gen: Generator[GenerationResponse | ToolCallResponse] = tinygrad_generate(
                            model=inference_model,
                            tokenizer=tokenizer,
                            task=task_params,
                            prompt=prompt,
                        )

                        if tool_parser:
                            gen = _parse_tool_calls(gen, tool_parser)

                        completion_tokens = 0
                        tokens_since_last_cancel_check = check_for_cancel_every
                        for response in gen:
                            tokens_since_last_cancel_check += 1
                            if tokens_since_last_cancel_check >= check_for_cancel_every:
                                tokens_since_last_cancel_check = 0
                                cancelled_tasks.update(cancel_receiver.collect())
                                want_to_cancel = (task.task_id in cancelled_tasks) or (
                                    TaskId("CANCEL_CURRENT_TASK") in cancelled_tasks
                                )
                                if want_to_cancel:
                                    break

                            match response:
                                case GenerationResponse():
                                    completion_tokens += 1
                                    if response.finish_reason == "error":
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=ErrorChunk(
                                                    error_message=response.text,
                                                    model=shard_metadata.model_card.model_id,
                                                ),
                                            )
                                        )
                                    else:
                                        assert response.finish_reason not in (
                                            "error",
                                            "tool_calls",
                                            "function_call",
                                        )
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=TokenChunk(
                                                    model=shard_metadata.model_card.model_id,
                                                    text=response.text,
                                                    token_id=response.token,
                                                    usage=response.usage,
                                                    finish_reason=response.finish_reason,
                                                    stats=response.stats,
                                                    logprob=response.logprob,
                                                    top_logprobs=response.top_logprobs,
                                                    is_thinking=response.is_thinking,
                                                ),
                                            )
                                        )
                                case ToolCallResponse():
                                    event_sender.send(
                                        ChunkGenerated(
                                            command_id=command_id,
                                            chunk=ToolCallChunk(
                                                tool_calls=response.tool_calls,
                                                model=shard_metadata.model_card.model_id,
                                                usage=response.usage,
                                                stats=response.stats,
                                            ),
                                        )
                                    )

                    except Exception as e:
                        event_sender.send(
                            ChunkGenerated(
                                command_id=command_id,
                                chunk=ErrorChunk(
                                    model=shard_metadata.model_card.model_id,
                                    finish_reason="error",
                                    error_message=str(e),
                                ),
                            )
                        )
                        raise

                    current_status = RunnerReady()
                    logger.info("tinygrad runner ready")

                case Shutdown():
                    current_status = RunnerShuttingDown()
                    logger.info("tinygrad runner shutting down")
                    if not TYPE_CHECKING:
                        del inference_model, tokenizer
                        cleanup_jit_state()
                        import gc

                        gc.collect()

                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    current_status = RunnerShutdown()

                case _:
                    raise ValueError(
                        f"Received {task.__class__.__name__} outside of state machine in {current_status=}"
                    )

            was_cancelled = (task.task_id in cancelled_tasks) or (
                TaskId("CANCEL_CURRENT_TASK") in cancelled_tasks
            )
            if not was_cancelled:
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
            event_sender.send(
                RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
            )

            if isinstance(current_status, RunnerShutdown):
                break


def _parse_tool_calls(
    responses: Generator[GenerationResponse | ToolCallResponse],
    tool_parser: ToolParser,
) -> Generator[GenerationResponse | ToolCallResponse]:
    """Wrap a generation stream to detect and parse tool calls."""
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    for response in responses:
        if not isinstance(response, GenerationResponse):
            yield response
            continue

        if response.text.startswith(tool_parser.start_parsing):
            in_tool_call = True

        if in_tool_call:
            tool_call_text_parts.append(response.text)
            if response.text.endswith(tool_parser.end_parsing):
                parsed = tool_parser.parse_tool_calls(
                    "".join(tool_call_text_parts).strip()
                )
                logger.info(f"parsed {tool_call_text_parts=} into {parsed=}")
                if parsed is not None:
                    yield ToolCallResponse(
                        tool_calls=parsed, usage=response.usage, stats=response.stats
                    )
                else:
                    logger.warning(
                        f"tool call parsing failed for text {''.join(tool_call_text_parts)}"
                    )
                    response = response.model_copy(
                        update={"text": "".join(tool_call_text_parts)}
                    )
                    yield response

                in_tool_call = False
                tool_call_text_parts = []
                continue

            if response.finish_reason is not None:
                logger.info(
                    "tool call parsing interrupted, yield partial tool call as text"
                )
                response = response.model_copy(
                    update={
                        "text": "".join(tool_call_text_parts),
                        "token": 0,
                    }
                )
                yield response

            continue

        yield response


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    if len(task_params.input) == 0:
        return
    prompt = task_params.input[0].content

    if not prompt:
        return

    if EXO_RUNNER_MUST_FAIL in prompt:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)
