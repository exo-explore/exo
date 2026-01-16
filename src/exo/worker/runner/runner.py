import json
import time
from collections.abc import Generator
from functools import cache
from typing import Any
from uuid import uuid4

import mlx.core as mx
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.chunks import TokenChunk, ToolCall, ToolCallFunction
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ChatCompletion,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
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
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
from exo.worker.engines.mlx.utils_mlx import (
    initialize_mlx,
    load_mlx_items,
    mlx_force_oom,
)
from exo.worker.runner.bootstrap import logger


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
):
    instance, runner_id, shard_metadata = (
        bound_instance.instance,
        bound_instance.bound_runner_id,
        bound_instance.bound_shard,
    )
    logger.info("hello from the runner")
    if getattr(shard_metadata, "immediate_exception", False):
        raise Exception("Fake exception - runner failed to spin up.")
    if timeout := getattr(shard_metadata, "should_timeout", 0):
        time.sleep(timeout)

    setup_start_time = time.time()

    model = None
    tokenizer = None
    group = None

    current_status: RunnerStatus = RunnerIdle()
    logger.info("runner created")
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )
    with task_receiver as tasks:
        for task in tasks:
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
            )
            event_sender.send(TaskAcknowledged(task_id=task.task_id))
            match task:
                case ConnectToGroup() if isinstance(
                    current_status, (RunnerIdle, RunnerFailed)
                ):
                    logger.info("runner connecting")
                    current_status = RunnerConnecting()
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    group = initialize_mlx(bound_instance)

                    logger.info("runner connected")
                    current_status = RunnerConnected()

                # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
                case LoadModel() if (
                    isinstance(current_status, RunnerConnected) and group is not None
                ) or (isinstance(current_status, RunnerIdle) and group is None):
                    current_status = RunnerLoading()
                    logger.info("runner loading")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    model, tokenizer = load_mlx_items(bound_instance, group)

                    current_status = RunnerLoaded()
                    logger.info("runner loaded")
                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    assert model
                    assert tokenizer
                    current_status = RunnerWarmingUp()
                    logger.info("runner warming up")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    logger.info(f"warming up inference for instance: {instance}")
                    toks = warmup_inference(
                        model=model,
                        tokenizer=tokenizer,
                        # kv_prefix_cache=kv_prefix_cache,  # supply for warmup-time prefix caching
                    )
                    logger.info(f"warmed up by generating {toks} tokens")
                    logger.info(
                        f"runner initialized in {time.time() - setup_start_time} seconds"
                    )
                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ChatCompletion(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    assert model
                    assert tokenizer
                    logger.info(f"received chat request: {str(task)[:500]}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    assert task_params.messages[0].content is not None
                    _check_for_debug_prompts(task_params.messages[0].content)

                    # Generate responses using the actual MLX generation
                    mlx_generator = mlx_generate(
                        model=model,
                        tokenizer=tokenizer,
                        task=task_params,
                    )

                    # GPT-OSS specific parsing to match other model formats.
                    if isinstance(model, GptOssModel):
                        mlx_generator = parse_gpt_oss(mlx_generator)

                    # Parse tool calls to place them in the tool calls section
                    mlx_generator = parse_tool_calls(
                        mlx_generator, tokenizer, task_params.tools
                    )

                    for response in mlx_generator:
                        match response:
                            case GenerationResponse():
                                if shard_metadata.device_rank == 0:
                                    event_sender.send(
                                        ChunkGenerated(
                                            command_id=command_id,
                                            chunk=TokenChunk(
                                                idx=response.token,
                                                model=shard_metadata.model_meta.model_id,
                                                text=response.text,
                                                token_id=response.token,
                                                finish_reason=response.finish_reason,
                                                stats=response.stats,
                                                tool_calls=response.tool_calls,
                                            ),
                                        )
                                    )
                                # case TokenizedResponse():
                                # TODO: something here ig

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case Shutdown():
                    current_status = RunnerShuttingDown()
                    logger.info("runner shutting down")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    current_status = RunnerShutdown()
                case _:
                    raise ValueError(
                        f"Received {task.__class__.__name__} outside of state machine in {current_status=}"
                    )
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete)
            )
            event_sender.send(
                RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
            )
            if isinstance(current_status, RunnerShutdown):
                del model, tokenizer, group
                mx.clear_cache()
                import gc

                gc.collect()
                break


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def parse_gpt_oss(
    responses: Generator[GenerationResponse],
) -> Generator[GenerationResponse]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False

    for response in responses:
        stream.process(response.token)

        delta = stream.last_content_delta
        ch = stream.current_channel

        if ch == "analysis" and not thinking:
            thinking = True
            yield response.model_copy(update={"text": "<think>"})

        if ch != "analysis" and thinking:
            thinking = False
            yield response.model_copy(update={"text": "</think>"})

        if delta:
            yield response.model_copy(update={"text": delta})

        if response.finish_reason is not None:
            if thinking:
                yield response.model_copy(update={"text": "</think>"})
            yield response
            break


def _generate_tool_call_id() -> str:
    return f"call_{uuid4().hex[:24]}"


def _parse_tool_call_content(
    content: str,
    tokenizer: TokenizerWrapper,
    tools: list[dict[str, Any]] | None,
) -> ToolCall | None:
    content = content.strip()
    if not content:
        return None

    tool_parser: Any = getattr(tokenizer, "tool_parser", None)
    if tool_parser is None:
        logger.warning("No tool_parser available for tokenizer")
        return None

    try:
        parsed: dict[str, Any] = tool_parser(content, tools)  # pyright: ignore[reportAny]
        if parsed and "name" in parsed:
            arguments: Any = parsed.get("arguments", {})  # pyright: ignore[reportAny]
            arguments_str: str = (
                json.dumps(arguments)
                if not isinstance(arguments, str)
                else arguments
            )
            return ToolCall(
                id=_generate_tool_call_id(),
                type="function",
                function=ToolCallFunction(
                    name=str(parsed["name"]),  # pyright: ignore[reportAny]
                    arguments=arguments_str,
                ),
            )
    except Exception as e:
        logger.warning(f"tool_parser failed: {e}")

    return None


def parse_tool_calls(
    responses: Generator[GenerationResponse],
    tokenizer: TokenizerWrapper,
    tools: list[dict[str, Any]] | None,
) -> Generator[GenerationResponse]:
    has_tool_calling = getattr(tokenizer, "has_tool_calling", False)
    if not has_tool_calling or tools is None:
        yield from responses
        return

    tool_call_start: str | None = getattr(tokenizer, "tool_call_start", None)
    tool_call_end: str | None = getattr(tokenizer, "tool_call_end", None)

    if tool_call_start is None or tool_call_end is None:
        yield from responses
        return

    in_tool_call = False
    tool_call_buffer: list[str] = []
    pending_tool_calls: list[ToolCall] = []

    for response in responses:
        if response.text == tool_call_start:
            in_tool_call = True
            tool_call_buffer = []
            continue

        if response.text == tool_call_end:
            in_tool_call = False
            parsed = _parse_tool_call_content(
                "".join(tool_call_buffer), tokenizer, tools
            )
            if parsed is not None:
                pending_tool_calls.append(parsed)
            continue

        if in_tool_call:
            tool_call_buffer.append(response.text)
            continue

        if response.finish_reason is None or not pending_tool_calls:
            yield response
        else:
            yield response.model_copy(
                update={
                    "finish_reason": "tool_calls",
                    "tool_calls": pending_tool_calls if pending_tool_calls else None,
                }
            )


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(
    prompt: str | ChatCompletionMessageText | list[ChatCompletionMessageText],
):
    if isinstance(prompt, list):
        if len(prompt) == 0:
            logger.debug("Empty message prompt received in debug prompt")
            return
        prompt = prompt[0]

    if isinstance(prompt, ChatCompletionMessageText):
        prompt = prompt.text

    if EXO_RUNNER_MUST_FAIL in prompt:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)
