import math
import resource
import time
from collections.abc import Generator
from functools import cache
from typing import TYPE_CHECKING, cast

import mlx.core as mx
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    HarmonyError,  # pyright: ignore[reportUnknownVariableType]
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
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
    ConnectToGroup,
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
    ToolCallItem,
    ToolCallResponse,
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
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.generate import (
    PrefillCancelled,
    mlx_generate,
    warmup_inference,
)
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    detect_thinking_prompt_suffix,
    initialize_mlx,
    load_mlx_items,
    mlx_force_oom,
    mx_any,
)
from exo.worker.runner.bootstrap import logger

from .tool_parsers import ToolParser, make_mlx_parser


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    instance, runner_id, shard_metadata = (
        bound_instance.instance,
        bound_instance.bound_runner_id,
        bound_instance.bound_shard,
    )
    model_id = shard_metadata.model_card.model_id
    device_rank = shard_metadata.device_rank
    logger.info("hello from the runner")
    if getattr(shard_metadata, "immediate_exception", False):
        raise Exception("Fake exception - runner failed to spin up.")
    if timeout := getattr(shard_metadata, "should_timeout", 0):
        time.sleep(timeout)

    setup_start_time = time.time()
    cancelled_tasks = set[TaskId]()

    inference_model: Model | None = None
    tokenizer = None
    tool_parser: ToolParser | None = None
    group = None
    kv_prefix_cache: KVPrefixCache | None = None
    check_for_cancel_every: int | None = None

    current_status: RunnerStatus = RunnerIdle()
    logger.info("runner created")
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
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))
                    group = initialize_mlx(bound_instance)

                    logger.info("runner connected")
                    current_status = RunnerConnected()

                # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
                case LoadModel() if (
                    isinstance(current_status, RunnerConnected) and group is not None
                ) or (isinstance(current_status, RunnerIdle) and group is None):
                    total_layers = shard_metadata.end_layer - shard_metadata.start_layer
                    current_status = RunnerLoading(
                        layers_loaded=0, total_layers=total_layers
                    )
                    logger.info("runner loading")
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

                    def on_layer_loaded(layers_loaded: int, total: int) -> None:
                        nonlocal current_status
                        current_status = RunnerLoading(
                            layers_loaded=layers_loaded, total_layers=total
                        )
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id,
                                runner_status=current_status,
                            )
                        )

                    assert (
                        ModelTask.TextGeneration in shard_metadata.model_card.tasks
                    ), f"Incorrect model task(s): {shard_metadata.model_card.tasks}"
                    inference_model, tokenizer = load_mlx_items(
                        bound_instance,
                        group,
                        on_timeout=on_model_load_timeout,
                        on_layer_loaded=on_layer_loaded,
                    )
                    logger.info(
                        f"model has_tool_calling={tokenizer.has_tool_calling} using tokens {tokenizer.tool_call_start}, {tokenizer.tool_call_end}"
                    )
                    if tokenizer.has_tool_calling:
                        assert tokenizer.tool_call_start
                        assert tokenizer.tool_call_end
                        assert tokenizer.tool_parser  # pyright: ignore[reportAny]
                        tool_parser = make_mlx_parser(
                            tokenizer.tool_call_start,
                            tokenizer.tool_call_end,
                            tokenizer.tool_parser,  # pyright: ignore[reportAny]
                        )
                    kv_prefix_cache = KVPrefixCache(group)
                    current_status = RunnerLoaded()
                    logger.info("runner loaded")
                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    current_status = RunnerWarmingUp()
                    logger.info("runner warming up")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))

                    logger.info(f"warming up inference for instance: {instance}")
                    assert inference_model
                    assert tokenizer

                    t = time.monotonic()
                    toks = warmup_inference(
                        model=cast(Model, inference_model),
                        tokenizer=tokenizer,
                        group=group,
                    )
                    logger.info(f"warmed up by generating {toks} tokens")
                    check_for_cancel_every = min(
                        math.ceil(toks / min(time.monotonic() - t, 0.001)), 100
                    )
                    if group is not None:
                        check_for_cancel_every = int(
                            mx.max(
                                mx.distributed.all_gather(
                                    mx.array([check_for_cancel_every]), group=group
                                )
                            ).item()
                        )

                    logger.info(
                        f"runner checking for cancellation every {check_for_cancel_every} tokens"
                    )
                    logger.info(
                        f"runner initialized in {time.time() - setup_start_time} seconds"
                    )
                    current_status = RunnerReady()
                    logger.info("runner ready")
                case TextGeneration(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    logger.info(f"received chat request: {task}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    event_sender.send(TaskAcknowledged(task_id=task.task_id))
                    assert inference_model
                    assert tokenizer
                    assert check_for_cancel_every

                    # Define callback to send prefill progress events
                    # and check for cancellation between prefill chunks.
                    # TODO(evan): kill the callbacks/runner refactor
                    #  Specifically the part that this is literally duplicated code.
                    def on_prefill_progress(
                        processed: int,
                        total: int,
                        _task_id: TaskId = task.task_id,
                        _group: mx.distributed.Group | None = group,
                    ) -> None:
                        if device_rank == 0:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=PrefillProgressChunk(
                                        model=model_id,
                                        processed_tokens=processed,
                                        total_tokens=total,
                                    ),
                                )
                            )
                        cancelled_tasks.update(cancel_receiver.collect())
                        want_to_cancel = (_task_id in cancelled_tasks) or (
                            TaskId("CANCEL_CURRENT_TASK") in cancelled_tasks
                        )
                        if mx_any(want_to_cancel, _group):
                            raise PrefillCancelled()

                    try:
                        _check_for_debug_prompts(task_params)

                        # Build prompt once - used for both generation and thinking detection
                        prompt = apply_chat_template(tokenizer, task_params)

                        # Generate responses using the actual MLX generation
                        mlx_generator = mlx_generate(
                            model=cast(Model, inference_model),
                            tokenizer=tokenizer,
                            task=task_params,
                            prompt=prompt,
                            kv_prefix_cache=kv_prefix_cache,
                            on_prefill_progress=on_prefill_progress,
                            group=group,
                        )

                        if tokenizer.has_thinking:
                            mlx_generator = parse_thinking_models(
                                mlx_generator,
                                tokenizer,
                                # For other thinking models (GLM, etc.), check if we need to
                                # prepend the thinking tag that was consumed by the chat template
                                starts_in_thinking=detect_thinking_prompt_suffix(
                                    prompt, tokenizer
                                ),
                            )

                        # Model-specific output parsing for tool calls.
                        if isinstance(inference_model, GptOssModel):
                            mlx_generator = parse_gpt_oss(mlx_generator)
                        elif (
                            isinstance(inference_model, DeepseekV32Model)
                            and "deepseek" in model_id.normalize().lower()
                        ):
                            mlx_generator = parse_deepseek_v32(mlx_generator)
                        elif tool_parser:
                            mlx_generator = parse_tool_calls(mlx_generator, tool_parser)

                        completion_tokens = 0
                        tokens_since_last_cancel_check = check_for_cancel_every
                        for response in mlx_generator:
                            tokens_since_last_cancel_check += 1
                            if tokens_since_last_cancel_check >= check_for_cancel_every:
                                tokens_since_last_cancel_check = 0
                                cancelled_tasks.update(cancel_receiver.collect())
                                want_to_cancel = (task.task_id in cancelled_tasks) or (
                                    TaskId("CANCEL_CURRENT_TASK") in cancelled_tasks
                                )
                                if mx_any(want_to_cancel, group):
                                    break

                            match response:
                                case GenerationResponse():
                                    completion_tokens += 1
                                    if (
                                        device_rank == 0
                                        and response.finish_reason == "error"
                                    ):
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=ErrorChunk(
                                                    error_message=response.text,
                                                    model=model_id,
                                                ),
                                            )
                                        )

                                    elif device_rank == 0:
                                        assert response.finish_reason not in (
                                            "error",
                                            "tool_calls",
                                            "function_call",
                                        )
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=TokenChunk(
                                                    model=model_id,
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
                                    if device_rank == 0:
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=ToolCallChunk(
                                                    tool_calls=response.tool_calls,
                                                    model=model_id,
                                                    usage=response.usage,
                                                    stats=response.stats,
                                                ),
                                            )
                                        )

                    except PrefillCancelled:
                        logger.info(f"Prefill cancelled for task {task.task_id}")
                    # can we make this more explicit?
                    except Exception as e:
                        if device_rank == 0:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=ErrorChunk(
                                        model=model_id,
                                        finish_reason="error",
                                        error_message=str(e),
                                    ),
                                )
                            )
                        raise

                    current_status = RunnerReady()
                    logger.info("runner ready")

                case Shutdown():
                    current_status = RunnerShuttingDown()
                    logger.info("runner shutting down")
                    if not TYPE_CHECKING:
                        del inference_model, tokenizer, group
                        mx.clear_cache()
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


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def parse_gpt_oss(
    responses: Generator[GenerationResponse],
) -> Generator[GenerationResponse | ToolCallResponse]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    for response in responses:
        assert isinstance(response, GenerationResponse)
        try:
            stream.process(response.token)
        except HarmonyError:
            logger.error("Encountered critical Harmony Error, returning early")
            return

        delta = stream.last_content_delta
        ch = stream.current_channel
        recipient = stream.current_recipient

        # Debug: log every token with state
        logger.debug(
            f"parse_gpt_oss token={response.token} text={response.text!r} "
            f"recipient={recipient!r} ch={ch!r} delta={delta!r} "
            f"state={stream.state} current_tool={current_tool_name!r}"
        )

        if recipient != current_tool_name:
            if current_tool_name is not None:
                prefix = "functions."
                if current_tool_name.startswith(prefix):
                    current_tool_name = current_tool_name[len(prefix) :]
                logger.info(
                    f"parse_gpt_oss yielding tool call: name={current_tool_name!r}"
                )
                yield ToolCallResponse(
                    tool_calls=[
                        ToolCallItem(
                            name=current_tool_name,
                            arguments="".join(tool_arg_parts).strip(),
                        )
                    ],
                    usage=response.usage,
                )
                tool_arg_parts = []
            current_tool_name = recipient

        # If inside a tool call, accumulate arguments
        if current_tool_name is not None:
            if delta:
                tool_arg_parts.append(delta)
            continue

        if ch == "analysis" and not thinking:
            thinking = True

        if ch != "analysis" and thinking:
            thinking = False

        if delta:
            yield response.model_copy(update={"text": delta, "is_thinking": thinking})

        if response.finish_reason is not None:
            yield response


def parse_deepseek_v32(
    responses: Generator[GenerationResponse],
) -> Generator[GenerationResponse | ToolCallResponse]:
    """Parse DeepSeek V3.2 DSML tool calls from the generation stream.

    Uses accumulated-text matching (not per-token marker checks) because
    DSML markers like <｜DSML｜function_calls> may span multiple tokens.
    Also handles <think>...</think> blocks for thinking mode.
    """
    from exo.worker.engines.mlx.dsml_encoding import (
        THINKING_END,
        THINKING_START,
        TOOL_CALLS_END,
        TOOL_CALLS_START,
        parse_dsml_output,
    )

    accumulated = ""
    in_tool_call = False
    thinking = False
    # Tokens buffered while we detect the start of a DSML block
    pending_buffer: list[GenerationResponse] = []
    # Text accumulated during a tool call block
    tool_call_text = ""

    for response in responses:
        assert isinstance(response, GenerationResponse)

        # ── Handle thinking tags ──
        if not thinking and THINKING_START in response.text:
            thinking = True
            # Yield any text before the <think> tag
            before = response.text[: response.text.index(THINKING_START)]
            if before:
                yield response.model_copy(update={"text": before})
            continue

        if thinking and THINKING_END in response.text:
            thinking = False
            # Yield any text after the </think> tag
            after = response.text[
                response.text.index(THINKING_END) + len(THINKING_END) :
            ]
            if after:
                yield response.model_copy(update={"text": after, "is_thinking": False})
            continue

        if thinking:
            yield response.model_copy(update={"is_thinking": True})
            continue

        # ── Handle tool call accumulation ──
        if in_tool_call:
            tool_call_text += response.text
            if TOOL_CALLS_END in tool_call_text:
                # Parse the accumulated DSML block
                parsed = parse_dsml_output(tool_call_text)
                if parsed is not None:
                    logger.info(f"parsed DSML tool calls: {parsed}")
                    yield ToolCallResponse(
                        tool_calls=parsed,
                        usage=response.usage,
                        stats=response.stats,
                    )
                else:
                    logger.warning(
                        f"DSML tool call parsing failed for: {tool_call_text}"
                    )
                    yield response.model_copy(update={"text": tool_call_text})
                in_tool_call = False
                tool_call_text = ""
                continue

            # EOS reached before end marker — yield buffered text as-is
            if response.finish_reason is not None:
                logger.info("DSML tool call parsing interrupted by EOS")
                yield response.model_copy(update={"text": tool_call_text})
                in_tool_call = False
                tool_call_text = ""
            continue

        # ── Detect start of tool call block ──
        accumulated += response.text

        if TOOL_CALLS_START in accumulated:
            # The start marker might be split across pending_buffer + current token
            start_idx = accumulated.index(TOOL_CALLS_START)
            # Yield any pending tokens that are purely before the marker
            pre_text = accumulated[:start_idx]
            if pre_text:
                # Flush pending buffer tokens that contributed text before the marker
                for buf_resp in pending_buffer:
                    if pre_text:
                        chunk = buf_resp.text
                        if len(chunk) <= len(pre_text):
                            yield buf_resp
                            pre_text = pre_text[len(chunk) :]
                        else:
                            yield buf_resp.model_copy(update={"text": pre_text})
                            pre_text = ""
            pending_buffer = []
            tool_call_text = accumulated[start_idx:]
            accumulated = ""

            # Check if the end marker is already present (entire tool call in one token)
            if TOOL_CALLS_END in tool_call_text:
                parsed = parse_dsml_output(tool_call_text)
                if parsed is not None:
                    logger.info(f"parsed DSML tool calls: {parsed}")
                    yield ToolCallResponse(
                        tool_calls=parsed,
                        usage=response.usage,
                        stats=response.stats,
                    )
                else:
                    logger.warning(
                        f"DSML tool call parsing failed for: {tool_call_text}"
                    )
                    yield response.model_copy(update={"text": tool_call_text})
                tool_call_text = ""
            else:
                in_tool_call = True
            continue

        # Check if accumulated text might be the start of a DSML marker
        # Buffer tokens if we see a partial match at the end
        if _could_be_dsml_prefix(accumulated):
            pending_buffer.append(response)
            continue

        # No partial match — flush all pending tokens and the current one
        for buf_resp in pending_buffer:
            yield buf_resp
        pending_buffer = []
        accumulated = ""
        yield response

    # Flush any remaining pending buffer at generator end
    for buf_resp in pending_buffer:
        yield buf_resp


def _could_be_dsml_prefix(text: str) -> bool:
    """Check if the end of text could be the start of a DSML function_calls marker.

    We look for suffixes of text that are prefixes of the TOOL_CALLS_START pattern.
    This allows us to buffer tokens until we can determine if a tool call is starting.
    """
    from exo.worker.engines.mlx.dsml_encoding import TOOL_CALLS_START

    # Only check the last portion of text that could overlap with the marker
    max_check = len(TOOL_CALLS_START)
    tail = text[-max_check:] if len(text) > max_check else text

    # Check if any suffix of tail is a prefix of TOOL_CALLS_START
    for i in range(len(tail)):
        suffix = tail[i:]
        if TOOL_CALLS_START.startswith(suffix):
            return True
    return False


def parse_thinking_models(
    responses: Generator[GenerationResponse],
    tokenizer: TokenizerWrapper,
    starts_in_thinking: bool = True,
) -> Generator[GenerationResponse]:
    """Route thinking tokens via is_thinking flag.

    Swallows think tag tokens, sets is_thinking on all others.
    Always yields tokens with finish_reason to avoid hanging the chunk stream.
    """
    in_thinking = starts_in_thinking
    for response in responses:
        if isinstance(response, ToolCallResponse):
            yield response
            continue

        is_think_tag = (
            tokenizer.think_end is not None and response.text == tokenizer.think_end
        ) or (
            tokenizer.think_start is not None and response.text == tokenizer.think_start
        )

        if is_think_tag:
            in_thinking = response.text != tokenizer.think_end
            # Never swallow finish_reason — the chunk stream needs it to terminate.
            if response.finish_reason is not None:
                yield response.model_copy(update={"text": "", "is_thinking": False})
            continue
        yield response.model_copy(update={"is_thinking": in_thinking})


def parse_tool_calls(
    responses: Generator[GenerationResponse], tool_parser: ToolParser
) -> Generator[GenerationResponse | ToolCallResponse]:
    in_tool_call = False
    tool_call_text_parts: list[str] = []
    for response in responses:
        if not in_tool_call and response.text.startswith(tool_parser.start_parsing):
            in_tool_call = True

        if in_tool_call:
            tool_call_text_parts.append(response.text)
            if response.text.endswith(tool_parser.end_parsing):
                # parse the actual tool calls from the tool call text
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
                    response.text = "".join(tool_call_text_parts)
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

        else:
            # fallthrough
            yield response


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Check for debug prompt triggers in the input.

    Extracts the first user input text and checks for debug triggers.
    """
    if len(task_params.input) == 0:
        logger.debug("Empty message list in debug prompt check")
        return
    prompt = task_params.input[0].content

    if not prompt:
        return

    if EXO_RUNNER_MUST_FAIL in prompt:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)
