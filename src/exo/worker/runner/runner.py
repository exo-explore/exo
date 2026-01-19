import base64
import time
from collections.abc import Generator
from functools import cache
from typing import Any, Literal, cast

import mlx.core as mx
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.shared.constants import EXO_MAX_CHUNK_SIZE
from exo.shared.models.model_cards import ModelId, ModelTask
from exo.shared.types.api import ChatCompletionMessageText, ImageGenerationStats
from exo.shared.types.chunks import ImageChunk, TokenChunk, FinishReason
from exo.shared.types.common import CommandId
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
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ImageGenerationResponse,
    PartialImageResponse,
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
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.image import (
    DistributedImageModel,
    generate_image,
    initialize_image_model,
    warmup_image_generator,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    detect_thinking_prompt_suffix,
    initialize_mlx,
    load_mlx_items,
    mlx_force_oom,
)
from exo.worker.parsing.stream import ChunkParser, ChunkParserConfig
from exo.worker.runner.bootstrap import logger


def _get_parser_config(model_id: str) -> ChunkParserConfig:
    """
    Map model_id -> ChunkParserConfig, covering all supported parser types.

    Key behavior:
    - If we return (None, None), parsing is disabled and tool calls will remain as text.
    - For families where the tool dialect differs from the reasoning dialect (e.g. Qwen3 Coder),
      we choose separate parser names for reasoning vs tool parsing.
    """
    lower_id = (model_id or "").lower()

    if "harmony" in lower_id or "gpt-oss" in lower_id:
        return ChunkParserConfig(
            reasoning_parser_name="harmony",
            tool_parser_name="harmony",
            enable_thinking=True,
        )

    if "solar" in lower_id and "open" in lower_id:
        return ChunkParserConfig(
            reasoning_parser_name="solar_open",
            tool_parser_name=None,
            enable_thinking=True,
        )

    if "minimax" in lower_id or "m2" in lower_id:
        return ChunkParserConfig(
            reasoning_parser_name="minimax_m2",
            tool_parser_name="minimax_m2",
            enable_thinking=True,
        )

    if "glm" in lower_id or "chatglm" in lower_id:
        return ChunkParserConfig(
            reasoning_parser_name="glm4_moe",
            tool_parser_name="glm47",
            enable_thinking=True,
        )

    if "nemotron" in lower_id and ("nano" in lower_id or "3" in lower_id):
        return ChunkParserConfig(
            reasoning_parser_name="nemotron3_nano",
            tool_parser_name=None,
            enable_thinking=True,
        )

    if "qwen3" in lower_id:
        if "vl" in lower_id:
            return ChunkParserConfig(
                reasoning_parser_name="qwen3_vl",
                tool_parser_name=None,
                enable_thinking=True,
            )

        if "moe" in lower_id:
            return ChunkParserConfig(
                reasoning_parser_name="qwen3_moe",
                tool_parser_name=None,
                enable_thinking=True,
            )

        if "coder" in lower_id:
            return ChunkParserConfig(
                reasoning_parser_name="qwen3",
                tool_parser_name="qwen3_coder",
                enable_thinking=True,
            )

        return ChunkParserConfig(
            reasoning_parser_name="qwen3",
            tool_parser_name=None,
            enable_thinking=True,
        )

    if "functiongemma" in lower_id or ("gemma" in lower_id and "function" in lower_id):
        return ChunkParserConfig(
            reasoning_parser_name=None,
            tool_parser_name="function_gemma",
            enable_thinking=True,
        )

    if "hermes" in lower_id or "tool-use" in lower_id:
        return ChunkParserConfig(
            reasoning_parser_name="hermes",
            tool_parser_name="json_tools",
            enable_thinking=True,
        )

    return ChunkParserConfig(
        reasoning_parser_name=None,
        tool_parser_name=None,
        enable_thinking=True,
    )


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
    device_rank = shard_metadata.device_rank
    logger.info("hello from the runner")
    if getattr(shard_metadata, "immediate_exception", False):
        raise Exception("Fake exception - runner failed to spin up.")
    if timeout := getattr(shard_metadata, "should_timeout", 0):
        time.sleep(timeout)

    setup_start_time = time.time()

    model: Model | DistributedImageModel | None = None
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

                    if ModelTask.TextGeneration in shard_metadata.model_card.tasks:
                        model, tokenizer = load_mlx_items(
                            bound_instance, group, on_timeout=on_model_load_timeout
                        )
                    elif (
                        ModelTask.TextToImage in shard_metadata.model_card.tasks
                        or ModelTask.ImageToImage in shard_metadata.model_card.tasks
                    ):
                        model = initialize_image_model(bound_instance)
                    else:
                        raise ValueError(
                            f"Unknown model task(s): {shard_metadata.model_card.tasks}"
                        )

                    current_status = RunnerLoaded()
                    logger.info("runner loaded")
                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    assert model

                    current_status = RunnerWarmingUp()
                    logger.info("runner warming up")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    logger.info(f"warming up inference for instance: {instance}")
                    if ModelTask.TextGeneration in shard_metadata.model_card.tasks:
                        assert not isinstance(model, DistributedImageModel)
                        assert tokenizer

                        toks = warmup_inference(
                            model=model,
                            tokenizer=tokenizer,
                            # kv_prefix_cache=kv_prefix_cache,  # supply for warmup-time prefix caching
                        )
                        logger.info(f"warmed up by generating {toks} tokens")
                        logger.info(
                            f"runner initialized in {time.time() - setup_start_time} seconds"
                        )
                    elif (
                        ModelTask.TextToImage in shard_metadata.model_card.tasks
                        or ModelTask.ImageToImage in shard_metadata.model_card.tasks
                    ):
                        assert isinstance(model, DistributedImageModel)
                        image = warmup_image_generator(model=model)
                        if image is not None:
                            logger.info(f"warmed up by generating {image.size} image")
                        else:
                            logger.info("warmup completed (non-primary node)")

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ChatCompletion(task_params=task_params, command_id=command_id) if (
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
                    assert model and not isinstance(model, DistributedImageModel)
                    assert tokenizer
                    assert task_params.messages[0].content is not None

                    try:
                        _check_for_debug_prompts(task_params.messages[0].content)

                        # Build prompt once - used for both generation and thinking detection
                        prompt = apply_chat_template(tokenizer, task_params)

                        # Generate responses using the actual MLX generation
                        mlx_generator = mlx_generate(
                            model=model,
                            tokenizer=tokenizer,
                            task=task_params,
                            prompt=prompt,
                        )

                        # GPT-OSS specific parsing to match other model formats.
                        if isinstance(model, GptOssModel):
                            mlx_generator = parse_gpt_oss(mlx_generator)

                    parser_cfg = _get_parser_config(shard_metadata.model_meta.model_id)
                    parser_cfg.tools = task_params.tools
                    parser = ChunkParser(parser_cfg)

                    has_generated_tool_calls = False
                    for response in mlx_generator:
                        match response:
                            case GenerationResponse():
                                if shard_metadata.device_rank == 0:
                                    parsed_chunks = parser.feed(response.text)

                                    if response.finish_reason is not None:
                                        parsed_chunks.extend(parser.flush())

                                    if not parsed_chunks and response.finish_reason is not None:
                                        parsed_chunks.append("")

                                    for i, p_chunk in enumerate(parsed_chunks):
                                        is_last_chunk = (response.finish_reason is not None) and (i == len(parsed_chunks) - 1)

                                        text_payload = ""
                                        reasoning_payload: str | None = None
                                        tool_calls_payload: list[dict[str, Any]] | None = None

                                        if isinstance(p_chunk, str):
                                            text_payload = p_chunk
                                        else:
                                            if "reasoning_content" in p_chunk:
                                                reasoning_payload = cast(str, p_chunk["reasoning_content"])
                                            elif (
                                                ("type" in p_chunk and p_chunk["type"] == "function")
                                                or ("name" in p_chunk)
                                            ):
                                                tool_calls_payload = [p_chunk]
                                                has_generated_tool_calls = True

                                        finish_reason = None
                                        if is_last_chunk:
                                            finish_reason = response.finish_reason
                                            if has_generated_tool_calls:
                                                finish_reason = "tool_calls"

                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=TokenChunk(
                                                    idx=response.token,
                                                    model=shard_metadata.model_meta.model_id,
                                                    text=text_payload,
                                                    token_id=response.token,
                                                    finish_reason=finish_reason,
                                                    stats=response.stats if is_last_chunk else None,
                                                    reasoning_content=reasoning_payload,
                                                    tool_calls=tool_calls_payload,
                                                ),
                                            )
                                        )
                                # case TokenizedResponse():
                                # TODO: something here ig

                    # can we make this more explicit?
                    except Exception as e:
                        if device_rank == 0:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=TokenChunk(
                                        idx=0,
                                        model=shard_metadata.model_card.model_id,
                                        text="",
                                        token_id=0,
                                        finish_reason="error",
                                        error_message=str(e),
                                    ),
                                )
                            )
                        raise

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ImageGeneration(
                    task_params=task_params, command_id=command_id
                ) if isinstance(current_status, RunnerReady):
                    assert isinstance(model, DistributedImageModel)
                    logger.info(f"received image generation request: {str(task)[:500]}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    try:
                        # Generate images using the image generation backend
                        # Track image_index for final images only
                        image_index = 0
                        for response in generate_image(model=model, task=task_params):
                            if (
                                shard_metadata.device_rank
                                == shard_metadata.world_size - 1
                            ):
                                match response:
                                    case PartialImageResponse():
                                        logger.info(
                                            f"sending partial ImageChunk {response.partial_index}/{response.total_partials}"
                                        )
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                    case ImageGenerationResponse():
                                        logger.info("sending final ImageChunk")
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                        image_index += 1
                    except Exception as e:
                        if shard_metadata.device_rank == shard_metadata.world_size - 1:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=ImageChunk(
                                        idx=0,
                                        model=shard_metadata.model_card.model_id,
                                        data="",
                                        chunk_index=0,
                                        total_chunks=1,
                                        image_index=0,
                                        finish_reason="error",
                                        error_message=str(e),
                                    ),
                                )
                            )
                        raise

                    current_status = RunnerReady()
                    logger.info("runner ready")
                case ImageEdits(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    assert isinstance(model, DistributedImageModel)
                    logger.info(f"received image edits request: {str(task)[:500]}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    try:
                        image_index = 0
                        for response in generate_image(model=model, task=task_params):
                            if (
                                shard_metadata.device_rank
                                == shard_metadata.world_size - 1
                            ):
                                match response:
                                    case PartialImageResponse():
                                        logger.info(
                                            f"sending partial ImageChunk {response.partial_index}/{response.total_partials}"
                                        )
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                    case ImageGenerationResponse():
                                        logger.info("sending final ImageChunk")
                                        _process_image_response(
                                            response,
                                            command_id,
                                            shard_metadata,
                                            event_sender,
                                            image_index,
                                        )
                                        image_index += 1
                    except Exception as e:
                        if shard_metadata.device_rank == shard_metadata.world_size - 1:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=ImageChunk(
                                        idx=0,
                                        model=shard_metadata.model_card.model_id,
                                        data="",
                                        chunk_index=0,
                                        total_chunks=1,
                                        image_index=0,
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


def parse_thinking_models(
    responses: Generator[GenerationResponse],
    tokenizer: TokenizerWrapper,
) -> Generator[GenerationResponse]:
    """
    For models that inject thinking tags in the prompt (like GLM-4.7),
    prepend the thinking tag to the output stream so the frontend
    can properly parse thinking content.
    """
    first = True
    for response in responses:
        if first:
            first = False
            yield response.model_copy(
                update={
                    "text": tokenizer.think_start,
                    "token": tokenizer.think_start_id,  # type: ignore
                }
            )
        yield response


def _send_image_chunk(
    encoded_data: str,
    command_id: CommandId,
    model_id: ModelId,
    event_sender: MpSender[Event],
    image_index: int,
    is_partial: bool,
    partial_index: int | None = None,
    total_partials: int | None = None,
    stats: ImageGenerationStats | None = None,
    image_format: Literal["png", "jpeg", "webp"] | None = None,
) -> None:
    """Send base64-encoded image data as chunks via events."""
    data_chunks = [
        encoded_data[i : i + EXO_MAX_CHUNK_SIZE]
        for i in range(0, len(encoded_data), EXO_MAX_CHUNK_SIZE)
    ]
    total_chunks = len(data_chunks)
    for chunk_index, chunk_data in enumerate(data_chunks):
        # Only include stats on the last chunk of the final image
        chunk_stats = (
            stats if chunk_index == total_chunks - 1 and not is_partial else None
        )
        event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=ImageChunk(
                    idx=chunk_index,
                    model=model_id,
                    data=chunk_data,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    image_index=image_index,
                    is_partial=is_partial,
                    partial_index=partial_index,
                    total_partials=total_partials,
                    stats=chunk_stats,
                    format=image_format,
                ),
            )
        )


def _process_image_response(
    response: ImageGenerationResponse | PartialImageResponse,
    command_id: CommandId,
    shard_metadata: ShardMetadata,
    event_sender: MpSender[Event],
    image_index: int,
) -> None:
    """Process a single image response and send chunks."""
    encoded_data = base64.b64encode(response.image_data).decode("utf-8")
    is_partial = isinstance(response, PartialImageResponse)
    # Extract stats from final ImageGenerationResponse if available
    stats = response.stats if isinstance(response, ImageGenerationResponse) else None
    _send_image_chunk(
        encoded_data=encoded_data,
        command_id=command_id,
        model_id=shard_metadata.model_card.model_id,
        event_sender=event_sender,
        image_index=response.partial_index if is_partial else image_index,
        is_partial=is_partial,
        partial_index=response.partial_index if is_partial else None,
        total_partials=response.total_partials if is_partial else None,
        stats=stats,
        image_format=response.format,
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
