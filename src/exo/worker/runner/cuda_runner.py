import time
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.generation.utils import GenerateOutput

from exo.download.download_utils import build_model_path
from exo.shared.models.model_cards import ModelTask
from exo.shared.types.api import (
    ChatCompletionMessage,
    ChatCompletionMessageText,
    GenerationStats,
)
from exo.shared.types.chunks import ErrorChunk, TokenChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
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
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.runner.bootstrap import logger

DEFAULT_MAX_NEW_TOKENS = 256


def _normalize_messages(
    messages: list[ChatCompletionMessage],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        content_text = ""
        if message.content is not None:
            if isinstance(message.content, ChatCompletionMessageText):
                content_text = message.content.text
            elif isinstance(message.content, list):
                content_text = "\n".join(c.text for c in message.content).strip()
            else:
                content_text = message.content
        if message.thinking is not None:
            if content_text:
                content_text = f"{content_text}\n{message.thinking}".strip()
            else:
                content_text = message.thinking
        if not content_text:
            continue
        normalized.append({"role": message.role, "content": content_text})
    return normalized


def _build_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[ChatCompletionMessage],
) -> str:
    normalized = _normalize_messages(messages)
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )
    prompt_lines = [f"{m['role']}: {m['content']}" for m in normalized]
    return "\n".join(prompt_lines) + "\nassistant:"


def _resolve_eos_token_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> set[int]:
    eos_ids: set[int] = set()
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_ids.add(eos_token_id)
    elif isinstance(eos_token_id, list):
        eos_ids.update(eos_token_id)
    return eos_ids


def _resolve_finish_reason(
    generated_ids: list[int],
    eos_token_ids: set[int],
    max_new_tokens: int,
) -> Literal["stop", "length"]:
    if generated_ids and eos_token_ids and generated_ids[-1] in eos_token_ids:
        return "stop"
    if len(generated_ids) >= max_new_tokens:
        return "length"
    return "stop"


def _build_stats(
    *,
    prompt_tokens: int,
    generation_tokens: int,
    generation_seconds: float,
    peak_memory_bytes: int,
) -> GenerationStats:
    prompt_tps = prompt_tokens / generation_seconds if generation_seconds > 0 else 0.0
    generation_tps = (
        generation_tokens / generation_seconds if generation_seconds > 0 else 0.0
    )
    return GenerationStats(
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        peak_memory_usage=Memory.from_bytes(peak_memory_bytes),
    )


def _load_model_and_tokenizer(
    shard_metadata: ShardMetadata,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CudaSingle runner.")

    device = torch.device("cuda")
    model_path = build_model_path(shard_metadata.model_card.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _generate_token_chunks(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    shard_metadata: ShardMetadata,
    task: ChatCompletion,
    event_sender: MpSender[Event],
) -> None:
    task_params = task.task_params
    if task_params.tools:
        logger.warning("CUDA runner does not support tool calling; ignoring tools.")

    prompt = _build_prompt(tokenizer, task_params.messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    max_new_tokens = task_params.max_tokens or DEFAULT_MAX_NEW_TOKENS
    temperature = task_params.temperature if task_params.temperature is not None else 0.7
    top_p = task_params.top_p if task_params.top_p is not None else 1.0
    do_sample = temperature > 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    generation_start = time.perf_counter()
    with torch.inference_mode():
        output: GenerateOutput = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_end = time.perf_counter()

    sequences = output.sequences
    generated_ids = sequences[0, input_ids.shape[-1] :].tolist()
    eos_token_ids = _resolve_eos_token_ids(tokenizer)
    finish_reason = _resolve_finish_reason(
        generated_ids, eos_token_ids, max_new_tokens
    )

    trimmed_ids = generated_ids
    if trimmed_ids and eos_token_ids and trimmed_ids[-1] in eos_token_ids:
        trimmed_ids = trimmed_ids[:-1]

    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    stats = _build_stats(
        prompt_tokens=int(input_ids.numel()),
        generation_tokens=len(trimmed_ids),
        generation_seconds=generation_end - generation_start,
        peak_memory_bytes=peak_bytes,
    )

    if not trimmed_ids:
        event_sender.send(
            ChunkGenerated(
                command_id=task.command_id,
                chunk=TokenChunk(
                    model=shard_metadata.model_card.model_id,
                    text="",
                    token_id=tokenizer.eos_token_id or 0,
                    finish_reason=finish_reason,
                    stats=stats,
                ),
            )
        )
        return

    for idx, token_id in enumerate(trimmed_ids):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        is_last = idx == len(trimmed_ids) - 1
        event_sender.send(
            ChunkGenerated(
                command_id=task.command_id,
                chunk=TokenChunk(
                    model=shard_metadata.model_card.model_id,
                    text=text,
                    token_id=token_id,
                    finish_reason=finish_reason if is_last else None,
                    stats=stats if is_last else None,
                ),
            )
        )


def _warmup_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> None:
    inputs = tokenizer("Hello", return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
):
    shard_metadata = bound_instance.bound_shard
    device_rank = shard_metadata.device_rank
    logger.info("hello from the CUDA runner")

    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    device: torch.device | None = None

    current_status: RunnerStatus = RunnerIdle()
    event_sender.send(
        RunnerStatusUpdated(
            runner_id=bound_instance.bound_runner_id, runner_status=current_status
        )
    )

    with task_receiver as tasks:
        for task in tasks:
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
            )
            event_sender.send(TaskAcknowledged(task_id=task.task_id))
            match task:
                case ConnectToGroup():
                    current_status = RunnerFailed(
                        error_message="CUDA runner does not support distributed groups."
                    )
                case LoadModel() if isinstance(current_status, RunnerIdle):
                    current_status = RunnerLoading()
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=bound_instance.bound_runner_id,
                            runner_status=current_status,
                        )
                    )
                    if ModelTask.TextGeneration not in shard_metadata.model_card.tasks:
                        raise ValueError(
                            f"CUDA runner only supports text generation, got {shard_metadata.model_card.tasks}"
                        )
                    model, tokenizer, device = _load_model_and_tokenizer(shard_metadata)
                    current_status = RunnerLoaded()
                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    assert model and tokenizer and device
                    current_status = RunnerWarmingUp()
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=bound_instance.bound_runner_id,
                            runner_status=current_status,
                        )
                    )
                    _warmup_model(model, tokenizer, device)
                    current_status = RunnerReady()
                case ChatCompletion() if isinstance(current_status, RunnerReady):
                    assert model and tokenizer and device
                    current_status = RunnerRunning()
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=bound_instance.bound_runner_id,
                            runner_status=current_status,
                        )
                    )
                    try:
                        _generate_token_chunks(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            shard_metadata=shard_metadata,
                            task=task,
                            event_sender=event_sender,
                        )
                    except Exception as exc:
                        if device_rank == 0:
                            event_sender.send(
                                ChunkGenerated(
                                    command_id=task.command_id,
                                    chunk=ErrorChunk(
                                        model=shard_metadata.model_card.model_id,
                                        error_message=str(exc),
                                    ),
                                )
                            )
                        raise
                    current_status = RunnerReady()
                case ImageGeneration() | ImageEdits():
                    if device_rank == 0:
                        event_sender.send(
                            ChunkGenerated(
                                command_id=task.command_id,
                                chunk=ErrorChunk(
                                    model=shard_metadata.model_card.model_id,
                                    error_message="CUDA runner does not support image tasks.",
                                ),
                            )
                        )
                    current_status = RunnerReady()
                case Shutdown():
                    current_status = RunnerShuttingDown()
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=bound_instance.bound_runner_id,
                            runner_status=current_status,
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
                RunnerStatusUpdated(
                    runner_id=bound_instance.bound_runner_id,
                    runner_status=current_status,
                )
            )
            if isinstance(current_status, RunnerShutdown):
                if device is not None and device.type == "cuda":
                    torch.cuda.empty_cache()
                break
