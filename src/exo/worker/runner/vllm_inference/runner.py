import contextlib
import gc
import json
import os
import re
import time
from enum import Enum
from functools import cache
from typing import Any, cast

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
import vllm
from anyio import WouldBlock
from openai_harmony import (
    HarmonyEncoding,
    HarmonyEncodingName,
    HarmonyError,
    Role,
    StreamableParser,
    load_harmony_encoding,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine

from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.types.api import (
    CompletionTokensDetails,
    PromptTokensDetails,
    ToolCallItem,
    Usage,
)
from exo.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId
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
from exo.vllm_patches.growable_cache import patch_vllm
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.tool_parsers import ToolParser, infer_tool_parser


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Shutdown = "Shutdown"


class ModelKind(str, Enum):
    GptOss = "gpt_oss"
    Deepseek = "deepseek"
    Generic = "generic"


DSML_TOKEN = "\uff5cDSML\uff5c"
DSML_TOOL_CALLS_START = f"<{DSML_TOKEN}function_calls>"
DSML_TOOL_CALLS_END = f"</{DSML_TOKEN}function_calls>"

_DSML_INVOKE_PATTERN = re.compile(
    rf"<{re.escape(DSML_TOKEN)}invoke\s+name=\"([^\"]+)\">"
    rf"(.*?)"
    rf"</{re.escape(DSML_TOKEN)}invoke>",
    re.DOTALL,
)

_DSML_PARAM_PATTERN = re.compile(
    rf"<{re.escape(DSML_TOKEN)}parameter\s+name=\"([^\"]+)\"\s+string=\"(true|false)\">"
    rf"(.*?)"
    rf"</{re.escape(DSML_TOKEN)}parameter>",
    re.DOTALL,
)

_LOSSY_TEMPLATE_PATTERN = re.compile(
    r"""inner_type\s*==\s*["']object \| object["']\s*or\s*inner_type\|length\s*>\s*\d+""",
)


def _parse_dsml_output(text: str) -> list[ToolCallItem] | None:
    tool_calls: list[ToolCallItem] = []
    for invoke_match in _DSML_INVOKE_PATTERN.finditer(text):
        func_name = invoke_match.group(1)
        invoke_body = invoke_match.group(2)
        args: dict[str, object] = {}
        for param_match in _DSML_PARAM_PATTERN.finditer(invoke_body):
            param_name = param_match.group(1)
            is_string = param_match.group(2) == "true"
            param_value = param_match.group(3)
            if is_string:
                args[param_name] = param_value
            else:
                try:
                    args[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    args[param_name] = param_value
        tool_calls.append(ToolCallItem(name=func_name, arguments=json.dumps(args)))
    return tool_calls if tool_calls else None


def _could_be_dsml_prefix(text: str) -> bool:
    max_check = len(DSML_TOOL_CALLS_START)
    tail = text[-max_check:] if len(text) > max_check else text
    for i in range(len(tail)):
        suffix = tail[i:]
        if DSML_TOOL_CALLS_START.startswith(suffix):
            return True
    return False


def _patch_lossy_chat_template(template: str) -> str | None:
    patched, n = _LOSSY_TEMPLATE_PATTERN.subn(
        lambda m: m.group(0).split(" or ")[0],
        template,
    )
    return patched if n > 0 else None


def _collect_nested_property_names(schema: dict[str, object]) -> set[str]:
    names: set[str] = set()
    properties_raw = schema.get("properties", {})
    if not isinstance(properties_raw, dict):
        return names
    properties = cast(dict[str, object], properties_raw)
    for prop_spec_raw in properties.values():
        if not isinstance(prop_spec_raw, dict):
            continue
        prop_spec = cast(dict[str, object], prop_spec_raw)
        if prop_spec.get("type") == "array":
            items_raw: object = prop_spec.get("items")
            if isinstance(items_raw, dict):
                items = cast(dict[str, object], items_raw)
                if items.get("type") == "object":
                    inner_props_raw: object = items.get("properties", {})
                    if isinstance(inner_props_raw, dict):
                        inner_props = cast(dict[str, object], inner_props_raw)
                        for k in inner_props:
                            names.add(str(k))
                        names.update(_collect_nested_property_names(items))
    return names


def _schemas_lost_in_prompt(prompt: str, tools: list[dict[str, Any]]) -> bool:
    for tool in tools:
        fn_raw = cast(object, tool.get("function", {}))
        if not isinstance(fn_raw, dict):
            continue
        fn = cast(dict[str, object], fn_raw)
        params_raw: object = fn.get("parameters", {})
        if not isinstance(params_raw, dict):
            continue
        params = cast(dict[str, object], params_raw)
        nested = _collect_nested_property_names(params)
        if nested and not all(name in prompt for name in nested):
            return True
    return False


def _normalize_tool_calls(msg_dict: dict[str, object]) -> None:
    tool_calls_raw: object = msg_dict.get("tool_calls")
    if not tool_calls_raw or not isinstance(tool_calls_raw, list):
        return
    tool_calls_list = cast(list[object], tool_calls_raw)
    for tc_raw in tool_calls_list:
        if not isinstance(tc_raw, dict):
            continue
        tc = cast(dict[str, object], tc_raw)
        func_raw: object = tc.get("function")
        if not isinstance(func_raw, dict):
            continue
        func = cast(dict[str, object], func_raw)
        args_raw: object = func.get("arguments")
        if isinstance(args_raw, str):
            with contextlib.suppress(json.JSONDecodeError):
                parsed: object = cast(object, json.loads(args_raw))
                func["arguments"] = parsed


@cache
def _get_gpt_oss_encoding() -> HarmonyEncoding:
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _detect_model_kind(model_id: str) -> ModelKind:
    lower = model_id.lower()
    if "gpt-oss" in lower or "gpt_oss" in lower:
        return ModelKind.GptOss
    if "deepseek" in lower:
        return ModelKind.Deepseek
    return ModelKind.Generic


def _make_usage(prompt_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetails(),
        completion_tokens_details=CompletionTokensDetails(),
    )


def _check_vllm_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available \u2014 vLLM requires a CUDA GPU")
    logger.info(
        f"vLLM pre-flight: vllm {vllm.__version__}, "
        f"torch {torch.__version__}, "
        f"CUDA {torch.version.cuda}, "
        f"GPU {torch.cuda.get_device_name(0)}"
    )


class Runner:
    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance

        self.instance, self.runner_id, self.shard_metadata = (
            self.bound_instance.instance,
            self.bound_instance.bound_runner_id,
            self.bound_instance.bound_shard,
        )
        self.model_id = self.shard_metadata.model_card.model_id
        self.model_path = EXO_MODELS_DIR / self.model_id.normalize()

        self.engine: LLMEngine | None = None
        self.tool_parser: ToolParser | None = None
        self.model_kind: ModelKind = ModelKind.Generic
        self.prompt_token_count: int = 0

        self.seen: set[TaskId] = set()
        self.active_tasks: dict[TaskId, TextGeneration] = {}

        logger.info("hello from the vllm runner")
        _check_vllm_available()
        self.setup_start_time = time.time()
        self.update_status(RunnerIdle())

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task_id: TaskId, task_status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=task_status)
        )

    def acknowledge_task(self, task: Task):
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self):
        with self.task_receiver:
            for task in self.task_receiver:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)
                self.handle_first_task(task)
                if isinstance(self.current_status, RunnerShutdown):
                    break

    def handle_first_task(self, task: Task):
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(
                self.current_status, (RunnerIdle, RunnerFailed)
            ):
                logger.info("vllm runner connecting (no-op)")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())

            case LoadModel() if isinstance(
                self.current_status, (RunnerConnected, RunnerIdle)
            ):
                logger.info("vllm runner loading model")
                self.update_status(RunnerLoading(layers_loaded=0, total_layers=1))
                self.acknowledge_task(task)
                self._load_model()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                logger.info("vllm runner warming up")
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerReady())
                logger.info(
                    f"vllm runner ready in {time.time() - self.setup_start_time:.1f}s"
                )

            case TextGeneration() if isinstance(self.current_status, RunnerReady):
                return_code = self.handle_generation_tasks(starting_task=task)
                if return_code == ExitCode.Shutdown:
                    return

            case Shutdown():
                self.shutdown(task)
                return

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                )

    def _load_model(self):
        patch_vllm()

        engine_args = EngineArgs(
            model=str(self.model_path),
            served_model_name=str(self.model_id),
            gpu_memory_utilization=0.05,
            trust_remote_code=self.shard_metadata.model_card.trust_remote_code,
        )

        self.engine = LLMEngine.from_engine_args(engine_args)
        self.model_kind = _detect_model_kind(str(self.model_id))

        tokenizer = self.engine.get_tokenizer()
        chat_template = getattr(tokenizer, "chat_template", None)
        if isinstance(chat_template, str):
            self.tool_parser = infer_tool_parser(chat_template)
            if self.tool_parser:
                logger.info(
                    f"inferred tool parser: {self.tool_parser.start_parsing} / {self.tool_parser.end_parsing}"
                )

        logger.info(
            f"vLLM engine loaded for {self.model_id} (kind={self.model_kind.value})"
        )

    def _format_prompt(self, params: TextGenerationTaskParams) -> str:
        assert self.engine is not None
        tokenizer = self.engine.get_tokenizer()

        if params.chat_template_messages is not None:
            formatted_messages: list[dict[str, Any]] = list(
                params.chat_template_messages
            )
            for msg in formatted_messages:
                _normalize_tool_calls(msg)
        else:
            formatted_messages = []
            if params.instructions:
                formatted_messages.append(
                    {"role": "system", "content": params.instructions}
                )
            for msg in params.input:
                if msg.content:
                    formatted_messages.append(
                        {"role": msg.role, "content": msg.content}
                    )

        partial_assistant_content: str | None = None
        if formatted_messages and formatted_messages[-1].get("role") == "assistant":
            last_content = cast(object, formatted_messages[-1].get("content", ""))
            partial_assistant_content = str(last_content)
            formatted_messages = formatted_messages[:-1]

        extra_kwargs: dict[str, bool | str] = {}
        if params.enable_thinking is not None:
            extra_kwargs["enable_thinking"] = params.enable_thinking
            extra_kwargs["thinking"] = params.enable_thinking
        if params.reasoning_effort is not None:
            extra_kwargs["reasoning_effort"] = params.reasoning_effort

        patched_template: str | None = None
        chat_template = getattr(tokenizer, "chat_template", None)
        if params.tools and isinstance(chat_template, str):
            patched_template = _patch_lossy_chat_template(chat_template)
            if patched_template is not None:
                logger.info(
                    "Patched lossy chat template (removed inner_type length guard)"
                )

        result = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=params.tools,
            **(
                {"chat_template": patched_template}
                if patched_template is not None
                else {}
            ),
            **extra_kwargs,
        )
        assert isinstance(result, str)

        if params.tools and _schemas_lost_in_prompt(result, params.tools):
            logger.warning("Chat template lost nested tool schemas even after patching")

        if partial_assistant_content:
            result += partial_assistant_content

        logger.info(result)

        self.prompt_token_count = len(tokenizer.encode(result))

        return result

    def _make_sampling_params(self, params: TextGenerationTaskParams) -> SamplingParams:
        kwargs: dict[str, object] = {}
        if params.max_output_tokens is not None:
            kwargs["max_tokens"] = params.max_output_tokens
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.top_k is not None:
            kwargs["top_k"] = params.top_k
        if params.min_p is not None:
            kwargs["min_p"] = params.min_p
        if params.stop is not None:
            kwargs["stop"] = params.stop
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.repetition_penalty is not None:
            kwargs["repetition_penalty"] = params.repetition_penalty
        if params.logprobs:
            kwargs["logprobs"] = params.top_logprobs or 1
        return SamplingParams(**kwargs)

    def handle_generation_tasks(self, starting_task: TextGeneration):
        self.update_status(RunnerRunning())
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)
        self.active_tasks[starting_task.task_id] = starting_task

        self._stream_generation(starting_task)
        self.active_tasks.pop(starting_task.task_id, None)

        while True:
            try:
                task = self.task_receiver.receive_nowait()
                if task.task_id in self.seen:
                    continue
                self.seen.add(task.task_id)

                match task:
                    case TextGeneration():
                        self.acknowledge_task(task)
                        self.active_tasks[task.task_id] = task
                        self._stream_generation(task)
                        self.active_tasks.pop(task.task_id, None)
                    case Shutdown():
                        self.shutdown(task)
                        return ExitCode.Shutdown
                    case _:
                        raise ValueError(f"Unexpected task {task.__class__.__name__}")
            except WouldBlock:
                break

        self.update_status(RunnerReady())
        return ExitCode.AllTasksComplete

    def _send_token_chunk(
        self,
        command_id: CommandId,
        text: str,
        is_thinking: bool,
        finish_reason: str | None,
        usage: Usage | None = None,
    ):
        mapped_reason = (
            finish_reason
            if finish_reason in ("stop", "length", "content_filter")
            else None
        )
        self.event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=TokenChunk(
                    model=self.model_id,
                    text=text,
                    token_id=0,
                    usage=usage,
                    finish_reason=mapped_reason,
                    is_thinking=is_thinking,
                ),
            )
        )

    def _send_tool_call_chunk(
        self,
        command_id: CommandId,
        tool_calls: list[ToolCallItem],
        usage: Usage | None = None,
    ):
        self.event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=ToolCallChunk(
                    tool_calls=tool_calls,
                    model=self.model_id,
                    usage=usage,
                ),
            )
        )

    def _stream_generation(self, task: TextGeneration):
        assert self.engine is not None
        params = task.task_params

        try:
            prompt = self._format_prompt(params)
            sampling_params = self._make_sampling_params(params)
            request_id = str(task.task_id)

            self.engine.add_request(request_id, prompt, sampling_params)

            match self.model_kind:
                case ModelKind.GptOss:
                    self._generate_gpt_oss(task, request_id)
                case ModelKind.Deepseek:
                    self._generate_deepseek(task, request_id, prompt)
                case ModelKind.Generic:
                    self._generate_generic(task, request_id, prompt)

            self.send_task_status(task.task_id, TaskStatus.Complete)
        except Exception as e:
            logger.opt(exception=e).error("vLLM generation failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        error_message=str(e),
                        model=self.model_id,
                    ),
                )
            )
            self.send_task_status(task.task_id, TaskStatus.Complete)

    def _generate_gpt_oss(self, task: TextGeneration, request_id: str):
        assert self.engine is not None
        encoding = _get_gpt_oss_encoding()
        stream = StreamableParser(encoding, role=Role.ASSISTANT)
        thinking = False
        current_tool_name: str | None = None
        tool_arg_parts: list[str] = []
        prev_token_count = 0

        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for output in outputs:
                if output.request_id != request_id:
                    continue
                completion = output.outputs[0]
                new_tokens = completion.token_ids[prev_token_count:]
                prev_token_count = len(completion.token_ids)
                finish_reason = completion.finish_reason

                for token_id in new_tokens:
                    try:
                        stream.process(token_id)
                    except HarmonyError:
                        logger.error("Harmony encoding error, stopping generation")
                        return

                    delta = stream.last_content_delta
                    channel = stream.current_channel
                    recipient = stream.current_recipient

                    if recipient != current_tool_name:
                        if current_tool_name is not None:
                            name = current_tool_name.removeprefix("functions.")
                            usage = _make_usage(
                                self.prompt_token_count, prev_token_count
                            )
                            self._send_tool_call_chunk(
                                task.command_id,
                                [
                                    ToolCallItem(
                                        name=name,
                                        arguments="".join(tool_arg_parts).strip(),
                                    )
                                ],
                                usage,
                            )
                            tool_arg_parts = []
                        current_tool_name = recipient

                    if current_tool_name is not None:
                        if delta:
                            tool_arg_parts.append(delta)
                        continue

                    if channel == "analysis" and not thinking:
                        thinking = True
                    if channel != "analysis" and thinking:
                        thinking = False

                    if delta:
                        self._send_token_chunk(task.command_id, delta, thinking, None)

                if finish_reason:
                    if current_tool_name is not None and tool_arg_parts:
                        name = current_tool_name.removeprefix("functions.")
                        usage = _make_usage(self.prompt_token_count, prev_token_count)
                        self._send_tool_call_chunk(
                            task.command_id,
                            [
                                ToolCallItem(
                                    name=name, arguments="".join(tool_arg_parts).strip()
                                )
                            ],
                            usage,
                        )
                        tool_arg_parts = []
                        current_tool_name = None
                    usage = _make_usage(self.prompt_token_count, prev_token_count)
                    self._send_token_chunk(
                        task.command_id, "", False, finish_reason, usage
                    )

    def _generate_deepseek(self, task: TextGeneration, request_id: str, prompt: str):
        assert self.engine is not None
        is_thinking = prompt.rstrip().endswith("<think>")
        accumulated = ""
        in_tool_call = False
        tool_call_text = ""
        pending_deltas: list[str] = []
        prev_text = ""
        prev_token_count = 0

        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for output in outputs:
                if output.request_id != request_id:
                    continue
                completion = output.outputs[0]
                delta = completion.text[len(prev_text) :]
                prev_text = completion.text
                prev_token_count = len(completion.token_ids)
                finish_reason = completion.finish_reason

                if not delta and not finish_reason:
                    continue

                if not is_thinking and "<think>" in delta:
                    is_thinking = True
                    before = delta[: delta.index("<think>")]
                    if before:
                        self._send_token_chunk(task.command_id, before, False, None)
                    continue

                if is_thinking and "</think>" in delta:
                    is_thinking = False
                    after = delta[delta.index("</think>") + len("</think>") :]
                    if after:
                        self._send_token_chunk(task.command_id, after, False, None)
                    continue

                if is_thinking:
                    if delta:
                        self._send_token_chunk(task.command_id, delta, True, None)
                    if finish_reason:
                        usage = _make_usage(self.prompt_token_count, prev_token_count)
                        self._send_token_chunk(
                            task.command_id, "", False, finish_reason, usage
                        )
                    continue

                if in_tool_call:
                    tool_call_text += delta
                    if DSML_TOOL_CALLS_END in tool_call_text:
                        parsed = _parse_dsml_output(tool_call_text)
                        if parsed is not None:
                            usage = _make_usage(
                                self.prompt_token_count, prev_token_count
                            )
                            self._send_tool_call_chunk(task.command_id, parsed, usage)
                        else:
                            self._send_token_chunk(
                                task.command_id, tool_call_text, False, None
                            )
                        in_tool_call = False
                        tool_call_text = ""
                        continue
                    if finish_reason:
                        self._send_token_chunk(
                            task.command_id, tool_call_text, False, finish_reason
                        )
                        in_tool_call = False
                        tool_call_text = ""
                    continue

                accumulated += delta

                if DSML_TOOL_CALLS_START in accumulated:
                    start_idx = accumulated.index(DSML_TOOL_CALLS_START)
                    pre_text = accumulated[:start_idx]
                    if pre_text:
                        for pd in pending_deltas:
                            if pre_text and len(pd) <= len(pre_text):
                                self._send_token_chunk(task.command_id, pd, False, None)
                                pre_text = pre_text[len(pd) :]
                            elif pre_text:
                                self._send_token_chunk(
                                    task.command_id, pre_text, False, None
                                )
                                pre_text = ""
                    pending_deltas = []
                    tool_call_text = accumulated[start_idx:]
                    accumulated = ""

                    if DSML_TOOL_CALLS_END in tool_call_text:
                        parsed = _parse_dsml_output(tool_call_text)
                        if parsed is not None:
                            usage = _make_usage(
                                self.prompt_token_count, prev_token_count
                            )
                            self._send_tool_call_chunk(task.command_id, parsed, usage)
                        else:
                            self._send_token_chunk(
                                task.command_id, tool_call_text, False, None
                            )
                        tool_call_text = ""
                    else:
                        in_tool_call = True
                    continue

                if _could_be_dsml_prefix(accumulated):
                    pending_deltas.append(delta)
                    continue

                for pd in pending_deltas:
                    self._send_token_chunk(task.command_id, pd, False, None)
                pending_deltas = []
                accumulated = ""

                if delta:
                    self._send_token_chunk(task.command_id, delta, False, None)

                if finish_reason:
                    usage = _make_usage(self.prompt_token_count, prev_token_count)
                    self._send_token_chunk(
                        task.command_id, "", False, finish_reason, usage
                    )

        for pd in pending_deltas:
            self._send_token_chunk(task.command_id, pd, False, None)

    def _generate_generic(self, task: TextGeneration, request_id: str, prompt: str):
        assert self.engine is not None
        is_thinking = prompt.rstrip().endswith("<think>")
        in_tool_call = False
        tool_call_parts: list[str] = []
        prev_text = ""
        prev_token_count = 0

        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for output in outputs:
                if output.request_id != request_id:
                    continue
                completion = output.outputs[0]
                delta = completion.text[len(prev_text) :]
                prev_text = completion.text
                prev_token_count = len(completion.token_ids)
                finish_reason = completion.finish_reason

                if not delta and not finish_reason:
                    continue

                if delta == "<think>":
                    is_thinking = True
                    continue
                if delta == "</think>":
                    is_thinking = False
                    continue

                if (
                    self.tool_parser
                    and not in_tool_call
                    and delta.startswith(self.tool_parser.start_parsing)
                ):
                    in_tool_call = True

                if in_tool_call:
                    assert self.tool_parser is not None
                    tool_call_parts.append(delta)
                    if delta.endswith(self.tool_parser.end_parsing):
                        combined = "".join(tool_call_parts)
                        parsed = self.tool_parser.parse(
                            combined.strip(), tools=task.task_params.tools
                        )
                        in_tool_call = False
                        tool_call_parts = []
                        if parsed:
                            usage = _make_usage(
                                self.prompt_token_count, prev_token_count
                            )
                            self._send_tool_call_chunk(task.command_id, parsed, usage)
                        else:
                            self._send_token_chunk(
                                task.command_id, combined, False, finish_reason
                            )
                    elif finish_reason:
                        combined = "".join(tool_call_parts)
                        in_tool_call = False
                        tool_call_parts = []
                        self._send_token_chunk(
                            task.command_id, combined, False, finish_reason
                        )
                    continue

                if finish_reason:
                    usage = _make_usage(self.prompt_token_count, prev_token_count)
                    self._send_token_chunk(
                        task.command_id,
                        delta or "",
                        is_thinking if delta else False,
                        finish_reason,
                        usage,
                    )
                elif delta:
                    self._send_token_chunk(task.command_id, delta, is_thinking, None)

    def shutdown(self, task: Task):
        logger.info("vllm runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        del self.engine
        self.engine = None
        gc.collect()
        torch.cuda.empty_cache()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())
