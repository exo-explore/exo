import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from exo.download.download_utils import resolve_local_gguf_path
from exo.shared.models.model_cards import ModelId
from exo.shared.types.api import ChatCompletionMessage
from exo.shared.types.chunks import ErrorChunk, TokenChunk
from exo.shared.types.events import (
    ChunkGenerated,
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
from exo.shared.types.worker.instances import BoundInstance, LlamaRpcInstance
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


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender,
    task_receiver: MpReceiver,
) -> None:
    if not isinstance(bound_instance.instance, LlamaRpcInstance):
        raise ValueError("llama_cpp_runner only supports LlamaRpcInstance")

    instance = bound_instance.instance
    shard_metadata = bound_instance.bound_shard
    runner_id = bound_instance.bound_runner_id

    is_primary = bound_instance.bound_node_id == instance.primary_node_id
    server_process: subprocess.Popen[str] | None = None

    current_status: RunnerStatus = RunnerIdle()
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )

    with task_receiver as tasks:
        for task in tasks:
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
            )
            event_sender.send(TaskAcknowledged(task_id=task.task_id))

            try:
                match task:
                    case ConnectToGroup() if isinstance(
                        current_status, (RunnerIdle, RunnerFailed)
                    ):
                        current_status = RunnerConnecting()
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        current_status = RunnerConnected()
                    case LoadModel() if isinstance(
                        current_status, (RunnerConnected, RunnerIdle)
                    ):
                        current_status = RunnerLoading()
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        server_process = _ensure_llama_server(
                            instance=instance,
                            shard_metadata=shard_metadata,
                            is_primary=is_primary,
                        )
                        current_status = RunnerLoaded()
                    case StartWarmup() if isinstance(current_status, RunnerLoaded):
                        current_status = RunnerWarmingUp()
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        if is_primary:
                            _wait_for_server(instance.http_port)
                        current_status = RunnerReady()
                    case ChatCompletion(
                        task_params=task_params, command_id=command_id
                    ) if isinstance(current_status, RunnerReady):
                        current_status = RunnerRunning()
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        if is_primary:
                            _stream_chat_completion(
                                event_sender=event_sender,
                                command_id=command_id,
                                model_id=shard_metadata.model_card.model_id,
                                http_port=instance.http_port,
                                messages=task_params.messages,
                                params=task_params.model_dump(),
                            )
                        current_status = RunnerReady()
                    case Shutdown():
                        current_status = RunnerShuttingDown()
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        if server_process is not None:
                            _stop_process(server_process)
                            server_process = None
                        current_status = RunnerShutdown()
                    case _:
                        raise ValueError(
                            f"Received {task.__class__.__name__} outside of state machine in {current_status=}"
                        )

            except Exception as e:
                logger.opt(exception=e).error("llama.cpp runner error")
                event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=runner_id,
                        runner_status=RunnerFailed(error_message=str(e)),
                    )
                )
                if isinstance(task, ChatCompletion) and is_primary:
                    event_sender.send(
                        ChunkGenerated(
                            command_id=task.command_id,
                            chunk=ErrorChunk(
                                model=shard_metadata.model_card.model_id,
                                finish_reason="error",
                                error_message=str(e),
                            ),
                        )
                    )
                current_status = RunnerFailed(error_message=str(e))

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


def _ensure_llama_server(
    *,
    instance: LlamaRpcInstance,
    shard_metadata: ShardMetadata,
    is_primary: bool,
) -> subprocess.Popen[str]:
    if is_primary:
        server_path = _resolve_llama_server_path()
        model_path = _resolve_gguf_model_path(shard_metadata.model_card.model_id)
        if model_path is None:
            raise ValueError(
                "No GGUF model found for llama.cpp backend. "
                "Provide a .gguf file path in the model id or place it in EXO_MODELS_DIR."
            )

        gpu_layers = os.getenv("EXO_LLAMA_GPU_LAYERS")
        if gpu_layers is None and sys.platform == "win32":
            gpu_layers = "-1"

        rpc_list = ",".join(
            str(host)
            for node_id, host in instance.rpc_hosts_by_node.items()
            if node_id != instance.primary_node_id
        )
        args_template = os.getenv(
            "EXO_LLAMA_SERVER_ARGS_TEMPLATE",
            "--model {model_path} --host {host} --port {http_port} --rpc {rpc_list} --n-gpu-layers {gpu_layers}",
        )
        if not rpc_list:
            args_template = args_template.replace("--rpc {rpc_list}", "")
        if gpu_layers is None:
            args_template = args_template.replace("--n-gpu-layers {gpu_layers}", "")
        args = _format_args(
            args_template,
            {
                "model_path": str(model_path),
                "host": "0.0.0.0",
                "http_port": str(instance.http_port),
                "rpc_list": rpc_list,
                "gpu_layers": gpu_layers or "",
            },
        )
        cmd = [server_path, *args]
        logger.info(f"Starting llama.cpp server: {' '.join(cmd)}")
        return subprocess.Popen(cmd)

    server_path = _resolve_rpc_server_path()
    args_template = os.getenv(
        "EXO_LLAMA_RPC_ARGS_TEMPLATE",
        "--host {host} --port {rpc_port}",
    )
    args = _format_args(
        args_template,
        {
            "host": "0.0.0.0",
            "rpc_port": str(instance.rpc_port),
        },
    )

    cmd = [server_path, *args]
    logger.info(f"Starting llama.cpp rpc server: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def _resolve_llama_server_path() -> str:
    configured = os.getenv("EXO_LLAMA_SERVER_PATH")
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return str(candidate)
        raise ValueError(
            "EXO_LLAMA_SERVER_PATH is set but the file was not found: "
            f"{configured}"
        )
    cwd_candidate = Path.cwd() / "llama-server.exe"
    if cwd_candidate.exists():
        return str(cwd_candidate)
    resolved = shutil.which("llama-server.exe") or shutil.which("llama-server")
    if resolved:
        return resolved
    raise ValueError(
        "llama.cpp server binary not found. "
        "Install llama.cpp and set EXO_LLAMA_SERVER_PATH to llama-server.exe."
    )


def _resolve_rpc_server_path() -> str:
    configured = os.getenv("EXO_LLAMA_RPC_SERVER_PATH")
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return str(candidate)
        raise ValueError(
            "EXO_LLAMA_RPC_SERVER_PATH is set but the file was not found: "
            f"{configured}"
        )
    cwd_candidate = Path.cwd() / "rpc-server.exe"
    if cwd_candidate.exists():
        return str(cwd_candidate)
    resolved = shutil.which("rpc-server.exe") or shutil.which("rpc-server")
    if resolved:
        return resolved
    raise ValueError(
        "llama.cpp rpc server binary not found. "
        "Install llama.cpp and set EXO_LLAMA_RPC_SERVER_PATH to rpc-server.exe."
    )


def _format_args(template: str, values: dict[str, str]) -> list[str]:
    filled = template.format(**values)
    parts = shlex.split(filled, posix=False)
    return [part for part in parts if part]


def _resolve_gguf_model_path(model_id: ModelId) -> Path | None:
    if gguf_path := resolve_local_gguf_path(model_id):
        if gguf_path.exists():
            return gguf_path
    return None


def _wait_for_server(http_port: int) -> None:
    url = f"http://127.0.0.1:{http_port}/v1/models"
    raw_timeout = os.getenv("EXO_LLAMA_SERVER_STARTUP_TIMEOUT_SECONDS")
    try:
        timeout_seconds = int(raw_timeout) if raw_timeout is not None else 0
    except ValueError:
        timeout_seconds = 0
    deadline = time.time() + timeout_seconds if timeout_seconds > 0 else None
    while deadline is None or time.time() < deadline:
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    return
        except Exception:
            time.sleep(0.5)


def _stream_chat_completion(
    *,
    event_sender: MpSender,
    command_id: str,
    model_id: ModelId,
    http_port: int,
    messages: list[ChatCompletionMessage],
    params: dict[str, Any],
) -> None:
    url = f"http://127.0.0.1:{http_port}/v1/chat/completions"

    payload: dict[str, Any] = {
        "model": params.get("model"),
        "messages": [m.model_dump(exclude_none=True) for m in messages],
        "stream": True,
    }
    for key in (
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "n",
    ):
        if params.get(key) is not None:
            payload[key] = params[key]

    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = parsed.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")
                text = delta.get("content") or ""
                if text:
                    event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=TokenChunk(
                                model=model_id,
                                text=text,
                                token_id=-1,
                                finish_reason=None,
                            ),
                        )
                    )
                if finish_reason is not None:
                    mapped_reason = (
                        finish_reason
                        if finish_reason in ("stop", "length", "content_filter")
                        else "stop"
                    )
                    event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=TokenChunk(
                                model=model_id,
                                text="",
                                token_id=-1,
                                finish_reason=mapped_reason,
                            ),
                        )
                    )


def _stop_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
