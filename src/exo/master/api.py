import asyncio
import os
import time
from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Any, Optional, cast

import anyio
from anyio import BrokenResourceError, create_task_group
from anyio.abc import TaskGroup
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from loguru import logger
from pydantic import BaseModel

from exo.master.placement import place_instance as get_instance_placements
from exo.shared.apply import apply
from exo.shared.election import ElectionMessage
from exo.shared.logging import InterceptLogger
from exo.shared.models.model_cards import MODEL_CARDS
from exo.shared.models.model_meta import get_model_meta
from exo.shared.types.api import (
    BenchChatCompletionResponse,
    BenchChatCompletionTaskParams,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
    CreateInstanceParams,
    CreateInstanceResponse,
    DeleteInstanceResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    GenerationStats,
    ModelList,
    ModelListModel,
    PlaceInstanceParams,
    PlacementPreview,
    PlacementPreviewResponse,
    StreamingChoiceResponse,
)
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import (
    ChatCompletion,
    Command,
    CreateInstance,
    DeleteInstance,
    ForwarderCommand,
    PlaceInstance,
    TaskFinished,
)
from exo.shared.types.common import CommandId, NodeId, SessionId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    ForwarderEvent,
    IndexedEvent,
)
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.state import State
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
)
from exo.shared.types.worker.shards import Sharding
from exo.utils.banner import print_startup_banner
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.dashboard_path import find_dashboard
from exo.utils.event_buffer import OrderedBuffer


class ExecuteRequest(BaseModel):
    """Request to execute a command."""

    command: list[str]
    cwd: Optional[str] = None
    env: Optional[dict[str, str]] = None


class ExecuteResponse(BaseModel):
    """Response from command execution."""

    exit_code: int
    stdout: str
    stderr: str


def chunk_to_response(
    chunk: TokenChunk, command_id: CommandId
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=ChatCompletionMessage(role="assistant", content=chunk.text),
                finish_reason=chunk.finish_reason,
            )
        ],
    )


async def resolve_model_meta(model_id: str) -> ModelMetadata:
    if model_id in MODEL_CARDS:
        model_card = MODEL_CARDS[model_id]
        return model_card.metadata
    else:
        return await get_model_meta(model_id)


class API:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        port: int,
        # Ideally this would be a MasterForwarderEvent but type system says no :(
        global_event_receiver: Receiver[ForwarderEvent],
        command_sender: Sender[ForwarderCommand],
        # This lets us pause the API if an election is running
        election_receiver: Receiver[ElectionMessage],
    ) -> None:
        self.state = State()
        self._event_log: list[Event] = []
        self.command_sender = command_sender
        self.global_event_receiver = global_event_receiver
        self.election_receiver = election_receiver
        self.event_buffer: OrderedBuffer[Event] = OrderedBuffer[Event]()
        self.node_id: NodeId = node_id
        self.session_id: SessionId = session_id
        self.last_completed_election: int = 0
        self.port = port

        self.paused: bool = False
        self.paused_ev: anyio.Event = anyio.Event()

        self.app = FastAPI()
        self._setup_exception_handlers()
        self._setup_cors()
        self._setup_routes()
        self._register_plugin_routes()

        self.app.mount(
            "/",
            StaticFiles(
                directory=find_dashboard(),
                html=True,
            ),
            name="dashboard",
        )

        self._chat_completion_queues: dict[CommandId, Sender[TokenChunk]] = {}
        self._tg: TaskGroup | None = None

    def reset(self, new_session_id: SessionId, result_clock: int):
        logger.info("Resetting API State")
        self.state = State()
        self.session_id = new_session_id
        self.event_buffer = OrderedBuffer[Event]()
        self._chat_completion_queues = {}
        self.unpause(result_clock)

    def unpause(self, result_clock: int):
        logger.info("Unpausing API")
        self.last_completed_election = result_clock
        self.paused = False
        self.paused_ev.set()
        self.paused_ev = anyio.Event()

    def _setup_exception_handlers(self) -> None:
        self.app.exception_handler(HTTPException)(self.http_exception_handler)

    async def http_exception_handler(
        self, _: Request, exc: HTTPException
    ) -> JSONResponse:
        err = ErrorResponse(
            error=ErrorInfo(
                message=exc.detail,
                type=HTTPStatus(exc.status_code).phrase,
                code=exc.status_code,
            )
        )
        return JSONResponse(err.model_dump(), status_code=exc.status_code)

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        self.app.get("/node_id")(lambda: self.node_id)
        self.app.post("/instance")(self.create_instance)
        self.app.post("/place_instance")(self.place_instance)
        self.app.get("/instance/placement")(self.get_placement)
        self.app.get("/instance/previews")(self.get_placement_previews)
        self.app.get("/instance/{instance_id}")(self.get_instance)
        self.app.delete("/instance/{instance_id}")(self.delete_instance)
        self.app.get("/models")(self.get_models)
        self.app.get("/v1/models")(self.get_models)
        self.app.post("/v1/chat/completions", response_model=None)(
            self.chat_completions
        )
        self.app.post("/bench/chat/completions")(self.bench_chat_completions)
        self.app.get("/state")(lambda: self.state)
        self.app.get("/events")(lambda: self._event_log)
        # Remote execution endpoint (used by exo-rsh for MPI)
        self.app.post("/execute")(self.execute)

    def _register_plugin_routes(self) -> None:
        """Register API routes from all loaded plugins."""
        import functools
        import inspect

        from exo.plugins.context import PluginContext
        from exo.plugins.registry import PluginRegistry

        registry = PluginRegistry.get()

        for method, path, handler, plugin in registry.get_all_api_routes():
            # Create a wrapper that injects the plugin context while preserving
            # the original function signature for FastAPI parameter extraction
            @functools.wraps(handler)
            async def wrapped_handler(  # pyright: ignore[reportAny]
                *args: Any,  # pyright: ignore[reportAny]
                _handler: Any = handler,  # pyright: ignore[reportAny]
                **kwargs: Any,  # pyright: ignore[reportAny]
            ) -> Any:
                context = PluginContext(
                    state=self.state,
                    send_command=self._send,
                    node_id=self.node_id,
                )
                # Pass context as first argument, then forward all other args
                return await _handler(context, *args, **kwargs)  # pyright: ignore[reportAny]

            # Modify the wrapper signature to match the original handler
            # but without the 'ctx' parameter (we inject it)
            orig_sig = inspect.signature(handler)
            params = list(orig_sig.parameters.values())
            # Remove the first parameter (ctx: PluginContext)
            if params and params[0].name in ("ctx", "context"):
                params = params[1:]
            wrapped_handler.__signature__ = orig_sig.replace(parameters=params)  # pyright: ignore[reportAttributeAccessIssue]

            # Register the route based on HTTP method
            if method == "get":
                self.app.get(path)(wrapped_handler)
            elif method == "post":
                self.app.post(path)(wrapped_handler)
            elif method == "delete":
                self.app.delete(path)(wrapped_handler)
            elif method == "put":
                self.app.put(path)(wrapped_handler)

            logger.debug(
                f"Registered plugin route: {method.upper()} {path} ({plugin.name})"
            )

    async def place_instance(self, payload: PlaceInstanceParams):
        command = PlaceInstance(
            model_meta=await resolve_model_meta(payload.model_id),
            sharding=payload.sharding,
            instance_meta=payload.instance_meta,
            min_nodes=payload.min_nodes,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_meta=command.model_meta,
        )

    async def create_instance(
        self, payload: CreateInstanceParams
    ) -> CreateInstanceResponse:
        instance = payload.instance
        model_meta = await resolve_model_meta(instance.shard_assignments.model_id)
        required_memory = model_meta.storage_size
        available_memory = self._calculate_total_available_memory()

        if required_memory > available_memory:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient memory to create instance. Required: {required_memory.in_gb:.1f}GB, Available: {available_memory.in_gb:.1f}GB",
            )

        command = CreateInstance(
            instance=instance,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_meta=model_meta,
        )

    async def get_placement(
        self,
        model_id: str,
        sharding: Sharding = Sharding.Pipeline,
        instance_meta: InstanceMeta = InstanceMeta.MlxRing,
        min_nodes: int = 1,
    ) -> Instance:
        model_meta = await resolve_model_meta(model_id)

        try:
            placements = get_instance_placements(
                PlaceInstance(
                    model_meta=model_meta,
                    sharding=sharding,
                    instance_meta=instance_meta,
                    min_nodes=min_nodes,
                ),
                node_memory=self.state.node_memory,
                node_network=self.state.node_network,
                topology=self.state.topology,
                current_instances=self.state.instances,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        current_ids = set(self.state.instances.keys())
        new_ids = [
            instance_id for instance_id in placements if instance_id not in current_ids
        ]
        if len(new_ids) != 1:
            raise HTTPException(
                status_code=500,
                detail="Expected exactly one new instance from placement",
            )

        return placements[new_ids[0]]

    async def get_placement_previews(
        self, model_id: ModelId
    ) -> PlacementPreviewResponse:
        seen: set[tuple[ModelId, Sharding, InstanceMeta, int]] = set()
        previews: list[PlacementPreview] = []
        if len(list(self.state.topology.list_nodes())) == 0:
            return PlacementPreviewResponse(previews=[])

        cards = [card for card in MODEL_CARDS.values() if card.short_id == model_id]
        if not cards:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        instance_combinations: list[tuple[Sharding, InstanceMeta, int]] = []
        for sharding in (Sharding.Pipeline, Sharding.Tensor):
            for instance_meta in (InstanceMeta.MlxRing, InstanceMeta.MlxJaccl):
                instance_combinations.extend(
                    [
                        (sharding, instance_meta, i)
                        for i in range(
                            1, len(list(self.state.topology.list_nodes())) + 1
                        )
                    ]
                )
        # TODO: PDD
        # instance_combinations.append((Sharding.PrefillDecodeDisaggregation, InstanceMeta.MlxRing, 1))

        for card in cards:
            model_meta = card.metadata
            for sharding, instance_meta, min_nodes in instance_combinations:
                try:
                    placements = get_instance_placements(
                        PlaceInstance(
                            model_meta=model_meta,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            min_nodes=min_nodes,
                        ),
                        node_memory=self.state.node_memory,
                        node_network=self.state.node_network,
                        topology=self.state.topology,
                        current_instances=self.state.instances,
                    )
                except ValueError as exc:
                    if (card.model_id, sharding, instance_meta, 0) not in seen:
                        previews.append(
                            PlacementPreview(
                                model_id=card.model_id,
                                sharding=sharding,
                                instance_meta=instance_meta,
                                instance=None,
                                error=str(exc),
                            )
                        )
                    seen.add((card.model_id, sharding, instance_meta, 0))
                    continue

                current_ids = set(self.state.instances.keys())
                new_instances = [
                    instance
                    for instance_id, instance in placements.items()
                    if instance_id not in current_ids
                ]

                if len(new_instances) != 1:
                    if (card.model_id, sharding, instance_meta, 0) not in seen:
                        previews.append(
                            PlacementPreview(
                                model_id=card.model_id,
                                sharding=sharding,
                                instance_meta=instance_meta,
                                instance=None,
                                error="Expected exactly one new instance from placement",
                            )
                        )
                    seen.add((card.model_id, sharding, instance_meta, 0))
                    continue

                instance = new_instances[0]
                shard_assignments = instance.shard_assignments
                node_ids = list(shard_assignments.node_to_runner.keys())

                memory_delta_by_node: dict[str, int] = {}
                if node_ids:
                    total_bytes = model_meta.storage_size.in_bytes
                    per_node = total_bytes // len(node_ids)
                    remainder = total_bytes % len(node_ids)
                    for index, node_id in enumerate(sorted(node_ids, key=str)):
                        extra = 1 if index < remainder else 0
                        memory_delta_by_node[str(node_id)] = per_node + extra

                if (
                    card.model_id,
                    sharding,
                    instance_meta,
                    len(node_ids),
                ) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=instance,
                            memory_delta_by_node=memory_delta_by_node or None,
                            error=None,
                        )
                    )
                seen.add((card.model_id, sharding, instance_meta, len(node_ids)))

        return PlacementPreviewResponse(previews=previews)

    def get_instance(self, instance_id: InstanceId) -> Instance:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        return self.state.instances[instance_id]

    async def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")

        command = DeleteInstance(
            instance_id=instance_id,
        )
        await self._send(command)
        return DeleteInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance_id,
        )

    async def _chat_chunk_stream(
        self, command_id: CommandId
    ) -> AsyncGenerator[TokenChunk, None]:
        """Yield `TokenChunk`s for a given command until completion."""

        try:
            self._chat_completion_queues[command_id], recv = channel[TokenChunk]()

            with recv as token_chunks:
                async for chunk in token_chunks:
                    yield chunk
                    if chunk.finish_reason is not None:
                        break

        except anyio.get_cancelled_exc_class():
            # TODO: TaskCancelled
            """
            self.command_sender.send_nowait(
                ForwarderCommand(origin=self.node_id, command=command)
            )
            """
            raise
        finally:
            command = TaskFinished(finished_command_id=command_id)
            await self._send(command)
            del self._chat_completion_queues[command_id]

    async def _generate_chat_stream(
        self, command_id: CommandId
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion stream as JSON strings."""

        async for chunk in self._chat_chunk_stream(command_id):
            if chunk.finish_reason == "error":
                error_response = ErrorResponse(
                    error=ErrorInfo(
                        message=chunk.error_message or "Internal server error",
                        type="InternalServerError",
                        code=500,
                    )
                )
                yield f"data: {error_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            chunk_response: ChatCompletionResponse = chunk_to_response(
                chunk, command_id
            )
            logger.debug(f"chunk_response: {chunk_response}")

            yield f"data: {chunk_response.model_dump_json()}\n\n"

            if chunk.finish_reason is not None:
                yield "data: [DONE]\n\n"

    async def _collect_chat_completion(
        self, command_id: CommandId
    ) -> ChatCompletionResponse:
        """Collect all token chunks for a chat completion and return a single response."""

        text_parts: list[str] = []
        model: str | None = None
        finish_reason: FinishReason | None = None

        async for chunk in self._chat_chunk_stream(command_id):
            if chunk.finish_reason == "error":
                raise HTTPException(
                    status_code=500,
                    detail=chunk.error_message or "Internal server error",
                )

            if model is None:
                model = chunk.model

            text_parts.append(chunk.text)

            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason

        combined_text = "".join(text_parts)
        assert model is not None

        return ChatCompletionResponse(
            id=command_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=combined_text,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )

    async def _collect_chat_completion_with_stats(
        self, command_id: CommandId
    ) -> BenchChatCompletionResponse:
        text_parts: list[str] = []
        model: str | None = None
        finish_reason: FinishReason | None = None

        stats: GenerationStats | None = None

        async for chunk in self._chat_chunk_stream(command_id):
            if chunk.finish_reason == "error":
                raise HTTPException(
                    status_code=500,
                    detail=chunk.error_message or "Internal server error",
                )

            if model is None:
                model = chunk.model

            text_parts.append(chunk.text)
            stats = chunk.stats or stats

            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason

        combined_text = "".join(text_parts)
        assert model is not None

        resp = BenchChatCompletionResponse(
            id=command_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content=combined_text
                    ),
                    finish_reason=finish_reason,
                )
            ],
            generation_stats=stats,
        )
        return resp

    async def _trigger_notify_user_to_download_model(self, model_id: str) -> None:
        logger.warning(
            "TODO: we should send a notification to the user to download the model"
        )

    async def chat_completions(
        self, payload: ChatCompletionTaskParams
    ) -> ChatCompletionResponse | StreamingResponse:
        """Handle chat completions, supporting both streaming and non-streaming responses."""
        model_meta = await resolve_model_meta(payload.model)
        payload.model = model_meta.model_id

        if not any(
            instance.shard_assignments.model_id == payload.model
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(payload.model)
            raise HTTPException(
                status_code=404, detail=f"No instance found for model {payload.model}"
            )

        command = ChatCompletion(
            request_params=payload,
        )
        await self._send(command)
        if payload.stream:
            return StreamingResponse(
                self._generate_chat_stream(command.command_id),
                media_type="text/event-stream",
            )

        return await self._collect_chat_completion(command.command_id)

    async def bench_chat_completions(
        self, payload: BenchChatCompletionTaskParams
    ) -> BenchChatCompletionResponse:
        model_meta = await resolve_model_meta(payload.model)
        payload.model = model_meta.model_id

        if not any(
            instance.shard_assignments.model_id == payload.model
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(payload.model)
            raise HTTPException(
                status_code=404, detail=f"No instance found for model {payload.model}"
            )

        payload.stream = False

        command = ChatCompletion(request_params=payload)
        await self._send(command)

        response = await self._collect_chat_completion_with_stats(command.command_id)
        return response

    def _calculate_total_available_memory(self) -> Memory:
        """Calculate total available memory across all nodes in bytes."""
        total_available = Memory()

        for memory in self.state.node_memory.values():
            total_available += memory.ram_available

        return total_available

    async def get_models(self) -> ModelList:
        """Returns list of available models."""
        return ModelList(
            data=[
                ModelListModel(
                    id=card.short_id,
                    hugging_face_id=card.model_id,
                    name=card.name,
                    description=card.description,
                    tags=card.tags,
                    storage_size_megabytes=int(card.metadata.storage_size.in_mb),
                    supports_tensor=card.metadata.supports_tensor,
                )
                for card in MODEL_CARDS.values()
            ]
        )

    async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        """Execute a command locally. Used by exo-rsh for MPI remote execution."""
        cmd_str = " ".join(request.command)
        logger.info(f"Executing: {cmd_str}")

        try:
            # Build environment
            env = os.environ.copy()
            if request.env:
                env.update(request.env)

            # Check if command contains shell metacharacters
            # If so, run through shell. mpirun sends complex commands like:
            # "VAR=value;export VAR;/path/to/prted --args"
            needs_shell = any(c in cmd_str for c in ";|&$`")

            if needs_shell:
                process = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=request.cwd,
                    env=env,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *request.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=request.cwd,
                    env=env,
                )

            stdout, stderr = await process.communicate()
            exit_code = process.returncode or 0

            logger.info(f"Command completed with exit code {exit_code}")

            return ExecuteResponse(
                exit_code=exit_code,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )

        except FileNotFoundError:
            logger.error(f"Command not found: {request.command[0]}")
            return ExecuteResponse(
                exit_code=127,
                stdout="",
                stderr=f"Command not found: {request.command[0]}",
            )
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecuteResponse(
                exit_code=1,
                stdout="",
                stderr=str(e),
            )

    async def run(self):
        cfg = Config()
        cfg.bind = f"0.0.0.0:{self.port}"
        # nb: shared.logging needs updating if any of this changes
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger

        async with create_task_group() as tg:
            self._tg = tg
            logger.info("Starting API")
            tg.start_soon(self._apply_state)
            tg.start_soon(self._pause_on_new_election)
            print_startup_banner(self.port)
            await serve(
                cast(ASGIFramework, self.app),
                cfg,
                shutdown_trigger=lambda: anyio.sleep_forever(),
            )

        self.command_sender.close()
        self.global_event_receiver.close()

    async def _apply_state(self):
        with self.global_event_receiver as events:
            async for f_event in events:
                if f_event.origin != self.session_id.master_node_id:
                    continue
                self.event_buffer.ingest(f_event.origin_idx, f_event.event)
                for idx, event in self.event_buffer.drain_indexed():
                    self._event_log.append(event)
                    self.state = apply(self.state, IndexedEvent(event=event, idx=idx))
                    if isinstance(event, ChunkGenerated):
                        assert isinstance(event.chunk, TokenChunk)
                        queue = self._chat_completion_queues.get(event.command_id)
                        if queue is not None:
                            try:
                                await queue.send(event.chunk)
                            except BrokenResourceError:
                                self._chat_completion_queues.pop(event.command_id, None)

    async def _pause_on_new_election(self):
        with self.election_receiver as ems:
            async for message in ems:
                if message.clock > self.last_completed_election:
                    self.paused = True

    async def _send(self, command: Command):
        while self.paused:
            await self.paused_ev.wait()
        await self.command_sender.send(
            ForwarderCommand(origin=self.node_id, command=command)
        )
