import base64
import contextlib
import json
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import uuid4

import anyio
from anyio import BrokenResourceError
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from loguru import logger

from exo.master.adapters.chat_completions import (
    chat_request_to_text_generation,
    collect_chat_response,
    generate_chat_stream,
)
from exo.master.adapters.claude import (
    claude_request_to_text_generation,
    collect_claude_response,
    generate_claude_stream,
)
from exo.master.adapters.ollama import (
    collect_ollama_chat_response,
    collect_ollama_generate_response,
    generate_ollama_chat_stream,
    generate_ollama_generate_stream,
    ollama_generate_request_to_text_generation,
    ollama_request_to_text_generation,
)
from exo.master.adapters.responses import (
    collect_responses_response,
    generate_responses_stream,
    responses_request_to_text_generation,
)
from exo.master.event_log import DiskEventLog
from exo.master.image_store import ImageStore
from exo.master.placement import place_instance as get_instance_placements
from exo.shared.apply import apply
from exo.shared.constants import (
    DASHBOARD_DIR,
    EXO_CACHE_HOME,
    EXO_EVENT_LOG_DIR,
    EXO_IMAGE_CACHE_DIR,
    EXO_MAX_CHUNK_SIZE,
    EXO_TRACING_CACHE_DIR,
)
from exo.shared.election import ElectionMessage
from exo.shared.logging import InterceptLogger
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    delete_custom_card,
    get_model_cards,
    is_custom_card,
)
from exo.shared.tracing import TraceEvent, compute_stats, export_trace, load_trace_file
from exo.shared.types.api import (
    AddCustomModelParams,
    AdvancedImageParams,
    BenchChatCompletionRequest,
    BenchChatCompletionResponse,
    BenchImageGenerationResponse,
    BenchImageGenerationTaskParams,
    CancelCommandResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CreateInstanceParams,
    CreateInstanceResponse,
    DeleteDownloadResponse,
    DeleteInstanceResponse,
    DeleteTracesRequest,
    DeleteTracesResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    GenerationStats,
    HuggingFaceSearchResult,
    ImageData,
    ImageEditsTaskParams,
    ImageGenerationResponse,
    ImageGenerationStats,
    ImageGenerationTaskParams,
    ImageListItem,
    ImageListResponse,
    ImageSize,
    ModelList,
    ModelListModel,
    PlaceInstanceParams,
    PlacementPreview,
    PlacementPreviewResponse,
    StartDownloadParams,
    StartDownloadResponse,
    ToolCall,
    TraceCategoryStats,
    TraceEventResponse,
    TraceListItem,
    TraceListResponse,
    TraceRankStats,
    TraceResponse,
    TraceStatsResponse,
    normalize_image_size,
)
from exo.shared.types.chunks import (
    ErrorChunk,
    ImageChunk,
    InputImageChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.claude_api import (
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
)
from exo.shared.types.commands import (
    Command,
    CreateInstance,
    DeleteDownload,
    DeleteInstance,
    DownloadCommand,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    SendInputChunk,
    StartDownload,
    TaskCancelled,
    TaskFinished,
    TextGeneration,
)
from exo.shared.types.common import CommandId, Id, NodeId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    TracesMerged,
)
from exo.shared.types.memory import Memory
from exo.shared.types.ollama_api import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModelDetails,
    OllamaModelTag,
    OllamaPsModel,
    OllamaPsResponse,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaTagsResponse,
)
from exo.shared.types.openai_responses import (
    ResponsesRequest,
    ResponsesResponse,
)
from exo.shared.types.state import State
from exo.shared.types.cluster import (
    ClusterDownloadSummary,
    ClusterHealthResponse,
    ClusterModelSummary,
    ClusterModelsResponse,
    ClusterNodeSummary,
    ClusterNodesResponse,
    ClusterOverviewResponse,
    ClusterTaskSummary,
    LoadModelRequest,
    LoadModelResponse,
    ModelStatusResponse,
    SwapModelRequest,
    SwapModelResponse,
    UnloadModelResponse,
)
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
)
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
)
from exo.shared.types.worker.shards import Sharding
from exo.utils.banner import print_startup_banner
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.power_sampler import PowerSampler
from exo.utils.task_group import TaskGroup

_API_EVENT_LOG_DIR = EXO_EVENT_LOG_DIR / "api"
ONBOARDING_COMPLETE_FILE = EXO_CACHE_HOME / "onboarding_complete"


def _format_to_content_type(image_format: Literal["png", "jpeg", "webp"] | None) -> str:
    return f"image/{image_format or 'png'}"


def _ensure_seed(params: AdvancedImageParams | None) -> AdvancedImageParams:
    """Ensure advanced params has a seed set for distributed consistency."""
    if params is None:
        return AdvancedImageParams(seed=random.randint(0, 2**32 - 1))
    if params.seed is None:
        return params.model_copy(update={"seed": random.randint(0, 2**32 - 1)})
    return params


class API:
    def __init__(
        self,
        node_id: NodeId,
        *,
        port: int,
        event_receiver: Receiver[IndexedEvent],
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        # This lets us pause the API if an election is running
        election_receiver: Receiver[ElectionMessage],
    ) -> None:
        self.state = State()
        self._event_log = DiskEventLog(_API_EVENT_LOG_DIR)
        self._system_id = SystemId()
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.event_receiver = event_receiver
        self.election_receiver = election_receiver
        self.node_id: NodeId = node_id
        self.last_completed_election: int = 0
        self.port = port

        self.paused: bool = False
        self.paused_ev: anyio.Event = anyio.Event()

        self.app = FastAPI()

        @self.app.middleware("http")
        async def _log_requests(  # pyright: ignore[reportUnusedFunction]
            request: Request,
            call_next: Callable[[Request], Awaitable[StreamingResponse]],
        ) -> StreamingResponse:
            logger.debug(f"API request: {request.method} {request.url.path}")
            return await call_next(request)

        self._setup_exception_handlers()
        self._setup_cors()
        self._setup_routes()

        self.app.mount(
            "/",
            StaticFiles(
                directory=DASHBOARD_DIR,
                html=True,
            ),
            name="dashboard",
        )

        self._text_generation_queues: dict[
            CommandId,
            Sender[TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk],
        ] = {}
        self._image_generation_queues: dict[
            CommandId, Sender[ImageChunk | ErrorChunk]
        ] = {}
        self._image_store = ImageStore(EXO_IMAGE_CACHE_DIR)
        self._tg: TaskGroup = TaskGroup()

    def reset(self, result_clock: int, event_receiver: Receiver[IndexedEvent]):
        logger.info("Resetting API State")
        self._event_log.close()
        self._event_log = DiskEventLog(_API_EVENT_LOG_DIR)
        self.state = State()
        self._system_id = SystemId()
        self._text_generation_queues = {}
        self._image_generation_queues = {}
        self.unpause(result_clock)
        self.event_receiver.close()
        self.event_receiver = event_receiver
        self._tg.start_soon(self._apply_state)

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
        self.app.post("/models/add")(self.add_custom_model)
        self.app.delete("/models/custom/{model_id:path}")(self.delete_custom_model)
        self.app.get("/models/search")(self.search_models)
        self.app.post("/v1/chat/completions", response_model=None)(
            self.chat_completions
        )
        self.app.post("/bench/chat/completions")(self.bench_chat_completions)
        self.app.post("/v1/images/generations", response_model=None)(
            self.image_generations
        )
        self.app.post("/bench/images/generations")(self.bench_image_generations)
        self.app.post("/v1/images/edits", response_model=None)(self.image_edits)
        self.app.post("/bench/images/edits")(self.bench_image_edits)
        self.app.get("/images")(self.list_images)
        self.app.get("/images/{image_id}")(self.get_image)
        self.app.post("/v1/messages", response_model=None)(self.claude_messages)
        self.app.post("/v1/responses", response_model=None)(self.openai_responses)
        self.app.post("/v1/cancel/{command_id}")(self.cancel_command)

        # Ollama API
        self.app.head("/ollama/")(self.ollama_version)
        self.app.head("/ollama/api/version")(self.ollama_version)
        self.app.post("/ollama/api/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/api/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/v1/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/generate", response_model=None)(self.ollama_generate)
        self.app.get("/ollama/api/tags")(self.ollama_tags)
        self.app.get("/ollama/api/api/tags")(self.ollama_tags)
        self.app.get("/ollama/api/v1/tags")(self.ollama_tags)
        self.app.post("/ollama/api/show")(self.ollama_show)
        self.app.get("/ollama/api/ps")(self.ollama_ps)
        self.app.get("/ollama/api/version")(self.ollama_version)

        self.app.get("/state")(lambda: self.state)
        self.app.get("/events")(self.stream_events)
        self.app.post("/download/start")(self.start_download)
        self.app.delete("/download/{node_id}/{model_id:path}")(self.delete_download)
        self.app.get("/v1/traces")(self.list_traces)
        self.app.post("/v1/traces/delete")(self.delete_traces)
        self.app.get("/v1/traces/{task_id}")(self.get_trace)
        self.app.get("/v1/traces/{task_id}/stats")(self.get_trace_stats)
        self.app.get("/v1/traces/{task_id}/raw")(self.get_trace_raw)
        self.app.get("/onboarding")(self.get_onboarding)
        self.app.post("/onboarding")(self.complete_onboarding)

        # Cluster management API — agent-friendly endpoints
        self.app.get("/v1/cluster")(self.cluster_overview)
        self.app.get("/v1/cluster/health")(self.cluster_health)
        self.app.get("/v1/cluster/nodes")(self.cluster_nodes)
        self.app.get("/v1/cluster/nodes/{node_id}")(self.cluster_node_detail)
        self.app.get("/v1/cluster/models")(self.cluster_models)
        self.app.post("/v1/cluster/models/load")(self.cluster_load_model)
        self.app.delete("/v1/cluster/models/{model_id:path}")(
            self.cluster_unload_model
        )
        self.app.post("/v1/cluster/models/swap")(self.cluster_swap_model)
        self.app.get("/v1/cluster/models/{model_id:path}/status")(
            self.cluster_model_status
        )

    async def place_instance(self, payload: PlaceInstanceParams):
        command = PlaceInstance(
            model_card=await ModelCard.load(payload.model_id),
            sharding=payload.sharding,
            instance_meta=payload.instance_meta,
            min_nodes=payload.min_nodes,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_card=command.model_card,
        )

    async def create_instance(
        self, payload: CreateInstanceParams
    ) -> CreateInstanceResponse:
        instance = payload.instance
        model_card = await ModelCard.load(instance.shard_assignments.model_id)
        required_memory = model_card.storage_size
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
            model_card=model_card,
        )

    async def get_placement(
        self,
        model_id: ModelId,
        sharding: Sharding = Sharding.Pipeline,
        instance_meta: InstanceMeta = InstanceMeta.MlxRing,
        min_nodes: int = 1,
    ) -> Instance:
        model_card = await ModelCard.load(model_id)

        try:
            placements = get_instance_placements(
                PlaceInstance(
                    model_card=model_card,
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
        self,
        model_id: ModelId,
        node_ids: Annotated[list[NodeId] | None, Query()] = None,
    ) -> PlacementPreviewResponse:
        seen: set[tuple[ModelId, Sharding, InstanceMeta, int]] = set()
        previews: list[PlacementPreview] = []
        required_nodes = set(node_ids) if node_ids else None

        if len(list(self.state.topology.list_nodes())) == 0:
            return PlacementPreviewResponse(previews=[])

        try:
            model_card = await ModelCard.load(model_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to load model card: {exc}"
            ) from exc
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

        for sharding, instance_meta, min_nodes in instance_combinations:
            try:
                placements = get_instance_placements(
                    PlaceInstance(
                        model_card=model_card,
                        sharding=sharding,
                        instance_meta=instance_meta,
                        min_nodes=min_nodes,
                    ),
                    node_memory=self.state.node_memory,
                    node_network=self.state.node_network,
                    topology=self.state.topology,
                    current_instances=self.state.instances,
                    required_nodes=required_nodes,
                )
            except ValueError as exc:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error=str(exc),
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            current_ids = set(self.state.instances.keys())
            new_instances = [
                instance
                for instance_id, instance in placements.items()
                if instance_id not in current_ids
            ]

            if len(new_instances) != 1:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error="Expected exactly one new instance from placement",
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            instance = new_instances[0]
            shard_assignments = instance.shard_assignments
            placement_node_ids = list(shard_assignments.node_to_runner.keys())

            memory_delta_by_node: dict[str, int] = {}
            if placement_node_ids:
                total_bytes = model_card.storage_size.in_bytes
                per_node = total_bytes // len(placement_node_ids)
                remainder = total_bytes % len(placement_node_ids)
                for index, node_id in enumerate(sorted(placement_node_ids, key=str)):
                    extra = 1 if index < remainder else 0
                    memory_delta_by_node[str(node_id)] = per_node + extra

            if (
                model_card.model_id,
                sharding,
                instance_meta,
                len(placement_node_ids),
            ) not in seen:
                previews.append(
                    PlacementPreview(
                        model_id=model_card.model_id,
                        sharding=sharding,
                        instance_meta=instance_meta,
                        instance=instance,
                        memory_delta_by_node=memory_delta_by_node or None,
                        error=None,
                    )
                )
            seen.add(
                (
                    model_card.model_id,
                    sharding,
                    instance_meta,
                    len(placement_node_ids),
                )
            )

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

    async def cancel_command(self, command_id: CommandId) -> CancelCommandResponse:
        """Cancel an active command by closing its stream and notifying workers."""
        sender = self._text_generation_queues.get(
            command_id
        ) or self._image_generation_queues.get(command_id)
        if sender is None:
            raise HTTPException(
                status_code=404,
                detail="Command not found or already completed",
            )

        await self._send(TaskCancelled(cancelled_command_id=command_id))
        sender.close()

        return CancelCommandResponse(
            message="Command cancelled.",
            command_id=command_id,
        )

    async def _token_chunk_stream(
        self, command_id: CommandId
    ) -> AsyncGenerator[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None
    ]:
        """Yield chunks for a given command until completion.

        This is the internal low-level stream used by all API adapters.
        """
        try:
            self._text_generation_queues[command_id], recv = channel[
                TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
            ]()

            with recv as token_chunks:
                async for chunk in token_chunks:
                    yield chunk
                    if isinstance(chunk, PrefillProgressChunk):
                        continue
                    if chunk.finish_reason is not None:
                        break

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._text_generation_queues:
                del self._text_generation_queues[command_id]

    async def _collect_text_generation_with_stats(
        self, command_id: CommandId
    ) -> BenchChatCompletionResponse:
        sampler = PowerSampler(get_node_system=lambda: self.state.node_system)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        model: ModelId | None = None
        finish_reason: FinishReason | None = None

        stats: GenerationStats | None = None

        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)

            async for chunk in self._token_chunk_stream(command_id):
                if isinstance(chunk, PrefillProgressChunk):
                    continue

                if chunk.finish_reason == "error":
                    raise HTTPException(
                        status_code=500,
                        detail=chunk.error_message or "Internal server error",
                    )

                if model is None:
                    model = chunk.model

                if isinstance(chunk, TokenChunk):
                    text_parts.append(chunk.text)

                if isinstance(chunk, ToolCallChunk):
                    tool_calls.extend(
                        ToolCall(
                            id=str(uuid4()),
                            index=i,
                            function=tool,
                        )
                        for i, tool in enumerate(chunk.tool_calls)
                    )

                stats = chunk.stats or stats

                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

            tg.cancel_scope.cancel()

        combined_text = "".join(text_parts)
        assert model is not None

        return BenchChatCompletionResponse(
            id=command_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=combined_text,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            generation_stats=stats,
            power_usage=sampler.result(),
        )

    async def _trigger_notify_user_to_download_model(self, model_id: ModelId) -> None:
        logger.warning(
            "TODO: we should send a notification to the user to download the model"
        )

    async def chat_completions(
        self, payload: ChatCompletionRequest
    ) -> ChatCompletionResponse | StreamingResponse:
        """OpenAI Chat Completions API - adapter."""
        task_params = chat_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = TextGeneration(task_params=task_params)
        await self._send(command)

        if payload.stream:
            return StreamingResponse(
                generate_chat_stream(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_chat_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def bench_chat_completions(
        self, payload: BenchChatCompletionRequest
    ) -> BenchChatCompletionResponse:
        task_params = chat_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        task_params = task_params.model_copy(update={"stream": False, "bench": True})

        command = TextGeneration(task_params=task_params)
        await self._send(command)

        return await self._collect_text_generation_with_stats(command.command_id)

    async def _resolve_and_validate_text_model(self, model_id: ModelId) -> ModelId:
        """Validate a text model exists and return the resolved model ID.

        Raises HTTPException 404 if no instance is found for the model.
        """
        if not any(
            instance.shard_assignments.model_id == model_id
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(model_id)
            raise HTTPException(
                status_code=404,
                detail=f"No instance found for model {model_id}",
            )
        return model_id

    async def _validate_image_model(self, model: ModelId) -> ModelId:
        """Validate model exists and return resolved model ID.

        Raises HTTPException 404 if no instance is found for the model.
        """
        model_card = await ModelCard.load(model)
        resolved_model = model_card.model_id
        if not any(
            instance.shard_assignments.model_id == resolved_model
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(resolved_model)
            raise HTTPException(
                status_code=404, detail=f"No instance found for model {resolved_model}"
            )
        return resolved_model

    def stream_events(self) -> StreamingResponse:
        def _generate_json_array(events: Iterable[Event]) -> Iterable[str]:
            yield "["
            first = True
            for event in events:
                if not first:
                    yield ","
                first = False
                yield event.model_dump_json()
            yield "]"

        return StreamingResponse(
            _generate_json_array(self._event_log.read_all()),
            media_type="application/json",
        )

    async def get_image(self, image_id: str) -> FileResponse:
        stored = self._image_store.get(Id(image_id))
        if stored is None:
            raise HTTPException(status_code=404, detail="Image not found or expired")
        return FileResponse(path=stored.file_path, media_type=stored.content_type)

    async def list_images(self, request: Request) -> ImageListResponse:
        """List all stored images."""
        stored_images = self._image_store.list_images()
        return ImageListResponse(
            data=[
                ImageListItem(
                    image_id=img.image_id,
                    url=self._build_image_url(request, img.image_id),
                    content_type=img.content_type,
                    expires_at=img.expires_at,
                )
                for img in stored_images
            ]
        )

    def _build_image_url(self, request: Request, image_id: Id) -> str:
        host = request.headers.get("host", f"localhost:{self.port}")
        scheme = "https" if request.url.scheme == "https" else "http"
        return f"{scheme}://{host}/v1/images/{image_id}"

    async def image_generations(
        self, request: Request, payload: ImageGenerationTaskParams
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image generation requests.

        When stream=True and partial_images > 0, returns a StreamingResponse
        with SSE-formatted events for partial and final images.
        """
        payload = payload.model_copy(
            update={
                "model": await self._validate_image_model(ModelId(payload.model)),
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            task_params=payload,
        )
        await self._send(command)

        # Check if streaming is requested
        if payload.stream and payload.partial_images and payload.partial_images > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=payload.n or 1,
                    response_format=payload.response_format or "b64_json",
                ),
                media_type="text/event-stream",
            )

        # Non-streaming: collect all image chunks
        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _generate_image_stream(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> AsyncGenerator[str, None]:
        """Generate SSE stream of partial and final images."""
        # Track chunks: {(image_index, is_partial): {chunk_index: data}}
        image_chunks: dict[tuple[int, bool], dict[int, str]] = {}
        image_total_chunks: dict[tuple[int, bool], int] = {}
        image_metadata: dict[tuple[int, bool], tuple[int | None, int | None]] = {}
        images_complete = 0

        try:
            self._image_generation_queues[command_id], recv = channel[
                ImageChunk | ErrorChunk
            ]()

            with recv as chunks:
                async for chunk in chunks:
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

                    key = (chunk.image_index, chunk.is_partial)

                    if key not in image_chunks:
                        image_chunks[key] = {}
                        image_total_chunks[key] = chunk.total_chunks
                        image_metadata[key] = (
                            chunk.partial_index,
                            chunk.total_partials,
                        )

                    image_chunks[key][chunk.chunk_index] = chunk.data

                    # Check if this image is complete
                    if len(image_chunks[key]) == image_total_chunks[key]:
                        full_data = "".join(
                            image_chunks[key][i] for i in range(len(image_chunks[key]))
                        )

                        partial_idx, total_partials = image_metadata[key]

                        if chunk.is_partial:
                            # Yield partial image event (always use b64_json for partials)
                            event_data = {
                                "type": "partial",
                                "image_index": chunk.image_index,
                                "partial_index": partial_idx,
                                "total_partials": total_partials,
                                "format": str(chunk.format),
                                "data": {
                                    "b64_json": full_data
                                    if response_format == "b64_json"
                                    else None,
                                },
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                        else:
                            # Final image
                            if response_format == "url":
                                image_bytes = base64.b64decode(full_data)
                                content_type = _format_to_content_type(chunk.format)
                                stored = self._image_store.store(
                                    image_bytes, content_type
                                )
                                url = self._build_image_url(request, stored.image_id)
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"url": url},
                                }
                            else:
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"b64_json": full_data},
                                }
                            yield f"data: {json.dumps(event_data)}\n\n"
                            images_complete += 1

                            if images_complete >= num_images:
                                yield "data: [DONE]\n\n"
                                break

                        # Clean up completed image chunks
                        del image_chunks[key]
                        del image_total_chunks[key]
                        del image_metadata[key]

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]

    async def _collect_image_chunks(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
        capture_stats: bool = False,
    ) -> tuple[list[ImageData], ImageGenerationStats | None]:
        """Collect image chunks and optionally capture stats."""
        # Track chunks per image: {image_index: {chunk_index: data}}
        # Only track non-partial (final) images
        image_chunks: dict[int, dict[int, str]] = {}
        image_total_chunks: dict[int, int] = {}
        image_formats: dict[int, Literal["png", "jpeg", "webp"] | None] = {}
        images_complete = 0
        stats: ImageGenerationStats | None = None

        try:
            self._image_generation_queues[command_id], recv = channel[
                ImageChunk | ErrorChunk
            ]()

            while images_complete < num_images:
                with recv as chunks:
                    async for chunk in chunks:
                        if chunk.finish_reason == "error":
                            raise HTTPException(
                                status_code=500,
                                detail=chunk.error_message or "Internal server error",
                            )

                        if chunk.is_partial:
                            continue

                        if chunk.image_index not in image_chunks:
                            image_chunks[chunk.image_index] = {}
                            image_total_chunks[chunk.image_index] = chunk.total_chunks
                            image_formats[chunk.image_index] = chunk.format

                        image_chunks[chunk.image_index][chunk.chunk_index] = chunk.data

                        if capture_stats and chunk.stats is not None:
                            stats = chunk.stats

                        if (
                            len(image_chunks[chunk.image_index])
                            == image_total_chunks[chunk.image_index]
                        ):
                            images_complete += 1

                        if images_complete >= num_images:
                            break

            images: list[ImageData] = []
            for image_idx in range(num_images):
                chunks_dict = image_chunks[image_idx]
                full_data = "".join(chunks_dict[i] for i in range(len(chunks_dict)))
                if response_format == "url" and request is not None:
                    image_bytes = base64.b64decode(full_data)
                    content_type = _format_to_content_type(image_formats.get(image_idx))
                    stored = self._image_store.store(image_bytes, content_type)
                    url = self._build_image_url(request, stored.image_id)
                    images.append(ImageData(b64_json=None, url=url))
                else:
                    images.append(
                        ImageData(
                            b64_json=full_data
                            if response_format == "b64_json"
                            else None,
                            url=None,
                        )
                    )

            return (images, stats if capture_stats else None)
        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]

    async def _collect_image_generation(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> ImageGenerationResponse:
        """Collect all image chunks (non-streaming) and return a single response."""
        images, _ = await self._collect_image_chunks(
            request, command_id, num_images, response_format, capture_stats=False
        )
        return ImageGenerationResponse(data=images)

    async def _collect_image_generation_with_stats(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> BenchImageGenerationResponse:
        sampler = PowerSampler(get_node_system=lambda: self.state.node_system)
        images: list[ImageData] = []
        stats: ImageGenerationStats | None = None
        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)
            images, stats = await self._collect_image_chunks(
                request, command_id, num_images, response_format, capture_stats=True
            )
            tg.cancel_scope.cancel()
        return BenchImageGenerationResponse(
            data=images, generation_stats=stats, power_usage=sampler.result()
        )

    async def bench_image_generations(
        self, request: Request, payload: BenchImageGenerationTaskParams
    ) -> BenchImageGenerationResponse:
        payload = payload.model_copy(
            update={
                "model": await self._validate_image_model(ModelId(payload.model)),
                "stream": False,
                "partial_images": 0,
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            task_params=payload,
        )
        await self._send(command)

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _send_image_edits_command(
        self,
        image: UploadFile,
        prompt: str,
        model: ModelId,
        n: int,
        size: ImageSize,
        response_format: Literal["url", "b64_json"],
        input_fidelity: Literal["low", "high"],
        stream: bool,
        partial_images: int,
        bench: bool,
        quality: Literal["high", "medium", "low"],
        output_format: Literal["png", "jpeg", "webp"],
        advanced_params: AdvancedImageParams | None,
    ) -> ImageEdits:
        """Prepare and send an image edits command with chunked image upload."""
        resolved_model = await self._validate_image_model(model)
        advanced_params = _ensure_seed(advanced_params)

        image_content = await image.read()
        image_data = base64.b64encode(image_content).decode("utf-8")

        image_strength = 0.7 if input_fidelity == "high" else 0.3

        data_chunks = [
            image_data[i : i + EXO_MAX_CHUNK_SIZE]
            for i in range(0, len(image_data), EXO_MAX_CHUNK_SIZE)
        ]
        total_chunks = len(data_chunks)

        command = ImageEdits(
            task_params=ImageEditsTaskParams(
                image_data="",
                total_input_chunks=total_chunks,
                prompt=prompt,
                model=resolved_model,
                n=n,
                size=size,
                response_format=response_format,
                image_strength=image_strength,
                stream=stream,
                partial_images=partial_images,
                bench=bench,
                quality=quality,
                output_format=output_format,
                advanced_params=advanced_params,
            ),
        )

        logger.info(
            f"Sending input image: {len(image_data)} bytes in {total_chunks} chunks"
        )
        for chunk_index, chunk_data in enumerate(data_chunks):
            await self._send(
                SendInputChunk(
                    chunk=InputImageChunk(
                        model=resolved_model,
                        command_id=command.command_id,
                        data=chunk_data,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                    )
                )
            )

        await self._send(command)
        return command

    async def image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        stream: str = Form("false"),
        partial_images: str = Form("0"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image editing requests (img2img)."""
        # Parse string form values to proper types
        stream_bool = stream.lower() in ("true", "1", "yes")
        partial_images_int = int(partial_images) if partial_images.isdigit() else 0

        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=stream_bool,
            partial_images=partial_images_int,
            bench=False,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        if stream_bool and partial_images_int > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=n,
                    response_format=response_format,
                ),
                media_type="text/event-stream",
            )

        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def bench_image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> BenchImageGenerationResponse:
        """Handle benchmark image editing requests with generation stats."""
        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=False,
            partial_images=0,
            bench=True,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def claude_messages(
        self, payload: ClaudeMessagesRequest
    ) -> ClaudeMessagesResponse | StreamingResponse:
        """Claude Messages API - adapter."""
        task_params = claude_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = TextGeneration(task_params=task_params)
        await self._send(command)

        if payload.stream:
            return StreamingResponse(
                generate_claude_stream(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_claude_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def openai_responses(
        self, payload: ResponsesRequest
    ) -> ResponsesResponse | StreamingResponse:
        """OpenAI Responses API."""
        task_params = responses_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(task_params.model)
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = TextGeneration(task_params=task_params)
        await self._send(command)

        if payload.stream:
            return StreamingResponse(
                generate_responses_stream(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            return StreamingResponse(
                collect_responses_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def _ollama_root(self) -> JSONResponse:
        """Respond to HEAD / from Ollama CLI connectivity checks."""
        return JSONResponse(content="Ollama is running")

    async def ollama_chat(
        self, request: Request
    ) -> OllamaChatResponse | StreamingResponse:
        """Ollama Chat API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaChatRequest.model_validate_json(body)
        task_params = ollama_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = TextGeneration(task_params=task_params)
        await self._send(command)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_chat_stream(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_chat_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def ollama_generate(
        self, request: Request
    ) -> OllamaGenerateResponse | StreamingResponse:
        """Ollama Generate API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaGenerateRequest.model_validate_json(body)
        task_params = ollama_generate_request_to_text_generation(payload)
        resolved_model = await self._resolve_and_validate_text_model(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": resolved_model})

        command = TextGeneration(task_params=task_params)
        await self._send(command)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_generate_stream(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_generate_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def ollama_tags(self) -> OllamaTagsResponse:
        """Returns list of models in Ollama tags format. We return the downloaded ones only."""

        def none_if_empty(value: str) -> str | None:
            return value or None

        downloaded_model_ids: set[str] = set()
        for node_downloads in self.state.downloads.values():
            for dl in node_downloads:
                if isinstance(dl, DownloadCompleted):
                    downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)

        cards = [
            c for c in await get_model_cards() if c.model_id in downloaded_model_ids
        ]

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return OllamaTagsResponse(
            models=[
                OllamaModelTag(
                    name=str(card.model_id),
                    model=str(card.model_id),
                    modified_at=now,
                    size=card.storage_size.in_bytes,
                    digest="sha256:000000000000",
                    details=OllamaModelDetails(
                        family=none_if_empty(card.family),
                        quantization_level=none_if_empty(card.quantization),
                    ),
                )
                for card in cards
            ]
        )

    async def ollama_show(self, request: Request) -> OllamaShowResponse:
        """Returns model information in Ollama show format."""
        body = await request.body()
        payload = OllamaShowRequest.model_validate_json(body)
        model_name = payload.name or payload.model
        if not model_name:
            raise HTTPException(status_code=400, detail="name or model is required")
        try:
            card = await ModelCard.load(ModelId(model_name))
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail=f"Model not found: {model_name}"
            ) from exc

        return OllamaShowResponse(
            modelfile=f"FROM {card.model_id}",
            template="{{ .Prompt }}",
            details=OllamaModelDetails(
                family=card.family or None,
                quantization_level=card.quantization or None,
            ),
        )

    async def ollama_ps(self) -> OllamaPsResponse:
        """Returns list of running models (active instances)."""
        models: list[OllamaPsModel] = []
        seen: set[str] = set()
        for instance in self.state.instances.values():
            model_id = str(instance.shard_assignments.model_id)
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                OllamaPsModel(
                    name=model_id,
                    model=model_id,
                    size=0,
                )
            )
        return OllamaPsResponse(models=models)

    async def ollama_version(self) -> dict[str, str]:
        """Returns version information for Ollama API compatibility."""
        return {"version": "exo v1.0"}

    # ------------------------------------------------------------------
    # Cluster management API — agent-friendly endpoints
    # ------------------------------------------------------------------

    def _build_node_summary(self, node_id: NodeId) -> ClusterNodeSummary:
        """Build a flat, agent-friendly summary for a single node."""
        identity = self.state.node_identities.get(node_id)
        memory = self.state.node_memory.get(node_id)
        disk = self.state.node_disk.get(node_id)
        system = self.state.node_system.get(node_id)
        network = self.state.node_network.get(node_id)
        tb = self.state.node_thunderbolt.get(node_id)
        rdma = self.state.node_rdma_ctl.get(node_id)
        last_seen = self.state.last_seen.get(node_id)

        # Determine status from last_seen
        status: str = "unknown"
        seconds_since: float | None = None
        if last_seen is not None:
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            delta = (now - last_seen).total_seconds()
            seconds_since = round(delta, 1)
            status = "online" if delta < 30 else "stale"

        # Collect loaded models for this node
        loaded_models: list[str] = []
        for instance in self.state.instances.values():
            if node_id in instance.shard_assignments.node_to_runner:
                model_name = instance.shard_assignments.model_id.short()
                if model_name not in loaded_models:
                    loaded_models.append(model_name)

        # IP addresses and connection types
        ip_addresses: list[str] = []
        connection_types: list[str] = []
        if network:
            for iface in network.interfaces:
                ip_addresses.append(iface.ip_address)
                if iface.interface_type not in connection_types:
                    connection_types.append(iface.interface_type)

        ram_total = round(memory.ram_total.in_gb, 1) if memory else 0.0
        ram_avail = round(memory.ram_available.in_gb, 1) if memory else 0.0
        ram_used = round(ram_total - ram_avail, 1)
        ram_pct = round((ram_used / ram_total) * 100, 1) if ram_total > 0 else 0.0

        return ClusterNodeSummary(
            node_id=str(node_id),
            friendly_name=identity.friendly_name if identity else "Unknown",
            chip=identity.chip_id if identity else "Unknown",
            os_version=identity.os_version if identity else "Unknown",
            ram_total_gb=ram_total,
            ram_available_gb=ram_avail,
            ram_used_gb=ram_used,
            ram_used_percent=ram_pct,
            disk_total_gb=round(disk.total.in_gb, 1) if disk else 0.0,
            disk_available_gb=round(disk.available.in_gb, 1) if disk else 0.0,
            gpu_usage_percent=round(system.gpu_usage, 1) if system else 0.0,
            cpu_p_usage_percent=round(system.pcpu_usage, 1) if system else 0.0,
            cpu_e_usage_percent=round(system.ecpu_usage, 1) if system else 0.0,
            temperature_c=round(system.temp, 1) if system else 0.0,
            power_watts=round(system.sys_power, 1) if system else 0.0,
            ip_addresses=ip_addresses,
            connection_types=connection_types,
            has_thunderbolt=bool(tb and len(tb.interfaces) > 0),
            rdma_enabled=bool(rdma and rdma.enabled),
            loaded_models=loaded_models,
            status=status,
            seconds_since_seen=seconds_since,
        )

    def _build_model_summary(
        self, instance_id: InstanceId, instance: Instance
    ) -> ClusterModelSummary:
        """Build a flat summary for a loaded model instance."""
        assignments = instance.shard_assignments
        model_id = str(assignments.model_id)
        model_name = assignments.model_id.short()
        nodes = list(str(n) for n in assignments.node_to_runner.keys())

        # Friendly names for nodes
        node_names = []
        for nid in assignments.node_to_runner.keys():
            identity = self.state.node_identities.get(nid)
            node_names.append(identity.friendly_name if identity else str(nid)[:8])

        # Determine instance type and sharding from the instance class
        from exo.shared.types.worker.instances import MlxJacclInstance, MlxRingInstance

        if isinstance(instance, MlxRingInstance):
            instance_type = "MlxRing"
        elif isinstance(instance, MlxJacclInstance):
            instance_type = "MlxJaccl"
        else:
            instance_type = "unknown"

        # Determine sharding from shard metadata
        sharding = "unknown"
        for shard_meta in assignments.runner_to_shard.values():
            sharding = type(shard_meta).__name__.replace("ShardMetadata", "").lower()
            break

        # Runner statuses for each node
        runner_statuses: dict[str, str] = {}
        ready = True
        overall_status = "unknown"
        for nid, runner_id in assignments.node_to_runner.items():
            runner_status = self.state.runners.get(runner_id)
            if runner_status is None:
                runner_statuses[str(nid)] = "unknown"
                ready = False
            else:
                status_name = type(runner_status).__name__.replace("Runner", "").lower()
                runner_statuses[str(nid)] = status_name
                if isinstance(runner_status, RunnerFailed):
                    ready = False
                    overall_status = "failed"
                elif isinstance(runner_status, RunnerLoading):
                    ready = False
                    overall_status = "loading"
                elif isinstance(runner_status, RunnerRunning):
                    overall_status = "running"
                elif isinstance(runner_status, RunnerReady) and overall_status not in (
                    "running",
                    "failed",
                    "loading",
                ):
                    overall_status = "ready"

        if overall_status == "unknown" and ready:
            overall_status = "ready"

        # Get storage size from any shard's model card
        storage_gb = 0.0
        for shard_meta in assignments.runner_to_shard.values():
            storage_gb = round(shard_meta.model_card.storage_size.in_gb, 1)
            break

        return ClusterModelSummary(
            instance_id=str(instance_id),
            model_id=model_id,
            model_name=model_name,
            sharding=sharding,
            instance_type=instance_type,
            nodes=nodes,
            node_names=node_names,
            storage_size_gb=storage_gb,
            runner_statuses=runner_statuses,
            ready=ready,
            status=overall_status,
        )

    def _build_download_summaries(self) -> list[ClusterDownloadSummary]:
        """Build flat download summaries from state."""
        summaries: list[ClusterDownloadSummary] = []
        for node_id, downloads in self.state.downloads.items():
            identity = self.state.node_identities.get(node_id)
            node_name = identity.friendly_name if identity else str(node_id)[:8]
            for dl in downloads:
                model_id = str(dl.shard_metadata.model_card.model_id)
                if isinstance(dl, DownloadCompleted):
                    summaries.append(
                        ClusterDownloadSummary(
                            node_id=str(node_id),
                            node_name=node_name,
                            model_id=model_id,
                            status="completed",
                            downloaded_gb=round(dl.total.in_gb, 1),
                            total_gb=round(dl.total.in_gb, 1),
                            progress_percent=100.0,
                        )
                    )
                elif isinstance(dl, DownloadOngoing):
                    prog = dl.download_progress
                    total_gb = prog.total.in_gb
                    dl_gb = prog.downloaded.in_gb
                    pct = round((dl_gb / total_gb) * 100, 1) if total_gb > 0 else 0.0
                    summaries.append(
                        ClusterDownloadSummary(
                            node_id=str(node_id),
                            node_name=node_name,
                            model_id=model_id,
                            status="downloading",
                            downloaded_gb=round(dl_gb, 1),
                            total_gb=round(total_gb, 1),
                            progress_percent=pct,
                            speed_mb_s=round(prog.speed / (1024 * 1024), 1),
                            eta_seconds=round(prog.eta_ms / 1000, 1)
                            if prog.eta_ms > 0
                            else None,
                        )
                    )
                elif isinstance(dl, DownloadPending):
                    summaries.append(
                        ClusterDownloadSummary(
                            node_id=str(node_id),
                            node_name=node_name,
                            model_id=model_id,
                            status="pending",
                            total_gb=round(dl.total.in_gb, 1),
                        )
                    )
                elif isinstance(dl, DownloadFailed):
                    summaries.append(
                        ClusterDownloadSummary(
                            node_id=str(node_id),
                            node_name=node_name,
                            model_id=model_id,
                            status="failed",
                            error=dl.error_message,
                        )
                    )
        return summaries

    def _build_task_summaries(self) -> list[ClusterTaskSummary]:
        """Build flat task summaries from state."""
        from exo.shared.types.tasks import (
            ImageEdits as ImageEditsTask,
            ImageGeneration as ImageGenTask,
            TextGeneration as TextGenTask,
        )

        summaries: list[ClusterTaskSummary] = []
        for task_id, task in self.state.tasks.items():
            task_type = type(task).__name__
            # Map to friendly names
            if isinstance(task, TextGenTask):
                task_type = "text_generation"
            elif isinstance(task, ImageGenTask):
                task_type = "image_generation"
            elif isinstance(task, ImageEditsTask):
                task_type = "image_edit"

            model_id = ""
            for inst in self.state.instances.values():
                if inst.shard_assignments.model_id and task.instance_id == getattr(
                    inst, "instance_id", None
                ):
                    model_id = str(inst.shard_assignments.model_id)
                    break

            summaries.append(
                ClusterTaskSummary(
                    task_id=str(task_id),
                    task_type=task_type,
                    status=task.task_status.value.lower(),
                    model_id=model_id,
                    instance_id=str(task.instance_id),
                )
            )
        return summaries

    async def cluster_health(self) -> ClusterHealthResponse:
        """Quick health check — is the cluster alive and how many nodes?"""
        nodes = list(self.state.topology.list_nodes())
        return ClusterHealthResponse(
            healthy=len(nodes) > 0,
            node_count=len(nodes),
            master_node_id=str(self.node_id),
        )

    async def cluster_overview(self) -> ClusterOverviewResponse:
        """One call to understand the entire cluster.

        Returns nodes, loaded models, active tasks, and downloads in a flat,
        agent-friendly format.  Designed so a single GET gives full situational
        awareness.
        """
        nodes = [
            self._build_node_summary(nid)
            for nid in self.state.topology.list_nodes()
        ]
        models = [
            self._build_model_summary(iid, inst)
            for iid, inst in self.state.instances.items()
        ]
        tasks = self._build_task_summaries()
        downloads = self._build_download_summaries()

        total_ram = sum(n.ram_total_gb for n in nodes)
        avail_ram = sum(n.ram_available_gb for n in nodes)
        used_ram = round(total_ram - avail_ram, 1)
        pct = round((used_ram / total_ram) * 100, 1) if total_ram > 0 else 0.0

        active_downloads = [d for d in downloads if d.status in ("pending", "downloading")]

        return ClusterOverviewResponse(
            node_count=len(nodes),
            total_ram_gb=round(total_ram, 1),
            available_ram_gb=round(avail_ram, 1),
            used_ram_gb=used_ram,
            ram_used_percent=pct,
            loaded_model_count=len(models),
            active_task_count=len([t for t in tasks if t.status in ("pending", "running")]),
            active_download_count=len(active_downloads),
            nodes=nodes,
            models=models,
            tasks=tasks,
            downloads=downloads,
            master_node_id=str(self.node_id),
        )

    async def cluster_nodes(self) -> ClusterNodesResponse:
        """List all nodes with agent-friendly summaries."""
        nodes = [
            self._build_node_summary(nid)
            for nid in self.state.topology.list_nodes()
        ]
        return ClusterNodesResponse(node_count=len(nodes), nodes=nodes)

    async def cluster_node_detail(self, node_id: str) -> ClusterNodeSummary:
        """Get detailed info for a single node."""
        nid = NodeId(node_id)
        all_nodes = list(self.state.topology.list_nodes())
        if nid not in all_nodes:
            raise HTTPException(
                status_code=404,
                detail=f"Node '{node_id}' not found. "
                f"Available nodes: {[str(n) for n in all_nodes]}",
            )
        return self._build_node_summary(nid)

    async def cluster_models(self) -> ClusterModelsResponse:
        """All loaded models and active downloads."""
        loaded = [
            self._build_model_summary(iid, inst)
            for iid, inst in self.state.instances.items()
        ]
        downloading = [
            d
            for d in self._build_download_summaries()
            if d.status in ("pending", "downloading")
        ]
        return ClusterModelsResponse(loaded=loaded, downloading=downloading)

    async def cluster_load_model(self, payload: LoadModelRequest) -> LoadModelResponse:
        """Load a model by name — the cluster auto-places it.

        This is the agent-friendly wrapper around ``place_instance``.  Just
        provide a model ID (e.g. ``mlx-community/Qwen3-30B-A3B-4bit``) and
        optional preferences; the cluster handles sharding and placement.
        """
        model_card = await ModelCard.load(ModelId(payload.model_id))
        required_memory = model_card.storage_size
        available_memory = self._calculate_total_available_memory()

        if required_memory > available_memory:
            # Build a helpful error that suggests what to unload
            loaded_models = []
            for inst in self.state.instances.values():
                for shard in inst.shard_assignments.runner_to_shard.values():
                    loaded_models.append(
                        f"{shard.model_card.model_id.short()} "
                        f"({shard.model_card.storage_size.in_gb:.1f}GB)"
                    )
                    break

            hint = ""
            if loaded_models:
                hint = (
                    f" Currently loaded: {', '.join(loaded_models)}. "
                    "Unload a model to free memory."
                )

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Insufficient memory to load {model_card.model_id.short()}. "
                    f"Need {required_memory.in_gb:.1f}GB, "
                    f"have {available_memory.in_gb:.1f}GB available.{hint}"
                ),
            )

        # Map user-friendly sharding preference to enum
        if payload.preferred_sharding == "tensor":
            sharding = Sharding.Tensor
        elif payload.preferred_sharding == "pipeline":
            sharding = Sharding.Pipeline
        else:
            # "auto" — prefer tensor if supported, fall back to pipeline
            sharding = (
                Sharding.Tensor if model_card.supports_tensor else Sharding.Pipeline
            )

        command = PlaceInstance(
            model_card=model_card,
            sharding=sharding,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=payload.min_nodes,
        )
        await self._send(command)

        return LoadModelResponse(
            message=f"Loading {model_card.model_id.short()} across the cluster.",
            command_id=str(command.command_id),
            model_id=str(model_card.model_id),
            estimated_memory_gb=round(model_card.storage_size.in_gb, 1),
        )

    async def cluster_unload_model(self, model_id: str) -> UnloadModelResponse:
        """Unload a model by model ID (not instance ID).

        Finds the instance matching the given model ID and deletes it.
        If multiple instances of the same model exist, unloads the first one.
        """
        target_instance_id: InstanceId | None = None
        target_model_id: str = ""
        freed_gb: float = 0.0

        for iid, instance in self.state.instances.items():
            inst_model = str(instance.shard_assignments.model_id)
            # Match on full ID or short name
            if inst_model == model_id or instance.shard_assignments.model_id.short() == model_id:
                target_instance_id = iid
                target_model_id = inst_model
                for shard in instance.shard_assignments.runner_to_shard.values():
                    freed_gb = round(shard.model_card.storage_size.in_gb, 1)
                    break
                break

        if target_instance_id is None:
            # Build helpful error with what IS loaded
            loaded = [
                f"{inst.shard_assignments.model_id.short()} (id: {inst.shard_assignments.model_id})"
                for inst in self.state.instances.values()
            ]
            hint = f" Loaded models: {', '.join(loaded)}" if loaded else " No models are currently loaded."
            raise HTTPException(
                status_code=404,
                detail=f"No loaded model matching '{model_id}'.{hint}",
            )

        command = DeleteInstance(instance_id=target_instance_id)
        await self._send(command)

        return UnloadModelResponse(
            message=f"Unloading {ModelId(target_model_id).short()}.",
            command_id=str(command.command_id),
            model_id=target_model_id,
            instance_id=str(target_instance_id),
            freed_memory_gb=freed_gb,
        )

    async def cluster_model_status(self, model_id: str) -> ModelStatusResponse:
        """Poll the status of a model by name.

        Returns whether it's loaded, loading, downloading, or not present.
        Designed for polling loops: call this after ``load`` or ``swap`` until
        ``ready`` is ``true``.
        """
        # Check loaded instances
        for iid, instance in self.state.instances.items():
            inst_model = str(instance.shard_assignments.model_id)
            if inst_model == model_id or instance.shard_assignments.model_id.short() == model_id:
                summary = self._build_model_summary(iid, instance)
                progress = None
                if summary.status == "loading":
                    # Try to get layer loading progress
                    for runner_id in instance.shard_assignments.node_to_runner.values():
                        rs = self.state.runners.get(runner_id)
                        if isinstance(rs, RunnerLoading) and rs.total_layers > 0:
                            pct = round((rs.layers_loaded / rs.total_layers) * 100)
                            progress = f"Loading layers: {rs.layers_loaded}/{rs.total_layers} ({pct}%)"
                            break

                return ModelStatusResponse(
                    model_id=inst_model,
                    found=True,
                    status=summary.status,
                    ready=summary.ready,
                    progress=progress,
                    nodes=summary.node_names,
                    instance_id=str(iid),
                )

        # Check downloads
        for node_id, downloads in self.state.downloads.items():
            for dl in downloads:
                dl_model = str(dl.shard_metadata.model_card.model_id)
                if dl_model == model_id or dl.shard_metadata.model_card.model_id.short() == model_id:
                    if isinstance(dl, DownloadOngoing):
                        prog = dl.download_progress
                        pct = round((prog.downloaded.in_gb / prog.total.in_gb) * 100) if prog.total.in_gb > 0 else 0
                        return ModelStatusResponse(
                            model_id=dl_model,
                            found=True,
                            status="downloading",
                            progress=f"Downloading: {prog.downloaded.in_gb:.1f}/{prog.total.in_gb:.1f}GB ({pct}%)",
                        )
                    elif isinstance(dl, DownloadPending):
                        return ModelStatusResponse(
                            model_id=dl_model,
                            found=True,
                            status="downloading",
                            progress="Download pending...",
                        )

        return ModelStatusResponse(model_id=model_id, found=False, status="not_loaded")

    async def cluster_swap_model(self, payload: SwapModelRequest) -> SwapModelResponse:
        """Swap one model for another in a single call.

        Unloads the old model then loads the new one.  Ideal for scheduled
        model rotation (e.g. large model overnight, small model during the day).
        """
        # Step 1: Find and unload the old model
        target_instance_id: InstanceId | None = None
        freed_gb: float = 0.0
        unloaded_model_name = payload.unload_model_id

        for iid, instance in self.state.instances.items():
            inst_model = str(instance.shard_assignments.model_id)
            if inst_model == payload.unload_model_id or instance.shard_assignments.model_id.short() == payload.unload_model_id:
                target_instance_id = iid
                unloaded_model_name = inst_model
                for shard in instance.shard_assignments.runner_to_shard.values():
                    freed_gb = round(shard.model_card.storage_size.in_gb, 1)
                    break
                break

        if target_instance_id is None:
            loaded = [
                inst.shard_assignments.model_id.short()
                for inst in self.state.instances.values()
            ]
            hint = f" Loaded: {', '.join(loaded)}" if loaded else " No models loaded."
            raise HTTPException(
                status_code=404,
                detail=f"Cannot swap: model '{payload.unload_model_id}' not found.{hint}",
            )

        # Step 2: Validate the new model can be loaded (with memory freed by unload)
        model_card = await ModelCard.load(ModelId(payload.load_model_id))
        available_after_unload = self._calculate_total_available_memory() + Memory.from_gb(freed_gb)

        if model_card.storage_size > available_after_unload:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Even after unloading {ModelId(unloaded_model_name).short()} "
                    f"(freeing {freed_gb:.1f}GB), not enough memory for "
                    f"{model_card.model_id.short()} "
                    f"(needs {model_card.storage_size.in_gb:.1f}GB, "
                    f"would have {available_after_unload.in_gb:.1f}GB)."
                ),
            )

        # Step 3: Send unload command
        unload_cmd = DeleteInstance(instance_id=target_instance_id)
        await self._send(unload_cmd)

        # Step 4: Send load command
        if payload.preferred_sharding == "tensor":
            sharding = Sharding.Tensor
        elif payload.preferred_sharding == "pipeline":
            sharding = Sharding.Pipeline
        else:
            sharding = (
                Sharding.Tensor if model_card.supports_tensor else Sharding.Pipeline
            )

        load_cmd = PlaceInstance(
            model_card=model_card,
            sharding=sharding,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=payload.min_nodes,
        )
        await self._send(load_cmd)

        return SwapModelResponse(
            message=(
                f"Swapping {ModelId(unloaded_model_name).short()} → "
                f"{model_card.model_id.short()}. "
                f"Poll GET /v1/cluster/models/{model_card.model_id}/status "
                f"until ready=true."
            ),
            unload_command_id=str(unload_cmd.command_id),
            load_command_id=str(load_cmd.command_id),
            unloaded_model=unloaded_model_name,
            loaded_model=str(model_card.model_id),
            freed_memory_gb=freed_gb,
            estimated_load_memory_gb=round(model_card.storage_size.in_gb, 1),
        )

    def _calculate_total_available_memory(self) -> Memory:
        """Calculate total available memory across all nodes in bytes."""
        total_available = Memory()

        for memory in self.state.node_memory.values():
            total_available += memory.ram_available

        return total_available

    async def get_models(self, status: str | None = Query(default=None)) -> ModelList:
        """Returns list of available models, optionally filtered by being downloaded."""
        cards = await get_model_cards()

        if status == "downloaded":
            downloaded_model_ids: set[str] = set()
            for node_downloads in self.state.downloads.values():
                for dl in node_downloads:
                    if isinstance(dl, DownloadCompleted):
                        downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)
            cards = [c for c in cards if c.model_id in downloaded_model_ids]

        return ModelList(
            data=[
                ModelListModel(
                    id=card.model_id,
                    hugging_face_id=card.model_id,
                    name=card.model_id.short(),
                    description="",
                    tags=[],
                    storage_size_megabytes=card.storage_size.in_mb,
                    supports_tensor=card.supports_tensor,
                    tasks=[task.value for task in card.tasks],
                    is_custom=is_custom_card(card.model_id),
                    family=card.family,
                    quantization=card.quantization,
                    base_model=card.base_model,
                    capabilities=card.capabilities,
                )
                for card in cards
            ]
        )

    async def add_custom_model(self, payload: AddCustomModelParams) -> ModelListModel:
        """Fetch a model from HuggingFace and save as a custom model card."""
        try:
            card = await ModelCard.fetch_from_hf(payload.model_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to fetch model: {exc}"
            ) from exc

        return ModelListModel(
            id=card.model_id,
            hugging_face_id=card.model_id,
            name=card.model_id.short(),
            description="",
            tags=[],
            storage_size_megabytes=int(card.storage_size.in_mb),
            supports_tensor=card.supports_tensor,
            tasks=[task.value for task in card.tasks],
            is_custom=True,
        )

    async def delete_custom_model(self, model_id: ModelId) -> JSONResponse:
        """Delete a user-added custom model card."""
        deleted = await delete_custom_card(model_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Custom model card not found")
        return JSONResponse(
            {"message": "Model card deleted", "model_id": str(model_id)}
        )

    async def search_models(
        self, query: str = "", limit: int = 20
    ) -> list[HuggingFaceSearchResult]:
        """Search HuggingFace Hub — tries mlx-community first, falls back to all of HuggingFace."""
        from huggingface_hub import ModelInfo, list_models

        def _to_results(models: Iterable[ModelInfo]) -> list[HuggingFaceSearchResult]:
            return [
                HuggingFaceSearchResult(
                    id=m.id,
                    author=m.author or "",
                    downloads=m.downloads or 0,
                    likes=m.likes or 0,
                    last_modified=str(m.last_modified or ""),
                    tags=list(m.tags or []),
                )
                for m in models
            ]

        # Search mlx-community first
        mlx_results = _to_results(
            list_models(
                search=query or None,
                author="mlx-community",
                sort="downloads",
                limit=limit,
            )
        )
        if mlx_results:
            return mlx_results

        # Fall back to searching all of HuggingFace
        return _to_results(
            list_models(
                search=query or None,
                sort="downloads",
                limit=limit,
            )
        )

    async def run(self):
        shutdown_ev = anyio.Event()

        try:
            async with self._tg as tg:
                logger.info("Starting API")
                tg.start_soon(self._apply_state)
                tg.start_soon(self._pause_on_new_election)
                tg.start_soon(self._cleanup_expired_images)
                print_startup_banner(self.port)
                tg.start_soon(self.run_api, shutdown_ev)
                try:
                    await anyio.sleep_forever()
                finally:
                    with anyio.CancelScope(shield=True):
                        shutdown_ev.set()
        finally:
            self._event_log.close()
            self.command_sender.close()
            self.event_receiver.close()

    async def run_api(self, ev: anyio.Event):
        cfg = Config()
        cfg.bind = [f"0.0.0.0:{self.port}"]
        # nb: shared.logging needs updating if any of this changes
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger
        with anyio.CancelScope(shield=True):
            await serve(
                cast(ASGIFramework, self.app),
                cfg,
                shutdown_trigger=ev.wait,
            )

    async def _apply_state(self):
        with self.event_receiver as events:
            async for i_event in events:
                self._event_log.append(i_event.event)
                self.state = apply(self.state, i_event)
                event = i_event.event

                if isinstance(event, ChunkGenerated):
                    if queue := self._image_generation_queues.get(
                        event.command_id, None
                    ):
                        assert isinstance(event.chunk, ImageChunk)
                        try:
                            await queue.send(event.chunk)
                        except BrokenResourceError:
                            self._image_generation_queues.pop(event.command_id, None)
                    if queue := self._text_generation_queues.get(
                        event.command_id, None
                    ):
                        assert not isinstance(event.chunk, ImageChunk)
                        try:
                            await queue.send(event.chunk)
                        except BrokenResourceError:
                            self._text_generation_queues.pop(event.command_id, None)
                if isinstance(event, TracesMerged):
                    self._save_merged_trace(event)

    def _save_merged_trace(self, event: TracesMerged) -> None:
        traces = [
            TraceEvent(
                name=t.name,
                start_us=t.start_us,
                duration_us=t.duration_us,
                rank=t.rank,
                category=t.category,
            )
            for t in event.traces
        ]
        output_path = EXO_TRACING_CACHE_DIR / f"trace_{event.task_id}.json"
        export_trace(traces, output_path)
        logger.debug(f"Saved merged trace to {output_path}")

    async def _pause_on_new_election(self):
        with self.election_receiver as ems:
            async for message in ems:
                if message.clock > self.last_completed_election:
                    self.paused = True

    async def _cleanup_expired_images(self):
        """Periodically clean up expired images from the store."""
        cleanup_interval_seconds = 300  # 5 minutes
        while True:
            await anyio.sleep(cleanup_interval_seconds)
            removed = self._image_store.cleanup_expired()
            if removed > 0:
                logger.debug(f"Cleaned up {removed} expired images")

    async def _send(self, command: Command):
        while self.paused:
            await self.paused_ev.wait()
        await self.command_sender.send(
            ForwarderCommand(origin=self._system_id, command=command)
        )

    async def _send_download(self, command: DownloadCommand):
        await self.download_command_sender.send(
            ForwarderDownloadCommand(origin=self._system_id, command=command)
        )

    async def start_download(
        self, payload: StartDownloadParams
    ) -> StartDownloadResponse:
        command = StartDownload(
            target_node_id=payload.target_node_id,
            shard_metadata=payload.shard_metadata,
        )
        await self._send_download(command)
        return StartDownloadResponse(command_id=command.command_id)

    async def delete_download(
        self, node_id: NodeId, model_id: ModelId
    ) -> DeleteDownloadResponse:
        command = DeleteDownload(
            target_node_id=node_id,
            model_id=ModelId(model_id),
        )
        await self._send_download(command)
        return DeleteDownloadResponse(command_id=command.command_id)

    @staticmethod
    def _get_trace_path(task_id: str) -> Path:
        trace_path = EXO_TRACING_CACHE_DIR / f"trace_{task_id}.json"
        if not trace_path.resolve().is_relative_to(EXO_TRACING_CACHE_DIR.resolve()):
            raise HTTPException(status_code=400, detail=f"Invalid task ID: {task_id}")
        return trace_path

    async def list_traces(self) -> TraceListResponse:
        traces: list[TraceListItem] = []

        for trace_file in sorted(
            EXO_TRACING_CACHE_DIR.glob("trace_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            # Extract task_id from filename (trace_{task_id}.json)
            task_id = trace_file.stem.removeprefix("trace_")
            stat = trace_file.stat()
            created_at = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            traces.append(
                TraceListItem(
                    task_id=task_id,
                    created_at=created_at,
                    file_size=stat.st_size,
                )
            )

        return TraceListResponse(traces=traces)

    async def get_trace(self, task_id: str) -> TraceResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)

        return TraceResponse(
            task_id=task_id,
            traces=[
                TraceEventResponse(
                    name=event.name,
                    start_us=event.start_us,
                    duration_us=event.duration_us,
                    rank=event.rank,
                    category=event.category,
                )
                for event in trace_events
            ],
        )

    async def get_trace_stats(self, task_id: str) -> TraceStatsResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)
        stats = compute_stats(trace_events)

        return TraceStatsResponse(
            task_id=task_id,
            total_wall_time_us=stats.total_wall_time_us,
            by_category={
                category: TraceCategoryStats(
                    total_us=cat_stats.total_us,
                    count=cat_stats.count,
                    min_us=cat_stats.min_us,
                    max_us=cat_stats.max_us,
                    avg_us=cat_stats.avg_us,
                )
                for category, cat_stats in stats.by_category.items()
            },
            by_rank={
                rank: TraceRankStats(
                    by_category={
                        category: TraceCategoryStats(
                            total_us=cat_stats.total_us,
                            count=cat_stats.count,
                            min_us=cat_stats.min_us,
                            max_us=cat_stats.max_us,
                            avg_us=cat_stats.avg_us,
                        )
                        for category, cat_stats in rank_stats.items()
                    }
                )
                for rank, rank_stats in stats.by_rank.items()
            },
        )

    async def get_trace_raw(self, task_id: str) -> FileResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        return FileResponse(
            path=trace_path,
            media_type="application/json",
            filename=f"trace_{task_id}.json",
        )

    async def delete_traces(self, request: DeleteTracesRequest) -> DeleteTracesResponse:
        deleted: list[str] = []
        not_found: list[str] = []
        for task_id in request.task_ids:
            trace_path = self._get_trace_path(task_id)
            if trace_path.exists():
                trace_path.unlink()
                deleted.append(task_id)
            else:
                not_found.append(task_id)
        return DeleteTracesResponse(deleted=deleted, not_found=not_found)

    async def get_onboarding(self) -> JSONResponse:
        return JSONResponse({"completed": ONBOARDING_COMPLETE_FILE.exists()})

    async def complete_onboarding(self) -> JSONResponse:
        ONBOARDING_COMPLETE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ONBOARDING_COMPLETE_FILE.write_text("true")
        return JSONResponse({"completed": True})
