import base64
import contextlib
import functools
import hashlib
import json
import random
import shutil
import socket
import time
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, Literal, cast, final
from uuid import uuid4

import anyio
import httpx
from anyio import BrokenResourceError, ClosedResourceError, to_thread
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from hypercorn.utils import LifespanTimeoutError, ShutdownError
from loguru import logger

from exo.api.adapters.chat_completions import (
    chat_request_to_text_generation,
    collect_chat_response,
    generate_chat_stream,
)
from exo.api.adapters.claude import (
    claude_request_to_text_generation,
    collect_claude_response,
    generate_claude_stream,
)
from exo.api.adapters.ollama import (
    collect_ollama_chat_response,
    collect_ollama_generate_response,
    generate_ollama_chat_stream,
    generate_ollama_generate_stream,
    ollama_generate_request_to_text_generation,
    ollama_request_to_text_generation,
)
from exo.api.adapters.responses import (
    collect_responses_response,
    generate_responses_stream,
    responses_request_to_text_generation,
)
from exo.api.collected_response import CollectedResponse
from exo.api.command_acceptance import (
    ClusterCommand,
    changed_by,
    command_already_applied,
    command_applied,
    undoing_command,
)
from exo.api.finished_commands import RESEND_INTERVAL_SECONDS, finishes_to_resend
from exo.api.keepalive import with_sse_keepalive
from exo.api.recent_events import RecentEvents
from exo.api.request_logger import RequestLogger
from exo.api.types import (
    AddCustomModelParams,
    AdvancedImageParams,
    AwaitInstanceReadyMessage,
    AwaitInstanceTimeoutMessage,
    BenchChatCompletionRequest,
    BenchChatCompletionResponse,
    BenchImageGenerationResponse,
    BenchImageGenerationTaskParams,
    CancelCommandResponse,
    CancelDownloadParams,
    CancelDownloadResponse,
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
    InstanceLinkBody,
    InstanceLinkResponse,
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
from exo.api.types.claude_api import (
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
)
from exo.api.types.ollama_api import (
    OllamaCapability,
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
from exo.api.types.openai_responses import (
    ResponsesRequest,
    ResponsesResponse,
)
from exo.master.image_store import ImageStore
from exo.master.placement import (
    instance_beyond_memory,
    shard_bytes,
)
from exo.master.placement import place_instance as get_instance_placements
from exo.routing.event_router import NodeEventReceiver
from exo.routing.membership import Membership, MembershipView
from exo.shared.apply import apply
from exo.shared.constants import (
    ENABLE_DISAGGREGATION,
    EXO_CACHE_HOME,
    EXO_EVENT_LOG_DIR,
    EXO_IMAGE_CACHE_DIR,
    EXO_MAX_CHUNK_SIZE,
    EXO_TRACING_CACHE_DIR,
    dashboard_dir,
)
from exo.shared.election import (
    DEFAULT_ELECTION_TIMEOUT,
    HEARTBEAT_INTERVAL,
    MASTER_SILENCE_TIMEOUT,
    ElectionMessage,
)
from exo.shared.logging import InterceptLogger
from exo.shared.model_readiness import (
    MODEL_READY_WAIT_SECONDS,
    NotReady,
    model_readiness,
)
from exo.shared.models import model_cards
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
)
from exo.shared.request_errors import (
    DEVICE_LEFT,
    INSTANCE_GONE_BEFORE_START,
    INTERRUPTED,
    MODEL_STOPPED,
    GenerationErrorCode,
)
from exo.shared.tracing import TraceEvent, compute_stats, export_trace, load_trace_file
from exo.shared.types.chunks import (
    ErrorChunk,
    ImageChunk,
    InputImageChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.commands import (
    AddCustomModelCard,
    CancelDownload,
    Command,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteDownload,
    DeleteInstance,
    DeleteInstanceLink,
    DownloadCommand,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    SendInputChunk,
    SetInstanceLink,
    StartDownload,
    TaskCancelled,
    TaskFinished,
    TextGeneration,
)
from exo.shared.types.common import CommandId, Id, NodeId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    InstanceDeleted,
    StateSnapshot,
    TaskCreated,
    TracesMerged,
)
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.text_generation import (
    Base64ImageHash,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.banner import print_startup_banner
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.power_sampler import PowerSampler
from exo.utils.task_group import TaskGroup

# Where that log used to be written; removed when the API starts
_OLD_API_EVENT_LOG_DIR = EXO_EVENT_LOG_DIR / "api"
ONBOARDING_COMPLETE_FILE = EXO_CACHE_HOME / "onboarding_complete"
# How often a request waiting for its model to be ready checks on it
_MODEL_READY_POLL_SECONDS = 0.2


# How long to wait for the cluster to accept a chat request before sending it again. A busy
# node can take a while to see a request accepted (up to 18 s on four nodes streaming 800
# events/s), and resending sooner only adds traffic. After about a minute the request ends
# with an error instead of waiting forever.
REQUEST_RESEND_WAITS = (10.0, 15.0, 30.0)
_NOT_ACCEPTED_MESSAGE = "The cluster didn't accept the request. Please check that the model is running and try again."

# A request that changes the cluster (launching or stopping a model, say) is answered once this
# node's state shows the change. Until then its command is sent again this often: it can be
# dropped under load, and the master can die before applying it.
CLUSTER_COMMAND_RESEND_INTERVAL = 5.0
# ...and the request fails after this long. It covers losing the master: followers notice a
# silent master within MASTER_SILENCE_TIMEOUT plus one heartbeat check, elect a new one (a
# round, and maybe a second if the first is interrupted) and catch up from its state, where
# the command is sent again.
CLUSTER_COMMAND_TIMEOUT = (
    MASTER_SILENCE_TIMEOUT + HEARTBEAT_INTERVAL + 2 * DEFAULT_ELECTION_TIMEOUT + 5.0
)
_NOT_APPLIED_MESSAGE = (
    "The cluster didn't confirm this change in time; the device coordinating it may have "
    "been lost. Check the cluster's state and try again."
)
# ...for a launch or a new link, which this node then calls off (see WithdrawnClusterCommand)
_WITHDRAWN_MESSAGE = (
    "The cluster didn't confirm this change in time; the device coordinating it may have "
    "been lost. The change was called off, so trying again won't make it twice."
)
# ...for a launch. The master also refuses, in its own log only, a launch its devices don't
# have the memory for: another launch sent at the same time can take the memory after this
# node checked. Nothing tells this node, so it learns of it only as the launch not showing.
_LAUNCH_WITHDRAWN_MESSAGE = (
    "The cluster didn't confirm this launch in time: the model may not have fitted in its "
    "devices' memory (another launch got there first), or the device coordinating the "
    "cluster may have been lost. The launch was called off, so trying again won't start "
    "it twice."
)


def _format_to_content_type(image_format: Literal["png", "jpeg", "webp"] | None) -> str:
    return f"image/{image_format or 'png'}"


def _image_stream_error_event(message: str) -> str:
    error_response = ErrorResponse(
        error=ErrorInfo(message=message, type="InternalServerError", code=500)
    )
    return f"data: {error_response.model_dump_json()}\n\n"


def _ensure_seed(params: AdvancedImageParams | None) -> AdvancedImageParams:
    """Ensure advanced params has a seed set for distributed consistency."""
    if params is None:
        return AdvancedImageParams(seed=random.randint(0, 2**32 - 1))
    if params.seed is None:
        return params.model_copy(update={"seed": random.randint(0, 2**32 - 1)})
    return params


def _require_disaggregation_enabled() -> None:
    if not ENABLE_DISAGGREGATION:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=(
                "Prefill/decode disaggregation is disabled. "
                "Set ENABLE_DISAGGREGATION=true to enable."
            ),
        )


@final
class ModelNotReadyError(HTTPException):
    """A 503 for a request whose model isn't ready to serve"""

    code: GenerationErrorCode = "model_not_ready"

    def __init__(self, not_ready: NotReady):
        super().__init__(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=not_ready.message,
            headers={"Retry-After": str(not_ready.retry_after_seconds)},
        )


@dataclass
class PendingClusterCommand:
    """A command that changes the cluster, waiting to be seen applied."""

    command: ClusterCommand
    applied: bool = False
    # Set when the command is seen applied, or should be sent again at once
    wake: anyio.Event = field(default_factory=anyio.Event)


@dataclass
class WithdrawnClusterCommand:
    """A launch or new link whose request was answered 503. A copy can still apply later
    (delivered late, or seen once a split heals); it is then undone, so a client that tried
    again doesn't end up with two."""

    command: ClusterCommand
    undo: ClusterCommand
    # Created when a copy applies and the undo starts; set once the undo has ended
    undo_finished: anyio.Event | None = None
    # When the request was answered 503 (time.monotonic())
    withdrawn_at: float = field(default_factory=time.monotonic)


class API:
    def __init__(
        self,
        node_id: NodeId,
        *,
        port: int,
        # Already bound to port and listening, see listen_on_api_port
        listening_socket: socket.socket,
        event_receiver: NodeEventReceiver,
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        # This lets us pause the API if an election is running
        election_receiver: Receiver[ElectionMessage],
        membership: Membership,
    ) -> None:
        self.state = State()
        self._recent_events = RecentEvents()
        self._system_id = SystemId()
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.event_receiver = event_receiver
        self.election_receiver = election_receiver
        self.membership = membership
        self.node_id: NodeId = node_id
        self.last_completed_election: int = 0
        self.port = port
        self._listening_socket = listening_socket

        self.paused: bool = False
        self.paused_ev: anyio.Event = anyio.Event()

        self.app = FastAPI()

        self.app.add_middleware(RequestLogger)

        self._setup_exception_handlers()
        self._setup_cors()
        self._setup_routes()

        self.app.mount(
            "/",
            StaticFiles(
                directory=dashboard_dir(),
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
        # Cancelling closes a command's stream just like losing its instance
        # does, but it was requested, so the stream should not report an error
        self._cancelled_command_ids: set[CommandId] = set()
        self._image_store = ImageStore(EXO_IMAGE_CACHE_DIR)
        self._tg: TaskGroup = TaskGroup()

    def reset(self, result_clock: int, event_receiver: NodeEventReceiver):
        logger.info("Resetting API State")
        self._recent_events = RecentEvents()
        self.state = State()
        self._system_id = SystemId()
        # Nothing routes chunks to the old queues any more, so close them to
        # end their requests with an error instead of leaving them hanging
        self._shutdown_queues(self._text_generation_queues)
        self._shutdown_queues(self._image_generation_queues)
        self._text_generation_queues = {}
        self._image_generation_queues = {}
        self.unpause(result_clock)
        self.event_receiver.close()
        self.event_receiver = event_receiver
        self._tg.start_soon(self._apply_state)

    def follow_new_master(self, result_clock: int) -> None:
        """Carry on under a new master that took over the cluster's state: keep ours until
        its snapshot replaces it, so the running models stay usable, but end the requests
        in flight, whose output the old master was routing."""
        logger.info("Following a new master")
        self._system_id = SystemId()
        self._shutdown_queues(self._text_generation_queues)
        self._shutdown_queues(self._image_generation_queues)
        self._text_generation_queues = {}
        self._image_generation_queues = {}
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
                code=exc.code
                if isinstance(exc, ModelNotReadyError)
                else exc.status_code,
            )
        )
        return JSONResponse(
            err.model_dump(), status_code=exc.status_code, headers=exc.headers
        )

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
        self.app.get("/instance/await", response_model=None)(self.await_instance)
        self.app.get("/instance/{instance_id}")(self.get_instance)
        self.app.delete("/instance/{instance_id}")(self.delete_instance)
        self.app.get("/v1/instance-links")(self.list_instance_links)
        self.app.post("/v1/instance-links")(self.create_instance_link)
        self.app.put("/v1/instance-links/{link_id}")(self.update_instance_link)
        self.app.delete("/v1/instance-links/{link_id}")(self.delete_instance_link)
        self.app.get("/v1/feature-flags")(self.get_feature_flags)
        self.app.get("/models")(self.get_models)
        self.app.get("/v1/models")(self.get_models)
        self.app.post("/models/add")(self.add_custom_model)
        self.app.delete("/models/custom/{model_id:path}")(self.delete_custom_model)
        self.app.get("/models/search")(self.search_models)
        self.app.post("/v1/chat/completions", response_model=None)(
            self.chat_completions
        )
        self.app.post("/bench/chat/completions", response_model=None)(
            self.bench_chat_completions
        )
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
        self.app.post("/ollama/v1/chat/completions", response_model=None)(
            self.chat_completions
        )
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

        self.app.get("/state")(self.get_state)
        self.app.get("/membership")(self.get_membership)
        self.app.get("/state/{path:path}")(self.get_state)
        self.app.get("/events")(self.get_events)
        self.app.post("/download/start")(self.start_download)
        self.app.delete("/download/{node_id}/{model_id:path}")(self.delete_download)
        self.app.post("/download/cancel")(self.cancel_download)
        self.app.get("/v1/traces")(self.list_traces)
        self.app.post("/v1/traces/delete")(self.delete_traces)
        self.app.get("/v1/traces/{task_id}")(self.get_trace)
        self.app.get("/v1/traces/{task_id}/stats")(self.get_trace_stats)
        self.app.get("/v1/traces/{task_id}/raw")(self.get_trace_raw)
        self.app.get("/onboarding")(self.get_onboarding)
        self.app.post("/onboarding")(self.complete_onboarding)

    def get_membership(self) -> MembershipView:
        """This node's own view of which nodes it can reach now (not the cluster's state: two
        nodes may disagree for a moment)"""
        return self.membership.view()

    def get_state(self, path: str = ""):
        if path == "":
            return self.state
        try:
            x = self.state.model_dump(by_alias=True)
            for attr in path.split("/"):
                if attr != "":
                    if isinstance(x, dict):
                        x = x[attr]  # pyright: ignore[reportUnknownVariableType]
                    elif isinstance(x, list):
                        x = x[int(attr)]  # pyright: ignore[reportUnknownVariableType]
            return cast(Any, x)  # pyright: ignore[reportAny]
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"unable to find path '{path.replace('/', '.')}' in state json",
            ) from e

    async def place_instance(self, payload: PlaceInstanceParams):
        # Refuse, with the reason, a placement the cluster can't hold. The master would
        # reject the command too, but only in its own log, so the caller never found out.
        placement = await self.get_placement(
            payload.model_id,
            payload.sharding,
            payload.instance_meta,
            payload.min_nodes,
        )
        # The placement made here is launched as it is, rather than asking the master to
        # place the model again: an instance with a known id can be sent again until it shows
        # up (to a new master, say) without ever starting two.
        command = CreateInstance(instance=placement)
        await self._apply_cluster_command(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_card=await ModelCard.load(payload.model_id),
        )

    async def create_instance(
        self, payload: CreateInstanceParams
    ) -> CreateInstanceResponse:
        instance = payload.instance
        model_card = await ModelCard.load(instance.shard_assignments.model_id)
        # Each node has to hold its own shard: memory free elsewhere is no use to it
        if (shortfall := instance_beyond_memory(instance, self.state)) is not None:
            raise HTTPException(status_code=400, detail=shortfall)

        command = CreateInstance(
            instance=instance,
        )
        await self._apply_cluster_command(command)

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
                node_backends=self.state.node_backends,
                topology=self.state.topology,
                current_instances=self.state.instances,
                download_status=self.state.downloads,
                node_rdma_ctl=self.state.node_rdma_ctl,
                runners=self.state.runners,
                node_identities=self.state.node_identities,
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
                    node_backends=self.state.node_backends,
                    topology=self.state.topology,
                    current_instances=self.state.instances,
                    required_nodes=required_nodes,
                    download_status=self.state.downloads,
                    node_rdma_ctl=self.state.node_rdma_ctl,
                    runners=self.state.runners,
                    node_identities=self.state.node_identities,
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

            # Each node's share: pipeline gives a node layers in proportion to its memory
            memory_delta_by_node = {
                str(node_id): shard_bytes(shard_assignments.runner_to_shard[runner_id])
                for node_id, runner_id in shard_assignments.node_to_runner.items()
            }

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

    async def await_instance(
        self,
        model_id: ModelId,
        timeout_seconds: float = Query(default=0.0, ge=0.0, le=300.0),
    ) -> StreamingResponse:
        _sleep = 0.1

        async def _stream() -> AsyncGenerator[str, None]:
            deadline = (
                None if timeout_seconds == 0 else anyio.current_time() + timeout_seconds
            )

            while True:
                for instance in self.state.instances.values():
                    if instance.shard_assignments.model_id == model_id:
                        payload = AwaitInstanceReadyMessage(instance=instance)
                        yield f"data: {payload.model_dump_json()}\n\n"
                        return

                if deadline is None:
                    await anyio.sleep(_sleep)
                else:
                    remaining = deadline - anyio.current_time()
                    if remaining <= 0:
                        payload = AwaitInstanceTimeoutMessage(
                            message=f"No instance found for model {model_id}"
                        )
                        yield f"data: {payload.model_dump_json()}\n\n"
                        return

                    await anyio.sleep(min(_sleep, remaining))

        return StreamingResponse(
            with_sse_keepalive(_stream()),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )

    async def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")

        command = DeleteInstance(
            instance_id=instance_id,
        )
        await self._apply_cluster_command(command)
        return DeleteInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance_id,
        )

    async def get_feature_flags(self) -> dict[str, bool]:
        return {"disaggregation": ENABLE_DISAGGREGATION}

    async def list_instance_links(self) -> list[InstanceLink]:
        if not ENABLE_DISAGGREGATION:
            return []
        return list(self.state.instance_links.values())

    async def create_instance_link(
        self, body: InstanceLinkBody
    ) -> InstanceLinkResponse:
        _require_disaggregation_enabled()
        return await self._set_instance_link(InstanceLinkId(), body)

    async def update_instance_link(
        self, link_id: InstanceLinkId, body: InstanceLinkBody
    ) -> InstanceLinkResponse:
        _require_disaggregation_enabled()
        return await self._set_instance_link(link_id, body)

    async def _set_instance_link(
        self, link_id: InstanceLinkId, body: InstanceLinkBody
    ) -> InstanceLinkResponse:
        command = SetInstanceLink(
            link_id=link_id,
            prefill_instances=list(body.prefill_instances),
            decode_instances=list(body.decode_instances),
        )
        await self._apply_cluster_command(command)
        return InstanceLinkResponse(
            message="Command received.", command_id=command.command_id
        )

    async def delete_instance_link(
        self, link_id: InstanceLinkId
    ) -> InstanceLinkResponse:
        _require_disaggregation_enabled()
        command = DeleteInstanceLink(link_id=link_id)
        await self._apply_cluster_command(command)
        return InstanceLinkResponse(
            message="Command received.", command_id=command.command_id
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

        self._cancelled_command_ids.add(command_id)
        await self._send(TaskCancelled(cancelled_command_id=command_id))
        sender.close()

        return CancelCommandResponse(
            message="Command cancelled.",
            command_id=command_id,
        )

    async def _token_chunk_stream(
        self, text_generation: TextGeneration
    ) -> AsyncGenerator[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None
    ]:
        """Yield chunks for a given command until completion.

        This is the internal low-level stream used by all API adapters.
        """
        command_id = text_generation.command_id
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
                else:
                    # The stream was closed before the generation finished
                    if command_id not in self._cancelled_command_ids:
                        yield ErrorChunk(
                            model=text_generation.task_params.model,
                            error_message=self._stream_end_reasons.get(
                                command_id, INTERRUPTED
                            ),
                        )

        except anyio.get_cancelled_exc_class():
            await self._send_from_cleanup(
                TaskCancelled(cancelled_command_id=command_id)
            )
            raise
        finally:
            await self._finish(command_id)
            if command_id in self._text_generation_queues:
                del self._text_generation_queues[command_id]
            self._cancelled_command_ids.discard(command_id)
            self._stream_end_reasons.pop(command_id, None)

    async def _collect_text_generation_with_stats(
        self, text_generation: TextGeneration
    ) -> BenchChatCompletionResponse:
        sampler = PowerSampler(get_node_system=lambda: self.state.node_system)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        model: ModelId | None = None
        finish_reason: FinishReason | None = None

        stats: GenerationStats | None = None

        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)

            async for chunk in self._token_chunk_stream(text_generation):
                if isinstance(chunk, PrefillProgressChunk):
                    continue

                sampler.mark_prefill_done()

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
            id=text_generation.command_id,
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

    async def _send_text_generation_with_images(
        self, task_params: TextGenerationTaskParams
    ) -> TextGeneration:
        task_params = task_params.with_card_sampling_defaults()
        images = task_params.images
        if not images:
            command = TextGeneration(task_params=task_params)
            await self._send(command)
            return command

        hashes = [hashlib.sha256(img.encode("ascii")).hexdigest() for img in images]
        all_hashes = {idx: Base64ImageHash(h) for idx, h in enumerate(hashes)}
        task_params = task_params.model_copy(
            update={"images": [], "image_hashes": all_hashes}
        )
        command = TextGeneration(task_params=task_params)

        # Every request carries its own images. Nodes keep them only while they need them,
        # and a node that restarted, joined later or caught up from a state snapshot never
        # saw an earlier request's images anyway.
        all_chunks: list[tuple[int, str]] = []
        for img_idx, img_data in enumerate(images):
            for i in range(0, len(img_data), EXO_MAX_CHUNK_SIZE):
                all_chunks.append((img_idx, img_data[i : i + EXO_MAX_CHUNK_SIZE]))

        for global_idx, (img_idx, chunk_data) in enumerate(all_chunks):
            await self._send(
                SendInputChunk(
                    chunk=InputImageChunk(
                        model=task_params.model,
                        command_id=command.command_id,
                        data=chunk_data,
                        chunk_index=global_idx,
                        total_chunks=len(all_chunks),
                        image_index=img_idx,
                    )
                )
            )

        await self._send(command)
        return command

    async def chat_completions(
        self, payload: ChatCompletionRequest
    ) -> ChatCompletionResponse | StreamingResponse:
        """OpenAI Chat Completions API - adapter."""
        task_params = await chat_request_to_text_generation(payload)
        validated_model = await self._validate_model_has_instance(task_params.model)
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_chat_stream(
                        command.command_id,
                        self._token_chunk_stream(command),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return CollectedResponse(
                collect_chat_response(
                    command.command_id,
                    self._token_chunk_stream(command),
                ),
                media_type="application/json",
            )

    async def bench_chat_completions(
        self, payload: BenchChatCompletionRequest
    ) -> BenchChatCompletionResponse | StreamingResponse:
        task_params = await chat_request_to_text_generation(payload)
        validated_model = await self._validate_model_has_instance(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        task_params = task_params.model_copy(
            update={
                "stream": False,
                "bench": True,
                "use_prefix_cache": payload.use_prefix_cache,
            }
        )

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_chat_stream(
                        command.command_id,
                        self._token_chunk_stream(command),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )

        return await self._collect_text_generation_with_stats(command)

    async def _validate_model_has_instance(self, model_id: ModelId) -> ModelId:
        """Validate a model has an instance that is ready to serve.
        If the model isn't even downloaded, triggers notification to user to download model.

        A model that isn't ready is waited for, up to MODEL_READY_WAIT_SECONDS
        (EXO_MODEL_READY_WAIT_SECONDS).

        Raises HTTPException 404 if no instance is found for the model, and 503 (with
        Retry-After, and what the model is doing) if it isn't ready after the wait, or failed
        to start.
        """
        await self._validate_model_has_any_instance(model_id)
        deadline = anyio.current_time() + MODEL_READY_WAIT_SECONDS
        while (not_ready := model_readiness(self.state, model_id)) is not None:
            if not_ready.failed or anyio.current_time() >= deadline:
                raise ModelNotReadyError(not_ready)
            await anyio.sleep(_MODEL_READY_POLL_SECONDS)
            # It may have been deleted meanwhile
            await self._validate_model_has_any_instance(model_id)
        return model_id

    async def _validate_model_has_any_instance(self, model_id: ModelId) -> None:
        if not any(
            instance.shard_assignments.model_id == model_id
            for instance in self.state.instances.values()
        ):
            # Check if model is actually downloaded
            model_is_downloaded = any(
                isinstance(download, DownloadCompleted)
                and download.shard_metadata.model_card.model_id == model_id
                for node_downloads in self.state.downloads.values()
                for download in node_downloads
            )
            if not model_is_downloaded:
                await self._trigger_notify_user_to_download_model(model_id)

            raise HTTPException(
                status_code=404, detail=f"No instance found for model {model_id}"
            )

    async def get_events(self) -> Response:
        # A copy: events keep arriving while it is serialized
        events = list(self._recent_events)
        # Built in one go off the event loop: streaming each event as its own chunk took
        # over 5 s for 10,000 events on a busy node
        body = await to_thread.run_sync(
            lambda: "[" + ",".join(event.model_dump_json() for event in events) + "]"
        )
        return Response(content=body, media_type="application/json")

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
                "model": await self._validate_model_has_instance(
                    ModelId(payload.model)
                ),
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
                        yield _image_stream_error_event(
                            chunk.error_message or "Internal server error"
                        )
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
                else:
                    # The stream was closed before every image was generated
                    if command_id not in self._cancelled_command_ids:
                        yield _image_stream_error_event(
                            self._stream_end_reasons.get(command_id, INTERRUPTED)
                        )
                        yield "data: [DONE]\n\n"

        except anyio.get_cancelled_exc_class():
            await self._send_from_cleanup(
                TaskCancelled(cancelled_command_id=command_id)
            )
            raise
        finally:
            await self._finish(command_id)
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]
            self._cancelled_command_ids.discard(command_id)
            self._stream_end_reasons.pop(command_id, None)

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
                    else:
                        # The stream was closed before every image was generated
                        raise HTTPException(
                            status_code=500,
                            detail="Command cancelled."
                            if command_id in self._cancelled_command_ids
                            else self._stream_end_reasons.get(command_id, INTERRUPTED),
                        )

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
            await self._send_from_cleanup(
                TaskCancelled(cancelled_command_id=command_id)
            )
            raise
        finally:
            await self._finish(command_id)
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]
            self._cancelled_command_ids.discard(command_id)
            self._stream_end_reasons.pop(command_id, None)

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
                "model": await self._validate_model_has_instance(
                    ModelId(payload.model)
                ),
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
        validated_model = await self._validate_model_has_instance(model)
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
                model=validated_model,
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
                        model=validated_model,
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
        task_params = await claude_request_to_text_generation(payload)
        validated_model = await self._validate_model_has_instance(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_claude_stream(
                        command.command_id,
                        payload.model,
                        self._token_chunk_stream(command),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return CollectedResponse(
                collect_claude_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command),
                ),
                media_type="application/json",
            )

    async def openai_responses(
        self, payload: ResponsesRequest
    ) -> ResponsesResponse | StreamingResponse:
        """OpenAI Responses API."""
        task_params = await responses_request_to_text_generation(payload)
        validated_model = await self._validate_model_has_instance(task_params.model)
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_responses_stream(
                        command.command_id,
                        payload.model,
                        self._token_chunk_stream(command),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            return CollectedResponse(
                collect_responses_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command),
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
        validated_model = await self._validate_model_has_instance(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_chat_stream(
                    command.command_id,
                    self._token_chunk_stream(command),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return CollectedResponse(
                collect_ollama_chat_response(
                    command.command_id,
                    self._token_chunk_stream(command),
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
        validated_model = await self._validate_model_has_instance(
            ModelId(task_params.model)
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_generate_stream(
                    command.command_id,
                    self._token_chunk_stream(command),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return CollectedResponse(
                collect_ollama_generate_response(
                    command.command_id,
                    self._token_chunk_stream(command),
                ),
                media_type="application/json",
            )

    async def ollama_tags(self) -> OllamaTagsResponse:
        """Returns list of models in Ollama tags format. We return the downloaded ones only."""

        downloaded_model_ids: set[ModelId] = set()
        for node_downloads in self.state.downloads.values():
            for dl in node_downloads:
                if isinstance(dl, DownloadCompleted):
                    downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)

        cards = [
            c
            for c in await model_cards.card_cache.list_all()
            if c.model_id in downloaded_model_ids
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
                        family=card.family or None,
                        quantization_level=card.quantization or None,
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

        capabilities: list[OllamaCapability] = []
        if ModelTask.TextGeneration in card.tasks:
            capabilities.extend(("completion", "tools"))
        if card.vision is not None:
            capabilities.append("vision")

        architecture = card.family or "unknown"
        model_info: dict[str, Any] = {
            "general.architecture": architecture,
            "general.basename": card.base_model or str(card.model_id),
        }
        if card.context_length > 0:
            model_info[f"{architecture}.context_length"] = card.context_length

        return OllamaShowResponse(
            modelfile=f"FROM {card.model_id}",
            template="{{ .Prompt }}",
            details=OllamaModelDetails(
                family=card.family or None,
                quantization_level=card.quantization or None,
            ),
            model_info=model_info,
            capabilities=capabilities,
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
        return {"version": "1.0.0"}

    async def get_models(self, status: str | None = Query(default=None)) -> ModelList:
        """Returns list of available models, optionally filtered by being downloaded."""
        cards = await model_cards.card_cache.list_all()

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
                    is_custom=card.is_custom,
                    family=card.family,
                    quantization=card.quantization,
                    base_model=card.base_model,
                    capabilities=card.capabilities,
                    reasoning_dialect=card.reasoning_dialect,
                    context_length=card.context_length,
                )
                for card in cards
            ]
        )

    async def add_custom_model(self, payload: AddCustomModelParams) -> ModelListModel:
        """Fetch a model from HuggingFace and save as a custom model card, then sync across the cluster."""
        try:
            card = await ModelCard.fetch_from_hf(payload.model_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to fetch model: {exc}"
            ) from exc

        await self._apply_cluster_command(AddCustomModelCard(model_card=card))

        # Immediately update the local cache so the subsequent GET /models
        # returns the new model without waiting for the event round-trip.
        model_cards.card_cache.cc[card.model_id] = card

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
        """Delete a user-added custom model card and sync deletion across the cluster."""
        card = model_cards.card_cache.get(model_id)
        if card is None or not card.is_custom:
            raise HTTPException(status_code=404, detail="Custom model card not found")

        await self._apply_cluster_command(DeleteCustomModelCard(model_id=model_id))

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
                await to_thread.run_sync(
                    partial(shutil.rmtree, _OLD_API_EVENT_LOG_DIR, ignore_errors=True)
                )
                tg.start_soon(self._apply_state)
                tg.start_soon(self._pause_on_new_election)
                tg.start_soon(self._cleanup_expired_images)
                tg.start_soon(self._resend_lost_finishes)
                tg.start_soon(self.run_api, shutdown_ev)
                tg.start_soon(self._print_banner_once_serving)
                try:
                    await anyio.sleep_forever()
                finally:
                    with anyio.CancelScope(shield=True):
                        # IMPORTANT: when new queues are added, update this (for proper shutdown semantics)
                        self._shutdown_queues(self._text_generation_queues)
                        self._shutdown_queues(self._image_generation_queues)

                        shutdown_ev.set()
        finally:
            self.command_sender.close()
            self.event_receiver.close()

    @staticmethod
    def _shutdown_queues[K, V](queues: dict[K, Sender[V]]):
        for v in queues.values():
            v.close()

    async def run_api(self, ev: anyio.Event):
        cfg = Config()
        # Serve the socket the node bound when it started; hypercorn takes ownership of it
        cfg.bind = [f"fd://{self._listening_socket.detach()}"]
        # nb: shared.logging needs updating if any of this changes
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger

        # prevents hangs when mid-request and connection refuses to close
        cfg.graceful_timeout = 2  # seconds
        cfg.shutdown_timeout = 3  # seconds

        with anyio.CancelScope(shield=True):
            try:
                await serve(
                    cast(ASGIFramework, self.app),
                    cfg,
                    shutdown_trigger=ev.wait,
                )
                if not ev.is_set():
                    raise ShutdownError(
                        "Server exited without shutdown trigger - exiting abnormally"
                    )
            except LifespanTimeoutError as e:
                logger.warning(
                    "Graceful server shutdown timed out, some connections forcebly closed"
                )
                logger.opt(exception=e).debug("")

    async def _print_banner_once_serving(self) -> None:
        """Print the startup banner once the API answers requests, so its URL works when shown."""
        async with httpx.AsyncClient(trust_env=False) as client:
            while True:
                try:
                    await client.get(f"http://127.0.0.1:{self.port}/node_id", timeout=1)
                    break
                except httpx.TransportError:
                    await anyio.sleep(0.05)
        print_startup_banner(self.port)

    async def _apply_state(self):
        with self.event_receiver as events:
            async for i_event in events:
                if isinstance(i_event, StateSnapshot):
                    self._apply_snapshot(i_event)
                    continue
                self._recent_events.append(i_event.event)
                previous_state = self.state
                self.state = apply(self.state, i_event)
                self._note_applied_cluster_commands()
                event = i_event.event

                if isinstance(event, TaskCreated) and isinstance(
                    event.task,
                    (TextGenerationTask, ImageGenerationTask, ImageEditsTask),
                ):
                    self._mark_accepted(event.task.command_id)
                    if event.task_id not in self.state.tasks:
                        await self._end_request_not_started(event.task)
                if isinstance(event, ChunkGenerated):
                    self._mark_accepted(event.command_id)
                    if queue := self._image_generation_queues.get(
                        event.command_id, None
                    ):
                        # Image runners report failures as ErrorChunks
                        assert isinstance(event.chunk, (ImageChunk, ErrorChunk))
                        try:
                            await queue.send(event.chunk)
                        except (BrokenResourceError, ClosedResourceError):
                            self._image_generation_queues.pop(event.command_id, None)
                    if queue := self._text_generation_queues.get(
                        event.command_id, None
                    ):
                        assert not isinstance(event.chunk, ImageChunk)
                        try:
                            await queue.send(event.chunk)
                        except (BrokenResourceError, ClosedResourceError):
                            self._text_generation_queues.pop(event.command_id, None)
                if isinstance(event, InstanceDeleted):
                    self._close_streams_for_instance(
                        event.instance_id,
                        self._why_deleted(
                            previous_state.instances.get(event.instance_id)
                        ),
                    )
                if isinstance(event, TracesMerged):
                    self._save_merged_trace(event)

    def _apply_snapshot(self, snapshot: StateSnapshot) -> None:
        self.state = snapshot.state
        self._note_applied_cluster_commands()
        # A snapshot comes from a master that is up and answering: usually a new one, which
        # never saw the commands sent to the old one. Send them again now rather than at the
        # next resend.
        for pending in self._pending_cluster_commands.values():
            pending.wake.set()
        # Output for in-flight requests may have been in the events we skipped
        in_flight = [
            *self._text_generation_queues.values(),
            *self._image_generation_queues.values(),
        ]
        if in_flight:
            logger.warning(
                f"Ending {len(in_flight)} in-flight request(s) that may have missed "
                "output while this node caught up from a state snapshot"
            )
        for sender in in_flight:
            sender.close()
        self._text_generation_queues.clear()
        self._image_generation_queues.clear()

    async def _end_request_not_started(
        self, task: TextGenerationTask | ImageGenerationTask | ImageEditsTask
    ) -> None:
        """End a request whose task the cluster didn't add, its instance having been deleted
        just before. The deletion ended the instance's requests before this one had a task,
        and nothing else would end it."""
        text_queue = self._text_generation_queues.get(task.command_id)
        image_queue = self._image_generation_queues.get(task.command_id)
        if text_queue is None and image_queue is None:
            return  # sent through another node's API
        logger.warning(
            f"Ending request {task.command_id}: it reached instance {task.instance_id} "
            "after the instance was deleted"
        )
        error = ErrorChunk(
            model=ModelId(task.task_params.model),
            error_message=INSTANCE_GONE_BEFORE_START,
        )
        with contextlib.suppress(BrokenResourceError, ClosedResourceError):
            if text_queue is not None:
                await text_queue.send(error)
            if image_queue is not None:
                await image_queue.send(error)

    def _why_deleted(self, instance: Instance | None) -> str:
        """Why an instance was deleted, as its requests are told. The master deletes an
        instance as soon as one of its nodes is gone, after removing the node from the
        topology if it timed out, so a node missing from the topology now means it left."""
        if instance is None:
            return INTERRUPTED
        if any(
            not self.state.topology.contains_node(node_id)
            for node_id in instance.shard_assignments.node_to_runner
        ):
            return DEVICE_LEFT
        return MODEL_STOPPED

    def _close_streams_for_instance(
        self,
        instance_id: InstanceId,
        reason: str = INTERRUPTED,
    ) -> None:
        """Close any active generation streams for commands running on the given instance."""
        for task in self.state.tasks.values():
            if task.instance_id != instance_id:
                continue
            if not isinstance(
                task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask)
            ):
                continue
            if sender := self._text_generation_queues.pop(task.command_id, None):
                self._stream_end_reasons[task.command_id] = reason
                sender.close()
            if sender := self._image_generation_queues.pop(task.command_id, None):
                self._stream_end_reasons[task.command_id] = reason
                sender.close()

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
                # A heartbeat can carry a newer clock without an election result following it
                if message.heartbeat:
                    continue
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

    @functools.cached_property
    def _stream_end_reasons(self) -> dict[CommandId, str]:
        """Why the API closed a request's stream before its generation finished, when it
        knows: the message the request ends with."""
        return {}

    @functools.cached_property
    def _awaiting_acceptance(self) -> dict[CommandId, anyio.Event]:
        """Generation requests (chats and images) sent but not yet seen as a task."""
        return {}

    @functools.cached_property
    def _request_inputs(self) -> dict[CommandId, list[SendInputChunk]]:
        """The image chunks sent ahead of generation requests not yet seen as a task."""
        return {}

    @functools.cached_property
    def _pending_cluster_commands(self) -> dict[CommandId, PendingClusterCommand]:
        """Commands that change the cluster, sent but not yet seen applied."""
        return {}

    @functools.cached_property
    def _withdrawn_cluster_commands(
        self,
    ) -> dict[InstanceId | InstanceLinkId | ModelId, WithdrawnClusterCommand]:
        """Launches and new links answered 503, by what they change, to undo should they still
        apply. Kept until then, or until a request about the same instance or link."""
        return {}

    @functools.cached_property
    def _finished_at(self) -> dict[CommandId, float]:
        """When this API finished each request whose task may still need a TaskFinished. Kept
        across resets: a change of master is when finishes get lost."""
        return {}

    async def _finish(self, command_id: CommandId) -> None:
        self._finished_at[command_id] = time.monotonic()
        await self._send_from_cleanup(TaskFinished(finished_command_id=command_id))

    async def _resend_lost_finishes(self) -> None:
        while True:
            await anyio.sleep(RESEND_INTERVAL_SECONDS)
            resend, forget = finishes_to_resend(
                self._finished_at, self.state, time.monotonic()
            )
            for command_id in forget:
                del self._finished_at[command_id]
            for command_id in resend:
                logger.info(
                    f"Resending the finish of {command_id}: its task is still in state"
                )
                await self._send(TaskFinished(finished_command_id=command_id))

    async def _send(self, command: Command):
        await self._forward(command)
        match command:
            case SendInputChunk(chunk=chunk):
                self._request_inputs.setdefault(chunk.command_id, []).append(command)
            case TextGeneration() | ImageGeneration() | ImageEdits() if (
                command.command_id not in self._awaiting_acceptance
                and self._tg.is_running()
            ):
                self._awaiting_acceptance[command.command_id] = anyio.Event()
                self._tg.start_soon(self._resend_until_accepted, command)
            case _:
                pass

    async def _forward(self, command: Command) -> None:
        while self.paused:
            await self.paused_ev.wait()
        await self.command_sender.send(
            ForwarderCommand(origin=self._system_id, command=command)
        )

    async def _send_from_cleanup(self, command: Command) -> None:
        # Stream cleanup runs after a client disconnect has cancelled the
        # stream, so an unshielded send would be cancelled before it reaches
        # the master. Skip _send's election pause too: a shielded wait on it
        # could block shutdown forever.
        with anyio.CancelScope(shield=True):
            await self.command_sender.send(
                ForwarderCommand(origin=self._system_id, command=command)
            )

    async def _resend_until_accepted(
        self, command: TextGeneration | ImageGeneration | ImageEdits
    ) -> None:
        """Send a generation request (a chat or an image) again until the cluster has
        accepted it.

        A command can be lost on its way to the master: dropped under load, or sent just as
        the master changed. The request would then wait forever for output that never comes.
        The master ignores a command it has already processed, so sending it again can't
        start the request twice.
        """
        command_id = command.command_id
        accepted = self._awaiting_acceptance[command_id]
        try:
            for wait in REQUEST_RESEND_WAITS:
                with anyio.move_on_after(wait):
                    await accepted.wait()
                    return
                if (
                    command_id not in self._text_generation_queues
                    and command_id not in self._image_generation_queues
                ):
                    return  # the request has already ended
                logger.warning(
                    f"Request {command_id} not accepted after {wait:.0f}s; sending it again"
                )
                for chunk in self._request_inputs.get(command_id, []):
                    await self._forward(chunk)
                await self._forward(command)
            not_accepted = ErrorChunk(
                model=ModelId(command.task_params.model),
                error_message=_NOT_ACCEPTED_MESSAGE,
            )
            text_queue = self._text_generation_queues.get(command_id)
            image_queue = self._image_generation_queues.get(command_id)
            if text_queue is not None or image_queue is not None:
                logger.warning(f"Giving up on request {command_id}: never accepted")
            with contextlib.suppress(BrokenResourceError, ClosedResourceError):
                if text_queue is not None:
                    await text_queue.send(not_accepted)
                if image_queue is not None:
                    await image_queue.send(not_accepted)
        finally:
            self._awaiting_acceptance.pop(command_id, None)
            self._request_inputs.pop(command_id, None)

    async def _apply_cluster_command(self, command: ClusterCommand) -> None:
        """Send a command that changes the cluster, and return once this node's state shows
        it applied. Raises a 503 if the change isn't seen within CLUSTER_COMMAND_TIMEOUT.

        Before, a launch sent as the master died was answered as accepted and never happened.
        After a 503 a launch or a new link is called off: should a copy still apply, this
        node undoes it, so the failure the client was told of holds.
        """
        target = changed_by(command)
        # A request about what a failed one called off takes over from it: its outcome is
        # the one the client acts on. If a late copy of the failed one is being undone, it
        # starts from the state after that.
        withdrawn = self._withdrawn_cluster_commands.pop(target, None)
        if withdrawn is not None and withdrawn.undo_finished is not None:
            await withdrawn.undo_finished.wait()
        undo = undoing_command(command, self.state)
        if await self._send_until_applied(command):
            return
        logger.warning(
            f"Giving up on {type(command).__name__} {command.command_id}: not applied "
            f"after {CLUSTER_COMMAND_TIMEOUT:.0f}s"
        )
        if undo is None:
            raise HTTPException(status_code=503, detail=_NOT_APPLIED_MESSAGE)
        self._withdrawn_cluster_commands[target] = WithdrawnClusterCommand(
            command=command, undo=undo
        )
        raise HTTPException(
            status_code=503,
            detail=_LAUNCH_WITHDRAWN_MESSAGE
            if isinstance(command, CreateInstance)
            else _WITHDRAWN_MESSAGE,
        )

    async def _send_until_applied(self, command: ClusterCommand) -> bool:
        """Send a command that changes the cluster until this node's state shows it applied,
        and say whether it did within CLUSTER_COMMAND_TIMEOUT.

        The master publishes only the events a command leads to, so the effect in the state is
        the only sign it arrived. The command is sent again whenever this node catches up from
        a new master's state, and every CLUSTER_COMMAND_RESEND_INTERVAL otherwise. Applying one
        of these commands twice changes nothing, and one that would change nothing now (a
        repeat) isn't sent at all.
        """
        if command_already_applied(command, self.state):
            return True
        command_id = command.command_id
        pending = PendingClusterCommand(command=command)
        self._pending_cluster_commands[command_id] = pending
        try:
            with anyio.move_on_after(CLUSTER_COMMAND_TIMEOUT):
                while not pending.applied:
                    await self._forward(command)
                    with anyio.move_on_after(CLUSTER_COMMAND_RESEND_INTERVAL):
                        await pending.wake.wait()
                    pending.wake = anyio.Event()
                    if not pending.applied:
                        logger.warning(
                            f"{type(command).__name__} {command_id} not applied yet; "
                            "sending it again"
                        )
            # One last look: the state may have caught up just as time ran out
            return pending.applied or command_applied(command, self.state)
        finally:
            self._pending_cluster_commands.pop(command_id, None)

    def _note_applied_cluster_commands(self) -> None:
        """Mark the pending cluster commands the state now shows applied, and undo the
        withdrawn ones that applied after all. Called after every change to the state, so a
        change undone soon after (an instance created, then deleted as its node is lost)
        still counts as applied."""
        for pending in self._pending_cluster_commands.values():
            if not pending.applied and command_applied(pending.command, self.state):
                pending.applied = True
                pending.wake.set()
        # A called-off change is watched for a copy that still applies for
        # CLUSTER_COMMAND_TIMEOUT after its request was answered: as long as a copy sent before
        # the answer can take to reach a master that applies it, through a change of master. A
        # copy applied later (at the heal of a longer split) isn't undone. Without a limit, a
        # launch the master refused would be checked on every event for the life of the node.
        now = time.monotonic()
        for target, withdrawn in list(self._withdrawn_cluster_commands.items()):
            if withdrawn.undo_finished is not None:
                continue
            if command_applied(withdrawn.command, self.state):
                withdrawn.undo_finished = anyio.Event()
                self._tg.start_soon(self._undo_withdrawn_command, withdrawn)
            elif now - withdrawn.withdrawn_at > CLUSTER_COMMAND_TIMEOUT:
                del self._withdrawn_cluster_commands[target]

    async def _undo_withdrawn_command(self, withdrawn: WithdrawnClusterCommand) -> None:
        command, undo = withdrawn.command, withdrawn.undo
        logger.warning(
            f"{type(command).__name__} {command.command_id} applied after its request "
            f"was answered as failed; undoing it with {type(undo).__name__}"
        )
        finished = withdrawn.undo_finished
        assert finished is not None
        try:
            undone = await self._send_until_applied(undo)
        finally:
            finished.set()
        target = changed_by(command)
        if undone:
            if self._withdrawn_cluster_commands.get(target) is withdrawn:
                del self._withdrawn_cluster_commands[target]
            return
        logger.error(
            f"Couldn't undo {type(command).__name__} {command.command_id}: "
            f"{type(undo).__name__} not applied after {CLUSTER_COMMAND_TIMEOUT:.0f}s; "
            "trying again at the next change to the state"
        )
        withdrawn.undo_finished = None

    def _mark_accepted(self, command_id: CommandId) -> None:
        if (accepted := self._awaiting_acceptance.get(command_id)) is not None:
            accepted.set()

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
        try:
            ModelId(model_id).normalize()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        command = DeleteDownload(
            target_node_id=node_id,
            model_id=ModelId(model_id),
        )
        await self._send_download(command)
        return DeleteDownloadResponse(command_id=command.command_id)

    async def cancel_download(
        self,
        payload: CancelDownloadParams,
    ) -> CancelDownloadResponse:
        command = CancelDownload(
            target_node_id=payload.target_node_id,
            model_id=payload.model_id,
        )
        await self._send_download(command)
        return CancelDownloadResponse(command_id=command.command_id)

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
