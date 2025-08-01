import asyncio
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Callable, List, Sequence, final

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from shared.db.sqlite.connector import AsyncSQLiteEventStorage
from shared.models.model_cards import MODEL_CARDS
from shared.models.model_meta import get_model_meta
from shared.types.api import (
    ChatCompletionMessage,
    ChatCompletionResponse,
    CreateInstanceResponse,
    CreateInstanceTaskParams,
    DeleteInstanceResponse,
    ModelList,
    ModelListModel,
    StreamingChoiceResponse,
)
from shared.types.common import CommandId
from shared.types.events import ChunkGenerated, Event
from shared.types.events.chunks import TokenChunk
from shared.types.events.commands import (
    ChatCompletionCommand,
    Command,
    CommandType,
    CreateInstanceCommand,
    DeleteInstanceCommand,
)
from shared.types.events.components import EventFromEventLog
from shared.types.models import ModelMetadata
from shared.types.state import State
from shared.types.tasks import ChatCompletionTaskParams
from shared.types.worker.common import InstanceId
from shared.types.worker.instances import Instance

# TODO: Make sure that when we package the app the dashboard is in the right place.
_ROOT_DIR = Path(__file__).resolve().parents[1]
_DASHBOARD_DIR = _ROOT_DIR / "dashboard"


def chunk_to_response(chunk: TokenChunk) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=chunk.command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=ChatCompletionMessage(
                    role='assistant',
                    content=chunk.text
                ),
                finish_reason=chunk.finish_reason
            )
        ]
    )

async def resolve_model_meta(model_id: str) -> ModelMetadata:
    if model_id in MODEL_CARDS:
        model_card = MODEL_CARDS[model_id]
        return model_card.metadata
    else:
        return await get_model_meta(model_id)

@final
class API:
    def __init__(self, command_buffer: List[Command], global_events: AsyncSQLiteEventStorage, get_state: Callable[[], State]) -> None:
        self.get_state = get_state
        self.command_buffer = command_buffer
        self.global_events = global_events

        self._app = FastAPI()
        self._setup_cors()
        self._setup_routes()

        self._app.mount("/", StaticFiles(directory=_DASHBOARD_DIR, html=True), name="dashboard")

    def _setup_cors(self) -> None:
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        self._app.post("/instance")(self.create_instance)
        self._app.get("/instance/{instance_id}")(self.get_instance)
        self._app.delete("/instance/{instance_id}")(self.delete_instance)
        self._app.get("/models")(self.get_models)
        self._app.get("/v1/models")(self.get_models)
        self._app.post("/v1/chat/completions")(self.chat_completions)
        self._app.get("/state")(self.get_state)

    @property
    def app(self) -> FastAPI:
        return self._app

    async def create_instance(self, payload: CreateInstanceTaskParams) -> CreateInstanceResponse:
        model_meta = await resolve_model_meta(payload.model_id)
        required_memory_bytes = model_meta.storage_size_kilobytes * 1024
        available_memory_bytes = self._calculate_total_available_memory()
        
        if required_memory_bytes > available_memory_bytes:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient memory to create instance. Required: {required_memory_bytes // (1024**3):.1f}GB, Available: {available_memory_bytes // (1024**3):.1f}GB"
            )

        command = CreateInstanceCommand(
            command_id=CommandId(),
            command_type=CommandType.CREATE_INSTANCE,
            model_meta=model_meta,
            instance_id=InstanceId(),
        )
        self.command_buffer.append(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_meta=model_meta,
            instance_id=command.instance_id,
        )

    def get_instance(self, instance_id: InstanceId) -> Instance:
        state = self.get_state()
        if instance_id not in state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        return state.instances[instance_id]

    def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        if instance_id not in self.get_state().instances:
            raise HTTPException(status_code=404, detail="Instance not found")

        command = DeleteInstanceCommand(
            command_id=CommandId(),
            command_type=CommandType.DELETE_INSTANCE,
            instance_id=instance_id,
        )
        self.command_buffer.append(command)
        return DeleteInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance_id,
        )

    async def _generate_chat_stream(self, command_id: CommandId) -> AsyncGenerator[str, None]:
        """Generate chat completion stream as JSON strings."""

        events = await self.global_events.get_events_since(0)
        prev_idx = await self.global_events.get_last_idx()

        finished = False
        while not finished:
            await asyncio.sleep(0.01)

            events: Sequence[EventFromEventLog[Event]] = await self.global_events.get_events_since(prev_idx)
            # TODO: Can do this with some better functionality to tail event log into an AsyncGenerator.
            prev_idx = events[-1].idx_in_log if events else prev_idx

            for wrapped_event in events:
                event = wrapped_event.event
                if isinstance(event, ChunkGenerated) and event.command_id == command_id:
                    assert isinstance(event.chunk, TokenChunk)
                    chunk_response: ChatCompletionResponse = chunk_to_response(event.chunk)
                    print(chunk_response)
                    yield f"data: {chunk_response.model_dump_json()}\n\n"

                    if event.chunk.finish_reason is not None:
                        yield "data: [DONE]"
                        finished = True

        return

    async def _trigger_notify_user_to_download_model(self, model_id: str) -> None:
        print("TODO: we should send a notification to the user to download the model")

    async def chat_completions(self, payload: ChatCompletionTaskParams) -> StreamingResponse:
        """Handle chat completions with proper streaming response."""
        model_meta = await resolve_model_meta(payload.model)
        payload.model = model_meta.model_id

        for instance in self.get_state().instances.values():
            if instance.shard_assignments.model_id == payload.model:
                break
        else:
            await self._trigger_notify_user_to_download_model(payload.model)
            raise HTTPException(status_code=404, detail=f"No instance found for model {payload.model}")

        command = ChatCompletionCommand(
            command_id=CommandId(),
            command_type=CommandType.CHAT_COMPLETION,
            request_params=payload,
        )
        self.command_buffer.append(command)
        return StreamingResponse(
            self._generate_chat_stream(command.command_id),
            media_type="text/plain"
        )

    def _calculate_total_available_memory(self) -> int:
        """Calculate total available memory across all nodes in bytes."""
        state = self.get_state()
        total_available = 0
        
        for node_profile in state.node_profiles.values():
            total_available += node_profile.memory.ram_available
            
        return total_available

    async def get_models(self) -> ModelList:
        """Returns list of available models."""
        return ModelList(data=[
            ModelListModel(
                id=card.short_id,
                hugging_face_id=card.model_id,
                name=card.name,
                description=card.description,
                tags=card.tags) for card in MODEL_CARDS.values()])


def start_fastapi_server(
    command_buffer: List[Command],
    global_events: AsyncSQLiteEventStorage,
    get_state: Callable[[], State],
    host: str = "0.0.0.0",
    port: int = 8000,
):
    api = API(command_buffer, global_events, get_state)

    uvicorn.run(api.app, host=host, port=port)
