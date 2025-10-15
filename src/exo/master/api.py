import asyncio
import os
import time
from collections.abc import AsyncGenerator
from typing import final

import uvicorn
from anyio import create_task_group
from anyio.abc import TaskGroup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from exo.shared.apply import apply
from exo.shared.models.model_cards import MODEL_CARDS
from exo.shared.models.model_meta import get_model_meta
from exo.shared.types.api import (
    ChatCompletionMessage,
    ChatCompletionResponse,
    CreateInstanceResponse,
    CreateInstanceTaskParams,
    DeleteInstanceResponse,
    ModelList,
    ModelListModel,
    StreamingChoiceResponse,
)
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import (
    ChatCompletion,
    Command,
    CreateInstance,
    DeleteInstance,
    ForwarderCommand,
    # TODO: SpinUpInstance
    TaskFinished,
)
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import ChunkGenerated, Event, ForwarderEvent, IndexedEvent
from exo.shared.types.models import ModelMetadata
from exo.shared.types.state import State
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.common import InstanceId
from exo.shared.types.worker.instances import Instance
from exo.utils.channels import Receiver, Sender
from exo.utils.event_buffer import OrderedBuffer


def chunk_to_response(chunk: TokenChunk) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=chunk.command_id,
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


@final
class API:
    def __init__(
        self,
        *,
        node_id: NodeId,
        port: int = 8000,
        # Ideally this would be a MasterForwarderEvent but type system says no :(
        global_event_receiver: Receiver[ForwarderEvent],
        command_sender: Sender[ForwarderCommand],
    ) -> None:
        self.state = State()
        self.command_sender = command_sender
        self.global_event_receiver = global_event_receiver
        self.event_buffer: OrderedBuffer[Event] = OrderedBuffer[Event]()
        self.node_id: NodeId = node_id
        self.port = port

        self.app = FastAPI()
        self._setup_cors()
        self._setup_routes()

        self.app.mount(
            "/",
            StaticFiles(
                directory=os.environ.get(
                    "DASHBOARD_DIR",
                    os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "../../../dashboard")
                    ),
                ),
                html=True,
            ),
            name="dashboard",
        )

        self._chat_completion_queues: dict[
            CommandId, asyncio.Queue[ChunkGenerated]
        ] = {}
        self._tg: TaskGroup | None = None

    def reset(self):
        self.state = State()
        self.event_buffer = OrderedBuffer[Event]()
        self._chat_completion_queues = {}

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        self.app.post("/instance")(self.create_instance)
        self.app.get("/instance/{instance_id}")(self.get_instance)
        self.app.delete("/instance/{instance_id}")(self.delete_instance)
        self.app.get("/models")(self.get_models)
        self.app.get("/v1/models")(self.get_models)
        self.app.post("/v1/chat/completions")(self.chat_completions)
        self.app.get("/state")(lambda: self.state)

    async def create_instance(
        self, payload: CreateInstanceTaskParams
    ) -> CreateInstanceResponse:
        model_meta = await resolve_model_meta(payload.model_id)
        required_memory_bytes = model_meta.storage_size.in_kb
        available_memory_bytes = self._calculate_total_available_memory()

        if required_memory_bytes > available_memory_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient memory to create instance. Required: {required_memory_bytes // (1024**3):.1f}GB, Available: {available_memory_bytes // (1024**3):.1f}GB",
            )

        command = CreateInstance(
            command_id=CommandId(),
            model_meta=model_meta,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_meta=model_meta,
        )

    def get_instance(self, instance_id: InstanceId) -> Instance:
        state = self.state
        if instance_id not in state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        return state.instances[instance_id]

    async def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")

        command = DeleteInstance(
            command_id=CommandId(),
            instance_id=instance_id,
        )
        await self._send(command)
        return DeleteInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance_id,
        )

    async def _generate_chat_stream(
        self, command_id: CommandId
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion stream as JSON strings."""

        self._chat_completion_queues[command_id] = asyncio.Queue()

        finished = False
        while not finished:
            # TODO: how long should this timeout be?
            chunk = await asyncio.wait_for(
                self._chat_completion_queues[command_id].get(), timeout=60
            )
            if chunk.command_id == command_id:
                assert isinstance(chunk.chunk, TokenChunk)
                chunk_response: ChatCompletionResponse = chunk_to_response(chunk.chunk)
                logger.debug(f"chunk_response: {chunk_response}")
                yield f"data: {chunk_response.model_dump_json()}\n\n"

                if chunk.chunk.finish_reason is not None:
                    yield "data: [DONE]\n\n"
                    finished = True

        command = TaskFinished(finished_command_id=command_id)
        await self._send(command)
        del self._chat_completion_queues[command_id]

    async def _trigger_notify_user_to_download_model(self, model_id: str) -> None:
        logger.warning(
            "TODO: we should send a notification to the user to download the model"
        )

    async def chat_completions(
        self, payload: ChatCompletionTaskParams
    ) -> StreamingResponse:
        """Handle chat completions with proper streaming response."""
        model_meta = await resolve_model_meta(payload.model)
        payload.model = model_meta.model_id

        # Preprocess messages for GPT-OSS harmony format if needed
        # TODO: This is slop surely we get rid
        if "gpt-oss" in payload.model.lower():
            import re

            for message in payload.messages:
                if message.content and "<|channel|>" in message.content:
                    # Parse harmony format tags
                    thinking_pattern = r"<\|channel\|>(.*?)(?=<\|message\|>|$)"
                    content_pattern = r"<\|message\|>(.*?)(?=<\|end\|>|$)"

                    thinking_match = re.search(
                        thinking_pattern, message.content, re.DOTALL
                    )
                    content_match = re.search(
                        content_pattern, message.content, re.DOTALL
                    )

                    if content_match:
                        # Extract the actual content
                        message.content = content_match.group(1).strip()
                    if thinking_match:
                        # Store thinking in the thinking field
                        message.thinking = thinking_match.group(1).strip()

        for instance in self.state.instances.values():
            if instance.shard_assignments.model_id == payload.model:
                break
        else:
            await self._trigger_notify_user_to_download_model(payload.model)
            raise HTTPException(
                status_code=404, detail=f"No instance found for model {payload.model}"
            )

        command = ChatCompletion(
            command_id=CommandId(),
            request_params=payload,
        )
        await self._send(command)
        return StreamingResponse(
            self._generate_chat_stream(command.command_id),
            media_type="text/event-stream",
        )

    def _calculate_total_available_memory(self) -> int:
        """Calculate total available memory across all nodes in bytes."""
        total_available = 0

        for node in self.state.topology.list_nodes():
            if node.node_profile is not None:
                total_available += node.node_profile.memory.ram_available.in_bytes

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
                )
                for card in MODEL_CARDS.values()
            ]
        )

    async def run(self):
        uvicorn_config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, access_log=False
        )
        uvicorn_server = uvicorn.Server(uvicorn_config)

        async with create_task_group() as tg:
            self._tg = tg
            logger.info("Starting API")
            tg.start_soon(uvicorn_server.serve)
            tg.start_soon(self._apply_state)
        self.command_sender.close()
        self.global_event_receiver.close()

    async def _apply_state(self):
        with self.global_event_receiver as events:
            async for event in events:
                self.event_buffer.ingest(event.origin_idx, event.event)
                for idx, event in self.event_buffer.drain_indexed():
                    self.state = apply(self.state, IndexedEvent(event=event, idx=idx))
                    if (
                        isinstance(event, ChunkGenerated)
                        and event.command_id in self._chat_completion_queues
                    ):
                        self._chat_completion_queues[event.command_id].put_nowait(event)

    async def _send(self, command: Command):
        await self.command_sender.send(
            ForwarderCommand(origin=self.node_id, command=command)
        )
