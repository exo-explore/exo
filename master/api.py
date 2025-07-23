import asyncio
import time
from asyncio.queues import Queue
from collections.abc import AsyncGenerator
from typing import List, Optional, Sequence, final

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from shared.db.sqlite.connector import AsyncSQLiteEventStorage
from shared.types.events import ChunkGenerated, Event
from shared.types.events.chunks import TokenChunk
from shared.types.events.components import EventFromEventLog
from shared.types.request import APIRequest, RequestId
from shared.types.tasks import ChatCompletionTaskParams


class Message(BaseModel):
    role: str
    content: str

class StreamingChoiceResponse(BaseModel):
    index: int
    delta: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[StreamingChoiceResponse]

def chunk_to_response(chunk: TokenChunk) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id='abc',
        created=int(time.time()),
        model='idk',
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=Message(
                    role='assistant',
                    content=chunk.text
                ),
                finish_reason=chunk.finish_reason
            )
        ]
    )


@final
class API:
    def __init__(self, command_queue: Queue[APIRequest], global_events: AsyncSQLiteEventStorage) -> None:
        self._app = FastAPI()
        self._setup_routes()

        self.command_queue = command_queue
        self.global_events = global_events

    def _setup_routes(self) -> None:
        # self._app.get("/topology/control_plane")(self.get_control_plane_topology)
        # self._app.get("/topology/data_plane")(self.get_data_plane_topology)
        # self._app.get("/instances/list")(self.list_instances)
        # self._app.post("/instances/create")(self.create_instance)
        # self._app.get("/instance/{instance_id}/read")(self.get_instance)
        # self._app.delete("/instance/{instance_id}/delete")(self.remove_instance)
        # self._app.get("/model/{model_id}/metadata")(self.get_model_data)
        # self._app.post("/model/{model_id}/instances")(self.get_instances_by_model)
        self._app.post("/v1/chat/completions")(self.chat_completions)

    @property
    def app(self) -> FastAPI:
        return self._app

    # def get_control_plane_topology(self):
    #     return {"message": "Hello, World!"}

    # def get_data_plane_topology(self):
    #     return {"message": "Hello, World!"}

    # def get_model_metadata(self, model_id: ModelId) -> ModelMetadata: ...

    # def download_model(self, model_id: ModelId) -> None: ...

    # def list_instances(self):
    #     return {"message": "Hello, World!"}

    # def create_instance(self, model_id: ModelId) -> InstanceId: ...

    # def get_instance(self, instance_id: InstanceId) -> Instance: ...

    # def remove_instance(self, instance_id: InstanceId) -> None: ...

    # def get_model_data(self, model_id: ModelId) -> ModelInfo: ...

    # def get_instances_by_model(self, model_id: ModelId) -> list[Instance]: ...

    async def _generate_chat_stream(self, payload: ChatCompletionTaskParams) -> AsyncGenerator[str, None]:
        """Generate chat completion stream as JSON strings."""
        events = await self.global_events.get_events_since(0)
        prev_idx = await self.global_events.get_last_idx()

        # At the moment, we just create the task in the API.
        # In the future, a `Request` will be created here and they will be bundled into `Task` objects by the master.
        request_id=RequestId()

        request = APIRequest(
            request_id=request_id,
            request_params=payload,
        )
        await self.command_queue.put(request)

        finished = False
        while not finished:
            await asyncio.sleep(0.01)

            events: Sequence[EventFromEventLog[Event]] = await self.global_events.get_events_since(prev_idx)
            # TODO: Can do this with some better functionality to tail event log into an AsyncGenerator.
            prev_idx = events[-1].idx_in_log if events else prev_idx

            for wrapped_event in events:
                event = wrapped_event.event
                if isinstance(event, ChunkGenerated) and event.request_id == request_id:
                    assert isinstance(event.chunk, TokenChunk)
                    chunk_response: ChatCompletionResponse = chunk_to_response(event.chunk)
                    print(chunk_response)
                    yield f"data: {chunk_response.model_dump_json()}\n\n"

                    if event.chunk.finish_reason is not None:
                        yield "data: [DONE]"
                        finished = True

        return

    async def chat_completions(self, payload: ChatCompletionTaskParams) -> StreamingResponse:
        """Handle chat completions with proper streaming response."""
        return StreamingResponse(
            self._generate_chat_stream(payload),
            media_type="text/plain"
        )



def start_fastapi_server(
    command_queue: Queue[APIRequest],
    global_events: AsyncSQLiteEventStorage,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    api = API(command_queue, global_events)

    uvicorn.run(api.app, host=host, port=port)