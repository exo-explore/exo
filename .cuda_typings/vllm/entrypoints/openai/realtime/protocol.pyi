from typing import Literal
from vllm.entrypoints.openai.engine.protocol import (
    OpenAIBaseModel as OpenAIBaseModel,
    UsageInfo as UsageInfo,
)
from vllm.utils import random_uuid as random_uuid

class InputAudioBufferAppend(OpenAIBaseModel):
    type: Literal["input_audio_buffer.append"]
    audio: str

class InputAudioBufferCommit(OpenAIBaseModel):
    type: Literal["input_audio_buffer.commit"]
    final: bool

class SessionUpdate(OpenAIBaseModel):
    type: Literal["session.update"]
    model: str | None

class SessionCreated(OpenAIBaseModel):
    type: Literal["session.created"]
    id: str
    created: int

class TranscriptionDelta(OpenAIBaseModel):
    type: Literal["transcription.delta"]
    delta: str

class TranscriptionDone(OpenAIBaseModel):
    type: Literal["transcription.done"]
    text: str
    usage: UsageInfo | None

class ErrorEvent(OpenAIBaseModel):
    type: Literal["error"]
    error: str
    code: str | None
