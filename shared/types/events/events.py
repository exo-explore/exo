from __future__ import annotations

from typing import Annotated, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, TypeAdapter, UuidVersion

from shared.openai import FinishReason, chat
from shared.types.common import NodeId
from shared.types.events.common import Event, EventTypes
from shared.types.models.common import ModelId
from shared.types.worker.common import InstanceId

_RequestId = Annotated[UUID, UuidVersion(4)]
RequestId = type("RequestId", (UUID,), {})
RequestIdParser: TypeAdapter[RequestId] = TypeAdapter(_RequestId)

_TimerId = Annotated[UUID, UuidVersion(4)]
TimerId = type("TimerId", (UUID,), {})
TimerIdParser: TypeAdapter[TimerId] = TypeAdapter(_TimerId)


class Shard(BaseModel):
    # TODO: this has changed
    model_id: ModelId


class InstanceComputePlan(BaseModel):
    # TODO: this has changed
    model_id: ModelId


class Timer(BaseModel):
    timer_id: TimerId


# Chat completions ----------------------------------------------------------------
class ChatCompletionsRequestStarted(Event[EventTypes.ChatCompletionsRequestStarted]):
    event_type: Literal[EventTypes.ChatCompletionsRequestStarted] = (
        EventTypes.ChatCompletionsRequestStarted
    )
    request_id: RequestId
    model_id: ModelId
    request: chat.completion_create_params.CompletionCreateParams


class ChatCompletionsRequestCompleted(
    Event[EventTypes.ChatCompletionsRequestCompleted]
):
    event_type: Literal[EventTypes.ChatCompletionsRequestCompleted] = (
        EventTypes.ChatCompletionsRequestCompleted
    )
    request_id: RequestId
    model_id: ModelId


class ChatCompletionsRequestFailed(Event[EventTypes.ChatCompletionsRequestFailed]):
    event_type: Literal[EventTypes.ChatCompletionsRequestFailed] = (
        EventTypes.ChatCompletionsRequestFailed
    )
    request_id: RequestId
    model_id: ModelId
    error_message: str


# Inference saga ------------------------------------------------------------------
class InferenceSagaStarted(Event[EventTypes.InferenceSagaStarted]):
    event_type: Literal[EventTypes.InferenceSagaStarted] = (
        EventTypes.InferenceSagaStarted
    )
    request_id: RequestId
    instance_id: InstanceId
    model_id: ModelId
    request: chat.completion_create_params.CompletionCreateParams


class InferencePrepareStarted(Event[EventTypes.InferencePrepareStarted]):
    event_type: Literal[EventTypes.InferencePrepareStarted] = (
        EventTypes.InferencePrepareStarted
    )
    request_id: RequestId
    instance_id: InstanceId
    target_node_id: NodeId
    hosts: List[str]
    shard: Shard  # replaces model_id, rank, start_layer, end_layer
    request: chat.completion_create_params.CompletionCreateParams


class InferencePrepareCompleted(Event[EventTypes.InferencePrepareCompleted]):
    event_type: Literal[EventTypes.InferencePrepareCompleted] = (
        EventTypes.InferencePrepareCompleted
    )
    request_id: RequestId
    instance_id: InstanceId
    target_node_id: NodeId
    hosts: List[str]
    shard: Shard


class InferenceTriggerStarted(Event[EventTypes.InferenceTriggerStarted]):
    event_type: Literal[EventTypes.InferenceTriggerStarted] = (
        EventTypes.InferenceTriggerStarted
    )
    request_id: RequestId
    instance_id: InstanceId
    target_node_id: NodeId
    hosts: List[str]
    shard: Shard
    request: chat.completion_create_params.CompletionCreateParams


class InferenceTriggerCompleted(Event[EventTypes.InferenceTriggerCompleted]):
    event_type: Literal[EventTypes.InferenceTriggerCompleted] = (
        EventTypes.InferenceTriggerCompleted
    )
    request_id: RequestId
    instance_id: InstanceId
    target_node_id: NodeId
    hosts: List[str]
    shard: Shard


class InferenceCompleted(Event[EventTypes.InferenceCompleted]):
    event_type: Literal[EventTypes.InferenceCompleted] = EventTypes.InferenceCompleted
    request_id: RequestId
    instance_id: InstanceId
    model_id: ModelId


class InferenceSagaCompleted(Event[EventTypes.InferenceSagaCompleted]):
    event_type: Literal[EventTypes.InferenceSagaCompleted] = (
        EventTypes.InferenceSagaCompleted
    )
    request_id: RequestId
    instance_id: InstanceId
    model_id: ModelId


# Instance setup saga ------------------------------------------------------------
class InstanceSetupSagaStarted(Event[EventTypes.InstanceSetupSagaStarted]):
    event_type: Literal[EventTypes.InstanceSetupSagaStarted] = (
        EventTypes.InstanceSetupSagaStarted
    )
    instance_id: str
    model_id: ModelId
    plan: InstanceComputePlan


class InstanceSetupSagaCompleted(Event[EventTypes.InstanceSetupSagaCompleted]):
    event_type: Literal[EventTypes.InstanceSetupSagaCompleted] = (
        EventTypes.InstanceSetupSagaCompleted
    )
    instance_id: InstanceId
    model_id: ModelId


class InstanceSetupSagaFailed(Event[EventTypes.InstanceSetupSagaFailed]):
    event_type: Literal[EventTypes.InstanceSetupSagaFailed] = (
        EventTypes.InstanceSetupSagaFailed
    )
    instance_id: InstanceId
    model_id: ModelId
    reason: str


# Shard lifecycle -----------------------------------------------------------------
class ShardAssigned(Event[EventTypes.ShardAssigned]):
    event_type: Literal[EventTypes.ShardAssigned] = EventTypes.ShardAssigned
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]


class ShardAssignFailed(Event[EventTypes.ShardAssignFailed]):
    event_type: Literal[EventTypes.ShardAssignFailed] = EventTypes.ShardAssignFailed
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]
    reason: str  # e.g. "not enough memory"


class ShardUnassigned(Event[EventTypes.ShardUnassigned]):
    event_type: Literal[EventTypes.ShardUnassigned] = EventTypes.ShardUnassigned
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]
    reason: str  # e.g. "instance did not receive request for 5 mins"


class ShardUnassignFailed(Event[EventTypes.ShardUnassignFailed]):
    event_type: Literal[EventTypes.ShardUnassignFailed] = EventTypes.ShardUnassignFailed
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]
    reason: str  # e.g. "process refused to quit"


class ShardKilled(Event[EventTypes.ShardKilled]):
    event_type: Literal[EventTypes.ShardKilled] = EventTypes.ShardKilled
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]


class ShardDied(Event[EventTypes.ShardDied]):
    event_type: Literal[EventTypes.ShardDied] = EventTypes.ShardDied
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]
    error_type: str
    error_message: str
    traceback: Optional[str] = None


class ShardSpawned(Event[EventTypes.ShardSpawned]):
    event_type: Literal[EventTypes.ShardSpawned] = EventTypes.ShardSpawned
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]


class ShardSpawnedFailed(Event[EventTypes.ShardSpawnedFailed]):
    event_type: Literal[EventTypes.ShardSpawnedFailed] = EventTypes.ShardSpawnedFailed
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]
    reason: str  # e.g. "not enough memory"


class ShardDespawned(Event[EventTypes.ShardDespawned]):
    event_type: Literal[EventTypes.ShardDespawned] = EventTypes.ShardDespawned
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]


# Node connectivity --------------------------------------------------------------
class NodeConnected(Event[EventTypes.NodeConnected]):
    event_type: Literal[EventTypes.NodeConnected] = EventTypes.NodeConnected
    remote_node_id: NodeId
    connection_id: str
    multiaddr: str
    remote_multiaddr: str
    ip: str
    remote_ip: str


class NodeConnectionProfiled(Event[EventTypes.NodeConnectionProfiled]):
    event_type: Literal[EventTypes.NodeConnectionProfiled] = (
        EventTypes.NodeConnectionProfiled
    )
    remote_node_id: NodeId
    connection_id: str
    latency_ms: int
    bandwidth_bytes_per_second: int


class NodeDisconnected(Event[EventTypes.NodeDisconnected]):
    event_type: Literal[EventTypes.NodeDisconnected] = EventTypes.NodeDisconnected
    remote_node_id: NodeId
    connection_id: str


class NodeStarted(Event[EventTypes.NodeStarted]):
    event_type: Literal[EventTypes.NodeStarted] = EventTypes.NodeStarted


# Device metrics -----------------------------------------------------------------
class DeviceRegistered(Event[EventTypes.DeviceRegistered]):
    event_type: Literal[EventTypes.DeviceRegistered] = EventTypes.DeviceRegistered
    device_id: str
    device_model: str
    device_type: str
    total_memory_bytes: int
    available_memory_bytes: int


class DeviceProfiled(Event[EventTypes.DeviceProfiled]):
    event_type: Literal[EventTypes.DeviceProfiled] = EventTypes.DeviceProfiled
    device_id: str
    total_memory_bytes: int
    available_memory_bytes: int
    total_flops_fp16: int


# Token streaming ----------------------------------------------------------------
class TokenGenerated(Event[EventTypes.TokenGenerated]):
    # TODO: replace with matt chunk code
    event_type: Literal[EventTypes.TokenGenerated] = EventTypes.TokenGenerated
    request_id: RequestId
    instance_id: InstanceId
    hosts: List[str]
    token: int
    text: str
    finish_reason: FinishReason


# Repo download progress ----------------------------------------------------------
class RepoProgressEvent(Event[EventTypes.RepoProgressEvent]):
    event_type: Literal[EventTypes.RepoProgressEvent] = EventTypes.RepoProgressEvent
    repo_id: str
    downloaded_bytes: int
    total_bytes: int
    speed_bytes_per_second: int


# Timers -------------------------------------------------------------------------
class TimerScheduled(Event[EventTypes.TimerScheduled]):
    event_type: Literal[EventTypes.TimerScheduled] = EventTypes.TimerScheduled
    timer: Timer


class TimerFired(Event[EventTypes.TimerFired]):
    event_type: Literal[EventTypes.TimerFired] = EventTypes.TimerFired
    timer: Timer
