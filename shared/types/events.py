from __future__ import annotations

from typing import Annotated, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, TypeAdapter, UuidVersion

from shared.types.event_sourcing import Event

_ModelId = Annotated[UUID, UuidVersion(4)]
ModelId = type("ModelId", (UUID,), {})
ModelIdParser: TypeAdapter[ModelId] = TypeAdapter(_ModelId)

_NodeId = Annotated[UUID, UuidVersion(4)]
NodeId = type("NodeId", (UUID,), {})
NodeIdParser: TypeAdapter[NodeId] = TypeAdapter(_NodeId)

_RequestId = Annotated[UUID, UuidVersion(4)]
RequestId = type("RequestId", (UUID,), {})
RequestIdParser: TypeAdapter[RequestId] = TypeAdapter(_RequestId)

_InstanceId = Annotated[UUID, UuidVersion(4)]
InstanceId = type("InstanceId", (UUID,), {})
InstanceIdParser: TypeAdapter[InstanceId] = TypeAdapter(_InstanceId)

_TimerId = Annotated[UUID, UuidVersion(4)]
TimerId = type("TimerId", (UUID,), {})
TimerIdParser: TypeAdapter[TimerId] = TypeAdapter(_TimerId)

class ModelMetadata(BaseModel):
    model_id: ModelId
    repo_id: str
    model_size: int

class ChatRequest(BaseModel):
    # TODO: from OpenAI
    prompt: str

class Shard(BaseModel):
    # TODO: this has changed
    model_id: ModelId

class InstanceComputePlan(BaseModel):
    # TODO: this has changed
    model_id: ModelId

class Timer(BaseModel):
    timer_id: TimerId

# Chat completions ----------------------------------------------------------------
class ChatCompletionsRequestStarted(Event[Literal["ChatCompletionsRequestStarted"]]):
    event_type = "ChatCompletionsRequestStarted"
    request_id: RequestId
    user_id: str
    model_id: ModelId
    request: ChatRequest


class ChatCompletionsRequestCompleted(Event[Literal["ChatCompletionsRequestCompleted"]]):
    event_type = "ChatCompletionsRequestCompleted"
    request_id: RequestId
    user_id: str
    model_id: str
    request: ChatRequest
    result: str


class ChatCompletionsRequestFailed(Event[Literal["ChatCompletionsRequestFailed"]]):
    event_type = "ChatCompletionsRequestFailed"
    request_id: RequestId
    user_id: str
    model_id: str
    request: ChatRequest
    reason: str


# Inference saga ------------------------------------------------------------------
class InferenceSagaStarted(Event[Literal["InferenceSagaStarted"]]):
    event_type = "InferenceSagaStarted"
    request_id: RequestId
    instance_id: InstanceId
    user_id: str
    model_id: str
    request: ChatRequest


class InferencePrepareStarted(Event[Literal["InferencePrepareStarted"]]):
    event_type = "InferencePrepareStarted"
    request_id: RequestId
    instance_id: InstanceId
    target_node_id: NodeId
    hosts: List[str]
    user_id: str
    shard: Shard  # replaces model_id, rank, start_layer, end_layer
    request: ChatRequest


class InferencePrepareCompleted(Event[Literal["InferencePrepareCompleted"]]):
    event_type = "InferencePrepareCompleted"
    request_id: RequestId
    instance_id: InstanceId
    target_node_id: NodeId
    hosts: List[str]
    user_id: str
    shard: Shard
    request: ChatRequest


class InferenceTriggerStarted(Event[Literal["InferenceTriggerStarted"]]):
    event_type = "InferenceTriggerStarted"
    request_id: RequestId
    instance_id: InstanceId
    target_node_id: NodeId
    hosts: List[str]
    user_id: str
    shard: Shard
    request: ChatRequest


class InferenceTriggerCompleted(Event[Literal["InferenceTriggerCompleted"]]):
    event_type = "InferenceTriggerCompleted"
    request_id: RequestId
    instance_id: InstanceId
    target_node_id: NodeId
    hosts: List[str]
    user_id: str
    shard: Shard
    request: ChatRequest


class InferenceCompleted(Event[Literal["InferenceCompleted"]]):
    event_type = "InferenceCompleted"
    request_id: RequestId
    instance_id: InstanceId
    user_id: str
    model_id: str
    request: ChatRequest
    result: str


class InferenceSagaCompleted(Event[Literal["InferenceSagaCompleted"]]):
    event_type = "InferenceSagaCompleted"
    request_id: RequestId
    instance_id: InstanceId
    user_id: str
    model_id: str
    request: ChatRequest
    result: str


# Instance setup saga ------------------------------------------------------------
class InstanceSetupSagaStarted(Event[Literal["InstanceSetupSagaStarted"]]):
    event_type = "InstanceSetupSagaStarted"
    instance_id: str
    model_id: ModelId
    plan: InstanceComputePlan


class InstanceSetupSagaCompleted(Event[Literal["InstanceSetupSagaCompleted"]]):
    event_type = "InstanceSetupSagaCompleted"
    instance_id: InstanceId
    model_id: ModelId


class InstanceSetupSagaFailed(Event[Literal["InstanceSetupSagaFailed"]]):
    event_type = "InstanceSetupSagaFailed"
    instance_id: InstanceId
    model_id: ModelId
    reason: str


# Shard lifecycle -----------------------------------------------------------------
class ShardAssigned(Event[Literal["ShardAssigned"]]):
    event_type = "ShardAssigned"
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]


class ShardAssignFailed(Event[Literal["ShardAssignFailed"]]):
    event_type = "ShardAssignFailed"
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]
    reason: str  # e.g. "not enough memory"


class ShardUnassigned(Event[Literal["ShardUnassigned"]]):
    event_type = "ShardUnassigned"
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]
    reason: str  # e.g. "instance did not receive request for 5 mins"


class ShardUnassignFailed(Event[Literal["ShardUnassignFailed"]]):
    event_type = "ShardUnassignFailed"
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]
    reason: str  # e.g. "process refused to quit"


class ShardKilled(Event[Literal["ShardKilled"]]):
    event_type = "ShardKilled"
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]


class ShardDied(Event[Literal["ShardDied"]]):
    event_type = "ShardDied"
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]
    error_type: str
    error_message: str
    traceback: Optional[str] = None


class ShardSpawned(Event[Literal["ShardSpawned"]]):
    event_type = "ShardSpawned"
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]


class ShardSpawnedFailed(Event[Literal["ShardSpawnedFailed"]]):
    event_type = "ShardSpawnedFailed"
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]
    reason: str  # e.g. "not enough memory"


class ShardDespawned(Event[Literal["ShardDespawned"]]):
    event_type = "ShardDespawned"
    instance_id: InstanceId
    shard: Shard
    target_node_id: NodeId
    hosts: List[str]


# Node connectivity --------------------------------------------------------------
class NodeConnected(Event[Literal["NodeConnected"]]):
    event_type = "NodeConnected"
    remote_node_id: NodeId
    connection_id: str
    multiaddr: str
    remote_multiaddr: str
    ip: str
    remote_ip: str


class NodeConnectionProfiled(Event[Literal["NodeConnectionProfiled"]]):
    event_type = "NodeConnectionProfiled"
    remote_node_id: NodeId
    connection_id: str
    latency_ms: int
    bandwidth_bytes_per_second: int


class NodeDisconnected(Event[Literal["NodeDisconnected"]]):
    event_type = "NodeDisconnected"
    remote_node_id: NodeId
    connection_id: str


class NodeStarted(Event[Literal["NodeStarted"]]):
    event_type = "NodeStarted"


# Device metrics -----------------------------------------------------------------
class DeviceRegistered(Event[Literal["DeviceRegistered"]]):
    event_type = "DeviceRegistered"
    device_id: str
    device_model: str
    device_type: str
    total_memory_bytes: int
    available_memory_bytes: int


class DeviceProfiled(Event[Literal["DeviceProfiled"]]):
    event_type = "DeviceProfiled"
    device_id: str
    total_memory_bytes: int
    available_memory_bytes: int
    total_flops_fp16: int

# Token streaming ----------------------------------------------------------------
class TokenGenerated(Event[Literal["TokenGenerated"]]):
    event_type = "TokenGenerated"
    request_id: RequestId
    instance_id: InstanceId
    model_id: str
    hosts: List[str]
    token: int
    text: str
    finish_reason: Optional[str] = None


# Repo download progress ----------------------------------------------------------
class RepoProgressEvent(Event[Literal["RepoProgressEvent"]]):
    event_type = "RepoProgressEvent"
    repo_id: str
    downloaded_bytes: int
    total_bytes: int
    speed_bytes_per_second: int


# Timers -------------------------------------------------------------------------
class TimerScheduled(Event[Literal["TimerScheduled"]]):
    event_type = "TimerScheduled"
    timer: Timer


class TimerFired(Event[Literal["TimerFired"]]):
    event_type = "TimerFired"
    timer: Timer
