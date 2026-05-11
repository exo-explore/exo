import time
from collections.abc import Generator
from typing import Annotated, Any, Final, Literal, get_args
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import ReasoningDialect, ReasoningEffort
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding, ShardMetadata
from exo.utils.pydantic_ext import FrozenModel

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call", "error"
]

# Upper bound for the per-request ``num_draft_tokens`` override. The runner
# allocates a fixed wire-protocol budget at warmup (``EXO_NUM_DRAFT_TOKENS``,
# default in ``defaults.py``), and per-request K is clamped to that budget
# inside ``generate.py``. The API-level cap exists only as a sanity guard
# against obviously-pathological inputs (negative values are blocked by
# ``ge=1``; values like ``10**9`` would still crash the runner subprocess
# via an unhandled ``ValueError`` if they escaped the API boundary).
#
# Codex flagged (PR #20 round 2 P2) that an earlier ``= 32`` cap was a
# regression for benchmarking and tuning flows: those flows sweep larger K
# values when the operator has explicitly raised ``EXO_NUM_DRAFT_TOKENS``
# (e.g. K=64 on a fat target / drafter pair) and previously the runner
# would handle them. The cap is intentionally raised to a value far above
# any realistic budget so it never gates legitimate sweeps; the runner's
# internal clamp in ``generate.py`` against ``EXO_NUM_DRAFT_TOKENS``
# remains the authoritative bound.
MAX_NUM_DRAFT_TOKENS_PER_REQUEST: Final[int] = 1024


class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponse(BaseModel):
    error: ErrorInfo


class ModelListModel(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "exo"
    # openwebui fields
    hugging_face_id: str = Field(default="")
    name: str = Field(default="")
    description: str = Field(default="")
    context_length: int = Field(default=0)
    tags: list[str] = Field(default=[])
    storage_size_megabytes: int = Field(default=0)
    supports_tensor: bool = Field(default=False)
    tasks: list[str] = Field(default=[])
    is_custom: bool = Field(default=False)
    family: str = Field(default="")
    quantization: str = Field(default="")
    base_model: str = Field(default="")
    capabilities: list[str] = Field(default_factory=list)
    reasoning_dialect: ReasoningDialect = "none"
    # Smaller draft models the runner can load alongside this target for
    # speculative decoding. Listed in preference order (`fastest` first).
    # Surfaced so dashboards and clients can pre-download a drafter and
    # pick which one to use at request time.
    drafter_model_ids: list[str] = Field(default_factory=list)


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelListModel]


class AgentEndpoint(BaseModel):
    name: str
    kind: Literal["default", "model", "instance"]
    openai_base_url: str
    claude_base_url: str | None
    model_id: ModelId | None
    target_instance_id: InstanceId | None
    active: bool
    description: str


class AgentEndpointList(BaseModel):
    object: Literal["list"] = "list"
    data: list[AgentEndpoint]


class ChatCompletionMessageText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatCompletionMessageImageUrl(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: dict[str, str]  # {"url": "data:image/png;base64,..."}


ChatCompletionContentPart = ChatCompletionMessageText | ChatCompletionMessageImageUrl


class ToolCallItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    index: int | None = None
    type: Literal["function"] = "function"
    function: ToolCallItem


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: (
        str | ChatCompletionContentPart | list[ChatCompletionContentPart] | None
    ) = None
    reasoning_content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class BenchChatCompletionMessage(ChatCompletionMessage):
    pass


class TopLogprobItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobsContentItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprobItem]


class Logprobs(BaseModel):
    content: list[LogprobsContentItem] | None = None


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails
    completion_tokens_details: CompletionTokensDetails


class StreamingChoiceResponse(BaseModel):
    index: int
    delta: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None


class GenerationStats(BaseModel):
    prompt_tps: float
    generation_tps: float
    prompt_tokens: int
    generation_tokens: int
    peak_memory_usage: Memory
    prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
    # Speculative-decoding telemetry. ``drafter_model_id`` is set whenever
    # speculative decoding actually ran for this request (drafter loaded *and*
    # not short-circuited by the short-skip threshold). ``accepted_draft_tokens``
    # counts ``stream_generate`` outputs with ``from_draft=True``: those are
    # tokens the drafter proposed *and* the target accepted. The user-facing
    # speedup is approximately ``accepted_draft_tokens / generation_tokens``.
    drafter_model_id: str | None = None
    accepted_draft_tokens: int = 0
    # Total drafts the drafter proposed across all spec-decode rounds.
    # ``0`` means either the drafter didn't run or the drafter implementation
    # doesn't surface proposal counts (currently only the pipelined drafter
    # does). The classical per-position acceptance rate is
    # ``accepted_draft_tokens / proposed_draft_tokens``; ``0`` here makes
    # that property return ``None`` rather than divide-by-zero. ``mlx_lm``'s
    # built-in ``stream_generate(draft_model=...)`` does not expose proposal
    # counts at all, so external-model-drafter requests will leave this at 0
    # while still populating ``accepted_draft_tokens``.
    proposed_draft_tokens: int = 0
    # Number of speculative-decoding rounds that actually ran. Each round
    # proposes ``num_draft_tokens`` drafts (truncated near max_tokens).
    # Useful for computing per-round latency in dashboards. ``0`` when the
    # drafter didn't run or doesn't surface round counts.
    spec_decode_rounds: int = 0
    # K used for speculative_generate_step (None when drafter didn't run).
    num_draft_tokens: int | None = None
    # Drafting strategy that actually ran for this request: "model" for
    # external-drafter spec decoding, "pipelined" for the pipelined+
    # remote drafter, "ngram" for in-context suffix lookup, "eagle" /
    # "lookahead" reserved for the upcoming auxiliary-head + Jacobi
    # drafters, "none" for non-speculative. None when the engine doesn't
    # surface drafting (e.g. image gen). Useful for telemetry dashboards
    # to attribute throughput wins to a specific strategy when running
    # mixed-mode A/B tests.
    draft_mode: (
        Literal["model", "pipelined", "ngram", "eagle", "lookahead", "none"] | None
    ) = None
    # Drafter architecture, when speculative decoding actually ran:
    # ``"standard"`` -- external sibling LM via ``mlx_lm.stream_generate``
    #   (the historical model-drafter / pipelined paths).
    # ``"mtp"`` -- Multi-Token-Prediction coupled drafter (gemma4_assistant)
    #   that consumes the target's last-layer hidden + per-layer-type shared
    #   KV every round.
    # ``"dflash"`` -- DFlash coupled drafter (qwen3_dflash) -- consumes a
    #   concatenated multi-layer hidden tensor, no shared KV.
    # ``None`` when ``draft_mode == "none"`` or the engine doesn't expose
    # drafter telemetry. Surfaced separately from ``draft_mode`` so dashboards
    # can disambiguate coupled vs. standard runs without re-shaping the
    # ``DraftMode`` literal: the on-the-wire ``draft_mode`` for coupled runs
    # remains ``"model"`` (the user-visible request mode) while ``drafter_kind``
    # carries the architecture. ``"ngram"`` and ``"none"`` runs leave this
    # ``None`` since there's no model-architecture distinction to surface.
    drafter_kind: Literal["standard", "mtp", "dflash"] | None = None

    @property
    def drafter_acceptance_fraction(self) -> float | None:
        """Fraction of *generated* tokens that came from the drafter.

        ``None`` when no drafter ran for the request. This is a slight
        misnomer relative to the speculative-decoding literature -- the true
        acceptance rate would divide by the drafter's proposal count, which
        ``stream_generate`` doesn't surface -- but it is the metric that
        directly maps to wall-clock speedup, so it's what we display.
        :attr:`drafter_acceptance_rate` exposes the classical metric for
        the pipelined drafter (which tracks proposal counts).

        Codex P2 (PR #19 round-(N+1)): n-gram speculation
        (``draft_mode="ngram"``) intentionally runs without a drafter
        model id because it's an in-process suffix-lookup over the
        prompt + partial generation rather than a separate model.
        Pre-fix this property returned ``None`` for every n-gram run
        (because ``drafter_model_id is None``), which misreported
        valid speculative runs as non-speculative in telemetry and
        broke acceptance metrics for n-gram A/B tests. Trust
        ``draft_mode`` as the canonical "did a drafter run?" signal:
        accept any non-``"none"`` mode, and fall back to the legacy
        ``drafter_model_id`` heuristic for streams that don't yet
        carry ``draft_mode`` (older recorded benches, partial
        responses).
        """
        if self.generation_tokens == 0:
            return None
        if self.draft_mode is None:
            # Older payload: only model-mode telemetry was
            # recorded historically.
            if self.drafter_model_id is None:
                return None
        elif self.draft_mode == "none":
            return None
        return self.accepted_draft_tokens / self.generation_tokens

    @property
    def drafter_acceptance_rate(self) -> float | None:
        """Classical acceptance rate: accepted / proposed (per-position).

        ``None`` when the drafter didn't run *or* when it doesn't track
        proposal counts (e.g. external-model drafter via mlx_lm). The
        pipelined drafter tracks this. Differs from
        :attr:`drafter_acceptance_fraction`: this divides by total drafts
        proposed (the standard literature metric for drafter quality);
        ``drafter_acceptance_fraction`` divides by total emitted tokens
        (the metric for end-to-end speedup).
        """
        if self.drafter_model_id is None or self.proposed_draft_tokens == 0:
            return None
        return self.accepted_draft_tokens / self.proposed_draft_tokens


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice | StreamingChoiceResponse]
    usage: Usage | None = None
    service_tier: str | None = None
    # Non-OpenAI extension: full generation stats for the request,
    # including spec-decode telemetry (drafter id, mode, K, accepted /
    # proposed draft tokens, spec rounds, peak memory, prefill TPS).
    # Standard OpenAI clients ignore unknown fields; exo's own benches
    # and dashboards read this for drafter-effectiveness reporting.
    # ``None`` for endpoints that don't run a generation pipeline (e.g.
    # tool-call-only completions).
    generation_stats: GenerationStats | None = None


class ImageGenerationStats(BaseModel):
    seconds_per_step: float
    total_generation_time: float

    num_inference_steps: int
    num_images: int

    image_width: int
    image_height: int

    peak_memory_usage: Memory


class NodePowerStats(BaseModel, frozen=True):
    node_id: NodeId
    samples: int
    avg_sys_power: float


class PowerUsage(BaseModel, frozen=True):
    elapsed_seconds: float
    nodes: list[NodePowerStats]
    total_avg_sys_power_watts: float
    total_energy_joules: float


class BenchChatCompletionResponse(ChatCompletionResponse):
    generation_stats: GenerationStats | None = None
    power_usage: PowerUsage | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: ModelId
    frequency_penalty: float | None = None
    messages: list[ChatCompletionMessage]
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[dict[str, Any]] | None = None
    reasoning_effort: ReasoningEffort | None = None
    enable_thinking: bool | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None
    # Speculative-decoding per-request overrides (item 9). These are exo
    # extensions to the OpenAI Chat Completions schema -- standard clients
    # ignore unknown fields and get the runner's defaults.
    #
    # ``use_drafter=False`` short-circuits to non-speculative; clients that
    # want a finer-grained switch use ``draft_mode`` to pick a specific
    # strategy. When both are set, the explicit ``draft_mode`` wins
    # (matches ``TextGenerationTaskParams`` resolution in
    # ``resolve_draft_mode``); see
    # ``src/exo/worker/engines/mlx/generator/drafter.py``.
    use_drafter: bool | None = None
    num_draft_tokens: int | None = Field(
        default=None,
        ge=1,
        le=MAX_NUM_DRAFT_TOKENS_PER_REQUEST,
        description=(
            "Per-request override for the number of speculative draft tokens "
            "per round (K). Validated as a positive integer up to "
            f"{MAX_NUM_DRAFT_TOKENS_PER_REQUEST} (a sanity guard against "
            "pathological values). The runner clamps K to its actual "
            "wire-protocol budget (``EXO_NUM_DRAFT_TOKENS``) internally, so "
            "benchmarking flows that sweep large K values are not gated by "
            "this bound."
        ),
    )
    # Per-request draft-strategy override. ``"model"`` uses the external
    # drafter, ``"pipelined"`` uses the pipelined+remote drafter, ``"ngram"``
    # uses CPU n-gram tables, ``"none"`` disables speculation. ``None`` defers
    # to the model card / runner default. Mirrors ``draft_mode`` on the task.
    draft_mode: Literal["model", "pipelined", "ngram", "none"] | None = None


class BenchChatCompletionRequest(ChatCompletionRequest):
    use_prefix_cache: bool = False


class AddCustomModelParams(BaseModel):
    model_id: ModelId


class HuggingFaceSearchResult(BaseModel):
    id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    tags: list[str] = Field(default_factory=list)


class PlaceInstanceParams(BaseModel):
    model_id: ModelId
    sharding: Sharding = Sharding.Pipeline
    instance_meta: InstanceMeta = InstanceMeta.MlxRing
    min_nodes: int = 1


class CreateInstanceParams(BaseModel):
    instance: Instance


class PlacementPreview(BaseModel):
    model_id: ModelId
    sharding: Sharding
    instance_meta: InstanceMeta
    instance: Instance | None = None
    # Keys are NodeId strings, values are additional bytes that would be used on that node
    memory_delta_by_node: dict[str, int] | None = None
    error: str | None = None


class PlacementPreviewResponse(BaseModel):
    previews: list[PlacementPreview]


class DeleteInstanceTaskParams(BaseModel):
    instance_id: str


class CreateInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    model_card: ModelCard


class DeleteInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    instance_id: InstanceId


class CancelCommandResponse(BaseModel):
    message: str
    command_id: CommandId


class InstanceLinkBody(BaseModel):
    prefill_instances: list[InstanceId]
    decode_instances: list[InstanceId]


class InstanceLinkResponse(BaseModel):
    message: str
    command_id: CommandId


ImageSize = Literal[
    "auto",
    "512x512",
    "768x768",
    "1024x768",
    "768x1024",
    "1024x1024",
    "1024x1536",
    "1536x1024",
]


def normalize_image_size(v: object) -> ImageSize:
    """Shared validator for ImageSize fields: maps None → "auto" and rejects invalid values."""
    if v is None:
        return "auto"
    if v not in get_args(ImageSize):
        raise ValueError(f"Invalid size: {v!r}. Must be one of {get_args(ImageSize)}")
    return v  # pyright: ignore[reportReturnType]


class AdvancedImageParams(BaseModel):
    seed: Annotated[int, Field(ge=0)] | None = None
    num_inference_steps: Annotated[int, Field(ge=1, le=100)] | None = None
    guidance: Annotated[float, Field(ge=1.0, le=20.0)] | None = None
    negative_prompt: str | None = None
    num_sync_steps: Annotated[int, Field(ge=1, le=100)] | None = None


class ImageGenerationTaskParams(BaseModel):
    prompt: str
    background: str | None = None
    model: str
    moderation: str | None = None
    n: int | None = 1
    output_compression: int | None = None
    output_format: Literal["png", "jpeg", "webp"] = "png"
    partial_images: int | None = 0
    quality: Literal["high", "medium", "low"] | None = "medium"
    response_format: Literal["url", "b64_json"] | None = "b64_json"
    size: ImageSize = "auto"
    stream: bool | None = False
    style: str | None = "vivid"
    user: str | None = None
    advanced_params: AdvancedImageParams | None = None
    # Internal flag for benchmark mode - set by API, preserved through serialization
    bench: bool = False

    @field_validator("size", mode="before")
    @classmethod
    def normalize_size(cls, v: object) -> ImageSize:
        return normalize_image_size(v)


class BenchImageGenerationTaskParams(ImageGenerationTaskParams):
    bench: bool = True


class ImageEditsTaskParams(BaseModel):
    """Internal task params for image-editing requests."""

    image_data: str = ""  # Base64-encoded image (empty when using chunked transfer)
    total_input_chunks: int = 0
    prompt: str
    model: str
    n: int | None = 1
    quality: Literal["high", "medium", "low"] | None = "medium"
    output_format: Literal["png", "jpeg", "webp"] = "png"
    response_format: Literal["url", "b64_json"] | None = "b64_json"
    size: ImageSize = "auto"
    image_strength: float | None = 0.7
    stream: bool = False
    partial_images: int | None = 0
    advanced_params: AdvancedImageParams | None = None
    bench: bool = False

    @field_validator("size", mode="before")
    @classmethod
    def normalize_size(cls, v: object) -> ImageSize:
        return normalize_image_size(v)

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "image_data":
                yield name, f"<{len(self.image_data)} chars>"
            elif name is not None:
                yield name, value


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "b64_json" and self.b64_json is not None:
                yield name, f"<{len(self.b64_json)} chars>"
            elif name is not None:
                yield name, value


class ImageGenerationResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageData]


class BenchImageGenerationResponse(ImageGenerationResponse):
    generation_stats: ImageGenerationStats | None = None
    power_usage: PowerUsage | None = None


class ImageListItem(BaseModel, frozen=True):
    image_id: str
    url: str
    content_type: str
    expires_at: float


class ImageListResponse(BaseModel, frozen=True):
    data: list[ImageListItem]


class StartDownloadParams(FrozenModel):
    target_node_id: NodeId
    shard_metadata: ShardMetadata


class StartDownloadResponse(FrozenModel):
    command_id: CommandId


class DeleteDownloadResponse(FrozenModel):
    command_id: CommandId


class CancelDownloadParams(FrozenModel):
    target_node_id: NodeId
    model_id: ModelId


class CancelDownloadResponse(FrozenModel):
    command_id: CommandId


class TraceEventResponse(FrozenModel):
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str


class TraceResponse(FrozenModel):
    task_id: str
    traces: list[TraceEventResponse]


class TraceCategoryStats(FrozenModel):
    total_us: int
    count: int
    min_us: int
    max_us: int
    avg_us: float


class TraceRankStats(FrozenModel):
    by_category: dict[str, TraceCategoryStats]


class TraceStatsResponse(FrozenModel):
    task_id: str
    total_wall_time_us: int
    by_category: dict[str, TraceCategoryStats]
    by_rank: dict[int, TraceRankStats]


class TraceListItem(FrozenModel):
    task_id: str
    created_at: str
    file_size: int


class TraceListResponse(FrozenModel):
    traces: list[TraceListItem]


class DeleteTracesRequest(FrozenModel):
    task_ids: list[str]


class DeleteTracesResponse(FrozenModel):
    deleted: list[str]
    not_found: list[str]
