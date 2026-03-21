from datetime import timedelta
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    model_validator,
)

from exo_core.models import CamelCaseModel, TaggedModel
from exo_core.types.common import NodeId
from exo_core.types.shards import ShardMetadata
from exo_core.utils.memory import Memory


class DownloadProgressData(CamelCaseModel):
    total: Memory
    downloaded: Memory
    downloaded_this_session: Memory

    completed_files: int
    total_files: int

    speed: float
    eta_ms: int

    files: dict[str, "DownloadProgressData"]


class BaseDownloadProgress(TaggedModel):
    node_id: NodeId
    shard_metadata: ShardMetadata
    model_directory: str = ""


class DownloadPending(BaseDownloadProgress):
    downloaded: Memory = Memory()
    total: Memory = Memory()


class DownloadCompleted(BaseDownloadProgress):
    total: Memory
    read_only: bool = False


class DownloadFailed(BaseDownloadProgress):
    error_message: str


class DownloadOngoing(BaseDownloadProgress):
    download_progress: DownloadProgressData


DownloadProgress = (
    DownloadPending | DownloadCompleted | DownloadFailed | DownloadOngoing
)


class ModelSafetensorsIndexMetadata(BaseModel):
    total_size: PositiveInt


class ModelSafetensorsIndex(BaseModel):
    metadata: ModelSafetensorsIndexMetadata | None
    weight_map: dict[str, str]


class FileListEntry(BaseModel):
    type: Literal["file", "directory"]
    path: str
    size: int | None = None


class RepoFileDownloadProgress(BaseModel):
    repo_id: str
    repo_revision: str
    file_path: str
    downloaded: Memory
    downloaded_this_session: Memory
    total: Memory
    speed: float
    eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]
    start_time: float

    model_config = ConfigDict(frozen=True)


class RepoDownloadProgress(BaseModel):
    repo_id: str
    repo_revision: str
    shard: ShardMetadata
    completed_files: int
    total_files: int
    downloaded: Memory
    downloaded_this_session: Memory
    total: Memory
    overall_speed: float
    overall_eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]
    file_progress: dict[str, RepoFileDownloadProgress] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class ConfigData(BaseModel):
    model_config = {"extra": "ignore"}  # Allow unknown fields

    architectures: list[str] | None = None
    hidden_size: Annotated[int, Field(ge=0)] | None = None
    num_key_value_heads: PositiveInt | None = None
    layer_count: int = Field(
        validation_alias=AliasChoices(
            "num_hidden_layers",
            "num_layers",
            "n_layer",
            "n_layers",
            "num_decoder_layers",
            "decoder_layers",
        )
    )

    @property
    def supports_tensor(self) -> bool:
        return self.architectures in [
            ["Glm4MoeLiteForCausalLM"],
            ["GlmMoeDsaForCausalLM"],
            ["DeepseekV32ForCausalLM"],
            ["DeepseekV3ForCausalLM"],
            ["Qwen3NextForCausalLM"],
            ["Qwen3MoeForCausalLM"],
            ["Qwen3_5MoeForConditionalGeneration"],
            ["Qwen3_5ForConditionalGeneration"],
            ["MiniMaxM2ForCausalLM"],
            ["LlamaForCausalLM"],
            ["GptOssForCausalLM"],
            ["Step3p5ForCausalLM"],
            ["NemotronHForCausalLM"],
        ]

    @model_validator(mode="before")
    @classmethod
    def defer_to_text_config(cls, data: dict[str, Any]):
        text_config = data.get("text_config")
        if text_config is None:
            return data

        for field in [
            "architectures",
            "hidden_size",
            "num_key_value_heads",
            "num_hidden_layers",
            "num_layers",
            "n_layer",
            "n_layers",
            "num_decoder_layers",
            "decoder_layers",
        ]:
            if (val := text_config.get(field)) is not None:  # pyright: ignore[reportAny]
                data[field] = val

        return data
