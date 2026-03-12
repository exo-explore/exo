from _typeshed import Incomplete
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

class ColQwen3Config(Qwen3VLConfig):
    model_type: str
    embed_dim: Incomplete
    dims: Incomplete
    dim: Incomplete
    projection_dim: Incomplete
    colbert_dim: Incomplete
    pooling: Incomplete
    def __init__(
        self,
        embed_dim: int | None = None,
        dims: int | None = None,
        dim: int | None = None,
        projection_dim: int | None = None,
        colbert_dim: int | None = None,
        pooling: str | None = None,
        **kwargs,
    ) -> None: ...

class OpsColQwen3Config(ColQwen3Config):
    model_type: str

class Qwen3VLNemotronEmbedConfig(ColQwen3Config):
    model_type: str
