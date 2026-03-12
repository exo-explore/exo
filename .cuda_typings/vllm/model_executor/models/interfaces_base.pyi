import torch
from _typeshed import Incomplete
from typing import ClassVar, Literal, Protocol, overload
from typing_extensions import TypeIs, TypeVar
from vllm.config import VllmConfig as VllmConfig
from vllm.config.model import AttnTypeStr as AttnTypeStr
from vllm.config.pooler import (
    SequencePoolingType as SequencePoolingType,
    TokenPoolingType as TokenPoolingType,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.pooler import Pooler as Pooler
from vllm.tasks import ScoreType as ScoreType
from vllm.utils.func_utils import supports_kw as supports_kw

logger: Incomplete
T = TypeVar("T", default=torch.Tensor)
T_co = TypeVar("T_co", default=torch.Tensor, covariant=True)

class VllmModel(Protocol[T_co]):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co: ...

@overload
def is_vllm_model(model: type[object]) -> TypeIs[type[VllmModel]]: ...
@overload
def is_vllm_model(model: object) -> TypeIs[VllmModel]: ...

class VllmModelForTextGeneration(VllmModel[T], Protocol[T]):
    def compute_logits(self, hidden_states: T) -> T | None: ...

@overload
def is_text_generation_model(
    model: type[object],
) -> TypeIs[type[VllmModelForTextGeneration]]: ...
@overload
def is_text_generation_model(model: object) -> TypeIs[VllmModelForTextGeneration]: ...

class VllmModelForPooling(VllmModel[T_co], Protocol[T_co]):
    is_pooling_model: ClassVar[Literal[True]]
    default_seq_pooling_type: ClassVar[SequencePoolingType]
    default_tok_pooling_type: ClassVar[TokenPoolingType]
    attn_type: ClassVar[AttnTypeStr]
    score_type: ClassVar[ScoreType]
    pooler: Pooler

@overload
def is_pooling_model(model: type[object]) -> TypeIs[type[VllmModelForPooling]]: ...
@overload
def is_pooling_model(model: object) -> TypeIs[VllmModelForPooling]: ...
def default_pooling_type(
    *,
    seq_pooling_type: SequencePoolingType = "LAST",
    tok_pooling_type: TokenPoolingType = "ALL",
): ...
def get_default_seq_pooling_type(
    model: type[object] | object,
) -> SequencePoolingType: ...
def get_default_tok_pooling_type(model: type[object] | object) -> TokenPoolingType: ...
def attn_type(attn_type: AttnTypeStr): ...
def get_attn_type(model: type[object] | object) -> AttnTypeStr: ...
def get_score_type(model: type[object] | object) -> ScoreType: ...
