import abc
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.config.utils import getattr_iter as getattr_iter
from vllm.model_executor.layers.pooler import DispatchPooler as DispatchPooler
from vllm.model_executor.models.interfaces import (
    SupportsCrossEncoding as SupportsCrossEncoding,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling as VllmModelForPooling,
)

class EmbeddingMixin(VllmModelForPooling, metaclass=abc.ABCMeta):
    default_seq_pooling_type: str
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class SequenceClassificationMixin(
    SupportsCrossEncoding, VllmModelForPooling, metaclass=abc.ABCMeta
):
    default_seq_pooling_type: str
    classifier: Incomplete
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
