import torch
from .interfaces import supports_multimodal as supports_multimodal
from .interfaces_base import (
    VllmModelForPooling as VllmModelForPooling,
    is_pooling_model as is_pooling_model,
)
from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.config import ModelConfig as ModelConfig, VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.pooler import Pooler as Pooler
from vllm.model_executor.models.config import (
    VerifyAndUpdateConfig as VerifyAndUpdateConfig,
)
from vllm.transformers_utils.config import (
    try_get_dense_modules as try_get_dense_modules,
)
from vllm.transformers_utils.repo_utils import get_hf_file_bytes as get_hf_file_bytes

logger: Incomplete

def as_embedding_model(cls) -> _T: ...
def as_seq_cls_model(cls) -> _T: ...

class SequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None: ...

def load_weights_using_from_2_way_softmax(
    model, weights: Iterable[tuple[str, torch.Tensor]]
): ...
def load_weights_no_post_processing(
    model, weights: Iterable[tuple[str, torch.Tensor]]
): ...

SEQ_CLS_LOAD_METHODS: Incomplete

def seq_cls_model_loader(model, weights: Iterable[tuple[str, torch.Tensor]]): ...
