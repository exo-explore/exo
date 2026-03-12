import abc
from vllm.compilation.decorators import support_torch_compile as support_torch_compile
from vllm.model_executor.models.transformers.base import Base as Base
from vllm.model_executor.models.transformers.causal import CausalMixin as CausalMixin
from vllm.model_executor.models.transformers.legacy import LegacyMixin as LegacyMixin
from vllm.model_executor.models.transformers.moe import MoEMixin as MoEMixin
from vllm.model_executor.models.transformers.multimodal import (
    DYNAMIC_ARG_DIMS as DYNAMIC_ARG_DIMS,
    MultiModalDummyInputsBuilder as MultiModalDummyInputsBuilder,
    MultiModalMixin as MultiModalMixin,
    MultiModalProcessingInfo as MultiModalProcessingInfo,
    MultiModalProcessor as MultiModalProcessor,
)
from vllm.model_executor.models.transformers.pooling import (
    EmbeddingMixin as EmbeddingMixin,
    SequenceClassificationMixin as SequenceClassificationMixin,
)
from vllm.model_executor.models.transformers.utils import (
    can_enable_torch_compile as can_enable_torch_compile,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY

class TransformersForCausalLM(CausalMixin, Base, metaclass=abc.ABCMeta): ...
class TransformersMoEForCausalLM(
    MoEMixin, CausalMixin, Base, metaclass=abc.ABCMeta
): ...
class TransformersMultiModalForCausalLM(
    MultiModalMixin, CausalMixin, Base, metaclass=abc.ABCMeta
): ...
class TransformersMultiModalMoEForCausalLM(
    MoEMixin, MultiModalMixin, CausalMixin, Base, metaclass=abc.ABCMeta
): ...
class TransformersEmbeddingModel(
    EmbeddingMixin, LegacyMixin, Base, metaclass=abc.ABCMeta
): ...
class TransformersMoEEmbeddingModel(
    EmbeddingMixin, MoEMixin, Base, metaclass=abc.ABCMeta
): ...
class TransformersMultiModalEmbeddingModel(
    EmbeddingMixin, MultiModalMixin, Base, metaclass=abc.ABCMeta
): ...
class TransformersForSequenceClassification(
    SequenceClassificationMixin, LegacyMixin, Base, metaclass=abc.ABCMeta
): ...
class TransformersMoEForSequenceClassification(
    SequenceClassificationMixin, MoEMixin, Base, metaclass=abc.ABCMeta
): ...
class TransformersMultiModalForSequenceClassification(
    SequenceClassificationMixin, MultiModalMixin, Base, metaclass=abc.ABCMeta
): ...

def __getattr__(name: str): ...
