import abc
import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import BatchFeature as BatchFeature
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.models.mistral3 import (
    Mistral3DummyInputsBuilder as Mistral3DummyInputsBuilder,
    Mistral3ForConditionalGeneration as Mistral3ForConditionalGeneration,
    Mistral3MultiModalProjector as Mistral3MultiModalProjector,
    Mistral3ProcessingInfo as Mistral3ProcessingInfo,
    init_vision_tower_for_llava as init_vision_tower_for_llava,
)
from vllm.model_executor.models.pixtral import (
    PixtralHFEncoderInfo as PixtralHFEncoderInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader as AutoWeightsLoader,
    WeightsMapper as WeightsMapper,
    init_vllm_registered_model as init_vllm_registered_model,
    maybe_prefix as maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY
from vllm.multimodal.cache import (
    BaseMultiModalProcessorCache as BaseMultiModalProcessorCache,
)
from vllm.multimodal.inputs import (
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems as ImageProcessorItems,
    MultiModalDataItems as MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    PromptReplacement as PromptReplacement,
    PromptUpdate as PromptUpdate,
    PromptUpdateDetails as PromptUpdateDetails,
)

class LightOnOCRMultiModalProcessor(
    BaseMultiModalProcessor[Mistral3ProcessingInfo]
): ...

class LightOnOCRForConditionalGeneration(
    Mistral3ForConditionalGeneration, metaclass=abc.ABCMeta
):
    hf_to_vllm_mapper: Incomplete
    config: Incomplete
    multimodal_config: Incomplete
    vision_tower: Incomplete
    multi_modal_projector: Incomplete
    language_model: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
