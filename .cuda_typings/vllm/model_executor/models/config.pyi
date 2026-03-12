from _typeshed import Incomplete
from vllm.config import ModelConfig as ModelConfig, VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.models import ModelRegistry as ModelRegistry
from vllm.platforms import current_platform as current_platform
from vllm.utils.math_utils import cdiv as cdiv, round_up as round_up
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE as STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec as FullAttentionSpec,
    MLAAttentionSpec as MLAAttentionSpec,
    MambaSpec as MambaSpec,
)

logger: Incomplete

class VerifyAndUpdateConfig:
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None: ...
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class DeepseekV32ForCausalLM(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: VllmConfig) -> None: ...

class Ernie4_5_VLMoeForConditionalGenerationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None: ...

class Gemma3TextModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class GptOssForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None: ...

class GteNewModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class HybridAttentionMambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: VllmConfig) -> None: ...

class JambaForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class JinaRobertaModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class JinaVLForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class LlamaBidirectionalConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class LlamaNemotronVLConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class MambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: VllmConfig) -> None: ...

class NemotronHForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None: ...

class NemotronHNanoVLV2Config(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class NomicBertModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class Qwen2ForProcessRewardModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class Qwen2ForRewardModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class Qwen3ForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class Qwen3VLForSequenceClassificationConfig(Qwen3ForSequenceClassificationConfig): ...

class Qwen3_5ForConditionalGenerationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None: ...

class SnowflakeGteNewModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

class VoyageQwen3BidirectionalEmbedModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None: ...

MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]]
