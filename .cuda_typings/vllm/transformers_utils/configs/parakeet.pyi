from dataclasses import dataclass
from transformers import ParakeetEncoderConfig, PretrainedConfig

class ParakeetConfig(ParakeetEncoderConfig):
    llm_hidden_size: int
    projection_hidden_size: int
    projection_bias: bool
    projection_eps: float
    sampling_rate: int
    @staticmethod
    def from_hf_config(
        config: PretrainedConfig, *, llm_hidden_size: int, max_model_len: int
    ) -> ParakeetConfig: ...

@dataclass(kw_only=True, frozen=True)
class ExtractorConfig:
    feature_size: int
    sampling_rate: int
    subsampling_factor: int
    subsampling_conv_kernel_size: int
    subsampling_conv_stride: int
    clip_duration_s: int = ...
    clip_min_duration_s: float = ...
    @staticmethod
    def from_hf_config(config: PretrainedConfig) -> ExtractorConfig: ...
