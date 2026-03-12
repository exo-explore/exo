from _typeshed import Incomplete
from transformers import PretrainedConfig, SiglipVisionConfig
from transformers.models.qwen2 import Qwen2Config

class BagelConfig(PretrainedConfig):
    model_type: str
    visual_gen: Incomplete
    visual_und: Incomplete
    llm_config: Incomplete
    vit_config: Incomplete
    vae_config: Incomplete
    latent_patch_size: Incomplete
    max_latent_size: Incomplete
    vit_max_num_patch_per_side: Incomplete
    connector_act: Incomplete
    interpolate_pos: Incomplete
    timestep_shift: Incomplete
    def __init__(
        self,
        visual_gen: bool = True,
        visual_und: bool = True,
        llm_config: dict | Qwen2Config | None = None,
        vit_config: dict | SiglipVisionConfig | None = None,
        vae_config: dict | None = None,
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
        vit_max_num_patch_per_side: int = 70,
        connector_act: str = "gelu_pytorch_tanh",
        interpolate_pos: bool = False,
        timestep_shift: float = 1.0,
        **kwargs,
    ) -> None: ...
    @property
    def hidden_size(self) -> int: ...
