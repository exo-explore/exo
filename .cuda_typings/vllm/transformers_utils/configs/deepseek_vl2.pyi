from _typeshed import Incomplete
from transformers import PretrainedConfig

class VisionEncoderConfig(PretrainedConfig):
    model_type: str
    model_name: str
    image_size: int
    patch_size: int
    width: int
    layers: int
    heads: int
    mlp_ratio: int
    global_pool: str
    ignore_head: bool
    class_token: bool
    num_classes: int
    use_checkpoint: bool
    weight_init: str
    deterministic: bool
    num_recomputing_layers: int
    def __init__(
        self,
        model_name: str = "vit_so400m_patch14_siglip_384.webli",
        image_size: int = 384,
        patch_size: int = 16,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        mlp_ratio: int = 4,
        global_pool: str = "map",
        ignore_head: bool = True,
        class_token: bool = False,
        num_classes: int = 0,
        use_checkpoint: bool = False,
        **kwargs,
    ) -> None: ...

class MlpProjectorConfig(PretrainedConfig):
    model_type: str
    projector_type: str
    input_dim: int
    n_embed: int
    depth: int
    mlp_ratio: int
    downsample_ratio: int
    token_pooling: bool
    def __init__(
        self,
        projector_type: str = "downsample_mlp_gelu",
        input_dim: int = 1152,
        n_embed: int = 2048,
        depth: int = 2,
        mlp_ratio: int = 1,
        downsample_ratio: int = 2,
        **kwargs,
    ) -> None: ...

class DeepseekVLV2Config(PretrainedConfig):
    model_type: str
    architectures: list[str] | None
    vision_config: VisionEncoderConfig
    projector_config: MlpProjectorConfig
    tile_tag: str
    global_view_pos: str
    candidate_resolutions: tuple[tuple[int, int]]
    text_config: Incomplete
    vocab_size: Incomplete
    def __init__(
        self,
        tile_tag: str = "tile_tag",
        global_view_pos: str = "head",
        candidate_resolutions: tuple[tuple[int, int]] = ((384, 384),),
        **kwargs,
    ) -> None: ...
