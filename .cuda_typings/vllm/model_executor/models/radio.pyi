import torch
import torch.nn as nn
from _typeshed import Incomplete
from dataclasses import dataclass
from transformers import PretrainedConfig as PretrainedConfig
from typing import TypeAlias
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.model_executor.models.intern_vit import (
    InternParallelAttention as InternParallelAttention,
    InternVisionEncoder as InternVisionEncoder,
    InternVisionEncoderLayer as InternVisionEncoderLayer,
)

input_dim_t: TypeAlias = int | tuple[int, int]
norm_t: TypeAlias
to_1tuple: Incomplete
to_2tuple: Incomplete
to_3tuple: Incomplete
to_4tuple: Incomplete
to_ntuple: Incomplete

def calc_seq_len(size: tuple[int, int], patch_size: int) -> int: ...
def calc_seq_lens(sizes: list[tuple[int, int]], patch_size: int) -> list[int]: ...

class ClsToken(nn.Module):
    ndim: Incomplete
    enabled: Incomplete
    num_registers: int
    num_tokens: Incomplete
    token: Incomplete
    num_patches: Incomplete
    def __init__(
        self,
        ndim: int,
        num_tokens: int = 1,
        enabled: bool = True,
        register_multiple: int | None = None,
        num_registers: int | None = None,
    ) -> None: ...
    def forward(self, x: torch.Tensor): ...

class ViTPatchGenerator(nn.Module):
    cpe_mode: Incomplete
    pos_dropout: Incomplete
    return_pos_enc: Incomplete
    patch_size: Incomplete
    abs_pos: Incomplete
    embed_dim: Incomplete
    num_rows: Incomplete
    num_cols: Incomplete
    input_dims: Incomplete
    num_patches: Incomplete
    max_input_dims: Incomplete
    im_to_patches: Incomplete
    embedder: Incomplete
    pos_embed: Incomplete
    cls_token: Incomplete
    patch_normalizer: Incomplete
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        input_dims: input_dim_t,
        abs_pos: bool = True,
        normalize_patches: bool = False,
        cls_token: bool = False,
        max_input_dims: input_dim_t | None = None,
        pos_dropout: float = 0.0,
        return_pos_enc: bool = False,
        num_cls_tokens: int = 1,
        register_multiple: int | None = None,
        num_registers: int | None = None,
        patch_bias: bool = False,
        device=None,
        dtype=None,
    ) -> None: ...
    def forward(
        self, x: torch.Tensor, imgs_sizes: list[tuple[int, int]] | None = None
    ) -> torch.Tensor: ...
    def apply_pos_enc_dynamic(
        self, patches: torch.Tensor, imgs_sizes: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def cls_token_dynamic(
        self, patches: torch.Tensor, imgs_sizes: list[tuple[int, int]]
    ) -> torch.Tensor: ...
    @property
    def apply_cls_token(self): ...
    @property
    def num_cls_tokens(self): ...
    @property
    def num_cls_patches(self): ...
    @property
    def num_registers(self): ...
    @property
    def num_skip(self): ...
    def embed_patches(self, x: torch.Tensor) -> torch.Tensor: ...
    def apply_pos_enc(
        self,
        patches: torch.Tensor,
        patch_idxs: torch.Tensor | None = None,
        input_size: tuple[int, int] | None = None,
    ) -> torch.Tensor: ...
    def get_pos_enc(
        self,
        batch_size: int,
        patch_idxs: torch.Tensor | None = None,
        input_size: tuple[int, int] | None = None,
    ) -> torch.Tensor: ...

class Im2Patches(nn.Module):
    patch_size: Incomplete
    def __init__(self, patch_size: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class ViTPatchLinear(nn.Linear):
    patch_size: Incomplete
    def __init__(
        self, patch_size: int, embed_dim: int, bias: bool = False, **factory
    ) -> None: ...

@dataclass(frozen=True, kw_only=True)
class MaskMetadata:
    cu_seqlens: torch.Tensor
    max_seqlen: torch.Tensor

class RadioParallelAttention(InternParallelAttention):
    def forward(
        self, x: torch.Tensor, mask_meta: MaskMetadata | None = None
    ) -> torch.Tensor: ...

class RadioVisionEncoderLayer(InternVisionEncoderLayer):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(
        self, hidden_states: torch.Tensor, mask_meta: MaskMetadata | None = None
    ): ...

class RadioVisionEncoder(InternVisionEncoder):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(
        self, inputs_embeds: torch.Tensor, mask_meta: MaskMetadata | None = None
    ): ...

class RadioInternVisionModel(nn.Module):
    packed_modules_mapping: Incomplete
    config: Incomplete
    patch_generator: Incomplete
    encoder: Incomplete
    def __init__(
        self,
        config: PretrainedConfig = None,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None: ...
    def get_input_embeddings(self): ...
    def inter_image_mask_metadata(
        self, imgs_sizes: list[tuple[int, int]], device: torch.device
    ) -> MaskMetadata: ...
    def forward(
        self, x: torch.Tensor, imgs_sizes: list[tuple[int, int]] | None = None
    ) -> torch.FloatTensor: ...

class RadioModel(nn.Module):
    packed_modules_mapping: Incomplete
    config: Incomplete
    model: Incomplete
    summary_idxs: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None: ...
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        pixel_embeds: torch.Tensor | None = None,
        *,
        imgs_sizes: list[tuple[int, int]] | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]: ...
    def load_weights(self, weights) -> set[str]: ...
