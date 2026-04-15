"""Install the attention/MoE split DecoderLayer.__call__ on a Qwen3.5 MoE model.

Unlike patches/qwen3_5_moe/apply.py (which installs fused local kernels),
this patch replaces DecoderLayer.__call__ with a two-rank split: rank 0 runs
attention + first residual, rank 1 runs post_attention_layernorm + MoE +
second residual, with one cross-rank send/recv pair per layer.

Must be called instead of apply_qwen35_batched_fused_patches — they both
replace DecoderLayer.__call__. In exo's current distributed load path
(utils_mlx.shard_and_load) maybe_apply_patches is only called in the
single-device branch, so there is no conflict for distributed AttnMoeSplit
runs — we call this directly from attn_moe_split_auto_parallel.
"""

import mlx.core as mx
import mlx.nn as nn
from loguru import logger
from mlx_lm.models.qwen3_5 import DecoderLayer

from .decoder import make_split_decoder_call


def apply_qwen35_attn_moe_split_patches(
    model: nn.Module, group: mx.distributed.Group
) -> nn.Module:
    """Install the split DecoderLayer.__call__ on a Qwen3.5 MoE model."""
    if group.size() != 2:
        raise ValueError(
            f"Qwen3.5 attn/moe split requires world_size==2, got {group.size()}"
        )

    DecoderLayer.__call__ = make_split_decoder_call(group)  # type: ignore[method-assign]

    logger.info(
        f"Qwen3.5 attn/moe split patch applied on rank {group.rank()}/{group.size()}"
    )
    return model
