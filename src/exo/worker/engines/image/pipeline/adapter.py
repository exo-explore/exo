from enum import Enum
from typing import Any, Protocol

import mlx.core as mx


class AttentionInterface(Protocol):
    num_heads: int
    head_dimension: int
    to_q: Any
    to_k: Any
    to_v: Any
    norm_q: Any
    norm_k: Any
    to_out: list[Any]


class JointAttentionInterface(AttentionInterface, Protocol):
    add_q_proj: Any
    add_k_proj: Any
    add_v_proj: Any
    norm_added_q: Any
    norm_added_k: Any
    to_add_out: Any


class JointBlockInterface(Protocol):
    attn: JointAttentionInterface
    norm1: Any  # Callable module: (hidden_states, text_embeddings) -> tuple[5 arrays]
    norm1_context: (
        Any  # Callable module: (hidden_states, text_embeddings) -> tuple[5 arrays]
    )
    norm2: Any
    norm2_context: Any
    ff: Any
    ff_context: Any


class SingleBlockInterface(Protocol):
    attn: AttentionInterface
    norm: Any  # Callable module: (hidden_states, text_embeddings) -> tuple[2 arrays]

    def _apply_feed_forward_and_projection(
        self, norm_hidden_states: mx.array, attn_output: mx.array, gate: mx.array
    ) -> mx.array:
        """Apply feed forward network and projection."""
        ...


class BlockWrapperMode(Enum):
    CACHING = "caching"  # Sync mode: compute full attention, populate cache
    PATCHED = "patched"  # Async mode: compute patch attention, use cached KV
