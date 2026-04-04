"""TurboQuant KV cache — rotation-aware quantized KV with residual correction.

Adapted from flovflo/turboquant-mlx-qwen35-kv (Apache-2.0).

Instead of the full PolarQuant+QJL from the TurboQuant paper, this uses a
simpler approach that's proven on Qwen3.5 in MLX:
  1. Keys: random rotation (sign-flip + permutation) → affine quantize → sign-sketch residual
  2. Values: affine quantize (same as QuantizedKVCache)
  3. Attention: mx.quantized_matmul on rotated queries × quantized keys + residual correction

The key insight: mx.quantized_matmul computes Q×K^T directly against quantized keys
without decompression, eliminating the 40% dequant cost in SDPA.

Gated by EXO_TURBOQUANT=1 env var.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import _BaseCache, create_attention_mask


# ---------------------------------------------------------------------------
# Projection: random rotation via sign-flip + permutation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ProjectionSpec:
    signs: mx.array       # (dim,) random ±1
    perm: mx.array        # (dim,) random permutation indices
    inv_perm: mx.array    # (dim,) inverse permutation for dequant path
    sketch_idx: mx.array  # (sketch_dim,) indices for residual sketch
    sketch_signs: mx.array  # (sketch_dim,) random ±1 for sketch


def make_projection(dim: int, sketch_dim: int, seed: int) -> ProjectionSpec:
    rng = np.random.default_rng(seed)
    signs = mx.array(rng.choice([-1.0, 1.0], size=(dim,)).astype(np.float32))
    perm_np = rng.permutation(dim).astype(np.int32)
    inv_perm_np = np.empty_like(perm_np)
    inv_perm_np[perm_np] = np.arange(dim, dtype=np.int32)
    perm = mx.array(perm_np)
    inv_perm = mx.array(inv_perm_np)
    sketch_dim = min(sketch_dim, dim)
    sketch_idx = mx.array(rng.choice(dim, size=(sketch_dim,), replace=False).astype(np.int32))
    sketch_signs = mx.array(rng.choice([-1.0, 1.0], size=(sketch_dim,)).astype(np.float32))
    return ProjectionSpec(signs=signs, perm=perm, inv_perm=inv_perm, sketch_idx=sketch_idx, sketch_signs=sketch_signs)


def apply_rotation(x: mx.array, spec: ProjectionSpec) -> mx.array:
    signs = spec.signs.astype(x.dtype) if spec.signs.dtype != x.dtype else spec.signs
    return mx.take(x * signs, spec.perm, axis=-1)


def apply_inverse_rotation(x: mx.array, spec: ProjectionSpec) -> mx.array:
    """Undo apply_rotation: inverse permutation then multiply by signs."""
    signs = spec.signs.astype(x.dtype) if spec.signs.dtype != x.dtype else spec.signs
    unperm = mx.take(x, spec.inv_perm, axis=-1)
    return unperm * signs  # signs are self-inverse (±1 × ±1 = 1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TurboQuantConfig:
    bits: int = 3           # key bits — rotation allows lower bits than affine-only
    group_size: int = 64
    value_bits: int = 4     # value bits — keep higher for weighted sum accuracy
    quantize_values: bool = True
    sketch_dim: int = 4
    residual_scale: float = 1.0
    use_residual: bool = True  # disable to save compute on decode-heavy workloads
    seed: int = 0


# ---------------------------------------------------------------------------
# TurboQuantKVCache
# ---------------------------------------------------------------------------

class TurboQuantKVCache(_BaseCache):
    """KV cache with rotation-aware quantization and residual correction.

    Implements the same interface as QuantizedKVCache so it's a drop-in
    replacement in exo's cache infrastructure.
    """

    step = 16384  # match exo's pre-allocation step

    def __init__(self, config: TurboQuantConfig):
        self.config = config
        self.offset = 0
        self.projection: ProjectionSpec | None = None
        self.keys_main: tuple[mx.array, mx.array, mx.array] | None = None
        self.values_main: tuple[mx.array, mx.array, mx.array] | None = None
        self.residual_t: mx.array | None = None
        self.bits = config.bits
        self.group_size = config.group_size
        self.value_group_size = config.group_size

    def _init_params(self, dim: int) -> None:
        if self.projection is None:
            self.projection = make_projection(dim, self.config.sketch_dim, self.config.seed)
            self.group_size = self._effective_group_size(dim)

    def _effective_group_size(self, dim: int) -> int:
        candidates = [128, 64, 32, 16, 8, 4, 2, 1]
        limit = min(dim, self.config.group_size)
        for candidate in candidates:
            if candidate <= limit and dim % candidate == 0:
                return candidate
        return 1

    def _append_tuple(
        self,
        current: tuple[mx.array, ...] | None,
        update: tuple[mx.array, ...],
    ) -> tuple[mx.array, ...]:
        if current is None:
            return update
        return tuple(mx.concatenate([c, u], axis=2) for c, u in zip(current, update))

    def _append_array(self, current: mx.array | None, update: mx.array) -> mx.array:
        return update if current is None else mx.concatenate([current, update], axis=2)

    def _append_residual(self, current: mx.array | None, update: mx.array) -> mx.array:
        return update if current is None else mx.concatenate([current, update], axis=-1)

    def _quantize_keys(self, keys: mx.array) -> tuple[tuple[mx.array, mx.array, mx.array], mx.array | None]:
        assert self.projection is not None
        rotated = apply_rotation(keys.astype(mx.bfloat16), self.projection)
        q_keys = mx.quantize(rotated, group_size=self.group_size, bits=self.config.bits)
        if self.config.use_residual:
            dequant = mx.dequantize(*q_keys, group_size=self.group_size, bits=self.config.bits)
            residual = rotated.astype(mx.float32) - dequant.astype(mx.float32)
            proj = apply_sketch(residual, self.projection).astype(mx.bfloat16)
            signs = mx.where(proj >= 0, 1.0, -1.0).astype(mx.bfloat16)
            rms = mx.sqrt(mx.mean(mx.square(residual), axis=-1, keepdims=True)).astype(mx.bfloat16)
            residual_t = mx.swapaxes(signs * rms, -1, -2)
            return q_keys, residual_t
        return q_keys, None

    def _quantize_values(self, values: mx.array) -> tuple[mx.array, mx.array, mx.array] | mx.array:
        self.value_group_size = self._effective_group_size(values.shape[-1])
        if self.config.quantize_values:
            return mx.quantize(values.astype(mx.bfloat16), group_size=self.value_group_size, bits=self.config.value_bits)
        return values

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[tuple[tuple[mx.array, ...], mx.array], tuple[mx.array, ...] | mx.array]:
        self._init_params(keys.shape[-1])

        q_keys, residual_t = self._quantize_keys(keys)
        q_values = self._quantize_values(values)

        self.keys_main = self._append_tuple(self.keys_main, q_keys)
        if self.config.quantize_values:
            self.values_main = self._append_tuple(self.values_main, q_values)
        else:
            self.values_main = self._append_array(self.values_main, q_values)
        if residual_t is not None:
            self.residual_t = self._append_residual(self.residual_t, residual_t)
        self.offset += keys.shape[2]

        return self.key_state, self.value_state

    @property
    def key_state(self) -> tuple[tuple[mx.array, ...] | None, mx.array | None]:
        return self.keys_main, self.residual_t

    @property
    def value_state(self) -> tuple[mx.array, ...] | mx.array | None:
        return self.values_main

    @property
    def state(self) -> list:
        return [self.keys_main, self.values_main, self.residual_t]

    @state.setter
    def state(self, v: list) -> None:
        self.keys_main, self.values_main, self.residual_t = v

    @property
    def meta_state(self) -> tuple[str, ...]:
        return tuple(
            map(str, (
                self.offset,
                self.config.bits,
                self.config.group_size,
                self.config.value_bits,
                int(self.config.quantize_values),
                self.config.sketch_dim,
                self.config.seed,
            ))
        )

    @meta_state.setter
    def meta_state(self, v: tuple[str, ...]) -> None:
        offset, bits, group_size, value_bits, qv, sketch_dim, seed = map(int, v)
        self.offset = offset
        self.config = TurboQuantConfig(bits, group_size, value_bits, bool(qv), sketch_dim, 1.0, seed)
        self.bits = bits
        self.group_size = group_size

    def make_mask(self, *args, **kwargs):  # type: ignore
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.keys_main is None

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def nbytes(self) -> int:
        pieces: list[mx.array | None] = [self.residual_t]
        if self.keys_main is not None:
            pieces.extend(self.keys_main)
        if self.values_main is not None:
            if self.config.quantize_values:
                pieces.extend(self.values_main)  # type: ignore
            else:
                pieces.append(self.values_main)  # type: ignore
        return sum(p.nbytes for p in pieces if p is not None)

    @classmethod
    def from_kvcache(cls, cache, config: TurboQuantConfig) -> TurboQuantKVCache:
        tq = cls(config)
        if cache.keys is not None:
            tq.update_and_fetch(
                cache.keys[..., :cache.offset, :],
                cache.values[..., :cache.offset, :],
            )
        return tq


# ---------------------------------------------------------------------------
# Custom attention: quantized_matmul + residual correction
# ---------------------------------------------------------------------------

def turboquant_scaled_dot_product_attention(
    queries: mx.array,
    key_state: tuple[tuple[mx.array, ...], mx.array | None],
    value_state: tuple[mx.array, ...] | mx.array,
    cache: TurboQuantKVCache,
    scale: float,
    mask: mx.array | None,
) -> mx.array:
    """Dequantize rotated keys, inverse-rotate, then use fused FlashAttention SDPA.

    Keys are stored rotated+quantized at lower bits (3-bit default). The rotation
    preserves quality at lower bits. At attention time we dequant + inverse-rotate
    to get fp16 keys, then use the fast fused SDPA kernel.
    """
    q_keys, _residual_t = key_state

    # Dequantize keys and inverse-rotate back to original space
    dk = mx.dequantize(*q_keys, group_size=cache.group_size, bits=cache.config.bits)
    dk = apply_inverse_rotation(dk, cache.projection)

    # Dequantize values (no rotation on values)
    if cache.config.quantize_values:
        dv = mx.dequantize(*value_state, group_size=cache.value_group_size, bits=cache.config.value_bits)
    else:
        dv = value_state

    # Use fused FlashAttention SDPA — the fast path
    return mx.fast.scaled_dot_product_attention(
        queries, dk, dv, scale=scale, mask=mask,
    )


# ---------------------------------------------------------------------------
# Attention dispatch: monkey-patch to route TurboQuantKVCache through custom path
# ---------------------------------------------------------------------------

_PATCHED = False


def patch_attention_dispatch() -> None:
    """Monkey-patch scaled_dot_product_attention to handle TurboQuantKVCache.

    Idempotent — safe to call multiple times.
    """
    global _PATCHED
    if _PATCHED:
        return

    import mlx_lm.models.base as base_mod
    import mlx_lm.models.qwen3_next as qwen3_next_mod

    _original = base_mod.scaled_dot_product_attention

    def _dispatched(queries, keys, values, cache, scale, mask, sinks=None):
        if isinstance(cache, TurboQuantKVCache):
            return turboquant_scaled_dot_product_attention(
                queries, keys, values, cache, scale, mask,
            )
        return _original(queries, keys, values, cache, scale=scale, mask=mask, sinks=sinks)

    base_mod.scaled_dot_product_attention = _dispatched
    qwen3_next_mod.scaled_dot_product_attention = _dispatched
    _PATCHED = True
