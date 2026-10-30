"""DeepSeek-V4.1 (text): MLX model definition.

Provenance:
- Ported from ml-explore/mlx-lm `mlx_lm/models/deepseek_v41.py`
  (https://github.com/ml-explore/mlx-lm/blob/94cdcae/mlx_lm/models/deepseek_v41.py, added in
  commit 94cdcae "Add DeepSeek-V4.1 model (#1895)", unchanged up to 6b572ab1), MIT licence,
  Copyright (c) 2023-2026 Apple Inc. Copied 2026-10-30.
- Engram (conditional n-gram memory), the n-gram hashing and the RoPE / quantization
  arithmetic follow DeepSeek's reference implementation, inference/model.py and
  inference/engram.py of https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash at revision
  314113f8a7dadb388215396ba5fbee2c610a04d6, MIT licence, Copyright (c) 2023 DeepSeek.

Local changes against mlx-lm's file:
- Engram at engram_layer_ids (mlx-lm raises on the official config and drops the engram
  weights). The tables stay in the checkpoint files, memory-mapped and read on the host
  (engram_tables="host", the default: no GPU memory), or are loaded and gathered on the GPU
  (engram_tables="device"). Model.load_engram(model_path) attaches the tokenizer's
  compressed token map and the memory maps once the weights are loaded.
- RoPE frequencies are computed as the reference computes them, rather than by mlx-lm's
  YarnRoPE (so global rope patches do not change them).
- Power-of-two quantization scales are computed on the float32 bits, as the reference
  kernels do (log2 + ceil can be one off just above a power of two).
- The logits are float32, from a float32 head, as the reference's ParallelHead.
- Pipeline stages can be laid out in either rank order (exo gives rank 0 the first layers)
  and evaluate around their sends and receives.
- LayerCache can be snapshotted for exo's prefix cache (prefix_cache_snapshot).
- Strict typing; the generic n-gram hash is written out for numpy and for MLX.
"""

import json
import math
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Self, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import numpy.typing as npt
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear

IntArray = npt.NDArray[np.int64]
ByteArray = npt.NDArray[np.uint8]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v41"
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    head_dim: int = 512
    q_lora_rank: int = 1280
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    rms_norm_eps: float = 1e-20

    sliding_window: int = 128
    compress_ratios: tuple[int, ...] = ()
    kv_source_layer_ids: tuple[int, ...] = ()
    index_source_layer_ids: tuple[int, ...] = ()
    compress_rope_theta: float = 160000.0
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512

    rope_theta: float = 10000.0
    rope_scaling: dict[str, float | int | str] | None = None
    max_position_embeddings: int = 1048576
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 0
    # "host": the engram tables stay in the checkpoint files and their rows are read
    # through a memory map (no GPU memory); "device": they are loaded and gathered on
    # the GPU.
    engram_tables: str = "host"

    @classmethod
    def from_dict(cls, params: dict[str, object]) -> Self:
        text_config = params.get("text_config")
        if isinstance(text_config, dict):
            params = {
                **cast(dict[str, object], text_config),
                "model_type": params["model_type"],
            }
        return cast(Self, super().from_dict(params))  # pyright: ignore[reportUnknownMemberType]

    def __post_init__(self) -> None:
        self.compress_ratios = tuple(self.compress_ratios)
        self.kv_source_layer_ids = tuple(self.kv_source_layer_ids)
        self.index_source_layer_ids = tuple(self.index_source_layer_ids)
        self.engram_layer_ids = tuple(self.engram_layer_ids)
        self.engram_num_embeddings = tuple(self.engram_num_embeddings)
        if len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError("engram_num_embeddings needs one entry per engram layer")
        if self.engram_tables not in ("host", "device"):
            raise ValueError(
                f"engram_tables must be 'host' or 'device', not {self.engram_tables!r}"
            )


# ---------------------------------------------------------------------------------------
# Reference arithmetic


def rope_periods(
    dims: int, base: float, scaling: dict[str, float | int | str] | None
) -> mx.array:
    """precompute_freqs_cis of the reference: the base frequencies, with YaRN's ramp
    between the beta_fast and beta_slow dimensions when the layer uses YaRN. Returned as
    periods (1 / frequency), which is what mx.fast.rope takes."""
    freqs = 1.0 / (base ** (np.arange(0, dims, 2, dtype=np.float32) / dims))
    if scaling:
        original = int(scaling["original_max_position_embeddings"])
        factor = float(scaling["factor"])
        beta_fast = float(scaling.get("beta_fast", 32))
        beta_slow = float(scaling.get("beta_slow", 1))

        def corrected_dim(rotations: float) -> float:
            return (
                dims
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dims - 1)
        ramp = np.clip(
            (np.arange(dims // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1
        )
        smooth = 1 - ramp
        freqs = (freqs / factor * (1 - smooth) + freqs * smooth).astype(np.float32)
    return mx.array((1.0 / freqs).astype(np.float32))


class RoPE(nn.Module):
    """Interleaved-pair rotary embedding of the last `dims` features at the reference's
    frequencies; `scale` multiplies the positions (a compressed group sits at index x
    ratio)."""

    def __init__(
        self, dims: int, base: float, scaling: dict[str, float | int | str] | None
    ):
        super().__init__()
        self.dims = dims
        self._periods = rope_periods(dims, base, scaling)

    def __call__(
        self, x: mx.array, offset: int = 0, scale: float = 1.0, inverse: bool = False
    ) -> mx.array:
        periods = -self._periods if inverse else self._periods
        return mx.fast.rope(
            x,
            self.dims,
            traditional=True,
            base=None,
            scale=scale,
            offset=offset,
            freqs=periods,
        )


def _apply_rope(
    x: mx.array,
    rope: RoPE,
    *,
    rope_dim: int,
    offset: int = 0,
    scale: float = 1.0,
    inverse: bool = False,
) -> mx.array:
    """Rotate the trailing rope_dim features by position, in float32 (as the reference)."""
    head = x[..., :-rope_dim]
    tail = x[..., -rope_dim:].astype(mx.float32)
    if x.ndim == 4:  # [B, L, H, D]: rotate along L
        tail = rope(
            tail.transpose(0, 2, 1, 3), offset=offset, scale=scale, inverse=inverse
        ).transpose(0, 2, 1, 3)
    else:
        tail = rope(tail, offset=offset, scale=scale, inverse=inverse)
    return mx.concatenate([head, tail.astype(x.dtype)], axis=-1)


def _pow2_ceil(x: mx.array) -> mx.array:
    """2 ** ceil(log2(x)) for positive float32 x, exact on the bits (the reference
    kernels' fast_round_scale)."""
    bits = x.astype(mx.float32).view(mx.uint32)
    exponent = mx.bitwise_and(mx.right_shift(bits, 23), 0xFF).astype(mx.int32)
    rounds_up = mx.not_equal(mx.bitwise_and(bits, 0x7FFFFF), 0).astype(mx.int32)
    return mx.left_shift((exponent + rounds_up).astype(mx.uint32), 23).view(mx.float32)


def fake_quant_fp8(x: mx.array, block_size: int = 32) -> mx.array:
    """Round to FP8 E4M3 and back, with a power-of-two scale per block (the reference's
    in-place act_quant with ue8m0 scales): what the sliding-window KV stores."""
    dtype, shape = x.dtype, x.shape
    blocks = x.astype(mx.float32).reshape(*shape[:-1], -1, block_size)
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=-1, keepdims=True), 1e-4)
    scale = _pow2_ceil(amax * mx.array(1.0 / 448.0, dtype=mx.float32))
    q = mx.clip(blocks / scale, -448.0, 448.0)
    return mx.reshape(mx.from_fp8(mx.to_fp8(q), mx.float32) * scale, shape).astype(
        dtype
    )


_FP4_VALUES = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=mx.float32)


def _round_fp4(x: mx.array) -> mx.array:
    """Round to the nearest E2M1 value, ties to the even code."""
    magnitude = mx.abs(x)
    index = mx.zeros(magnitude.shape, dtype=mx.int32)
    for threshold in (0.25, 1.25, 2.5, 5.0):
        index = index + (magnitude > threshold).astype(mx.int32)
    for threshold in (0.75, 1.75, 3.5):
        index = index + (magnitude >= threshold).astype(mx.int32)
    value = _FP4_VALUES[index]
    return mx.where(x < 0, -value, value)


def fake_quant_fp4(
    x: mx.array, *, block_size: int, e4m3_scale: bool = False
) -> mx.array:
    """Round to FP4 E2M1 and back: an E4M3 scale per block (the compressed KV) or a power
    of two (the indexer's queries and keys), as the reference's fp4_act_quant."""
    dtype, shape = x.dtype, x.shape
    blocks = x.astype(mx.float32).reshape(*shape[:-1], -1, block_size)
    amax = mx.max(mx.abs(blocks), axis=-1, keepdims=True)
    if e4m3_scale:
        amax = mx.maximum(amax, 6.0 * 2.0**-9)
        scale = mx.from_fp8(mx.to_fp8(amax / 6.0), mx.float32)
    else:
        amax = mx.maximum(amax, 6.0 * 2.0**-126)
        scale = _pow2_ceil(amax * mx.array(1.0 / 6.0, dtype=mx.float32))
    q = _round_fp4(mx.clip(blocks / scale, -6.0, 6.0)) * scale
    return mx.reshape(q, shape).astype(dtype)


def _make_sinkhorn_kernel() -> "mx.fast.MetalKernel | None":
    if not mx.metal.is_available():
        return None
    # The loop of _sinkhorn in one launch, one thread per matrix, in the same order.
    source = """
        uint n = thread_position_in_grid.x;
        if (n >= comb_shape[0]) {
            return;
        }
        float m[HC * HC];
        for (int k = 0; k < HC * HC; k++) {
            m[k] = comb[n * HC * HC + k];
        }
        float e = eps[0];
        for (int t = 0; t < 2 * ITERS - 1; t++) {
            bool rows = t % 2 == 1;
            for (int a = 0; a < HC; a++) {
                float s = 0.0f;
                for (int b = 0; b < HC; b++) {
                    s += rows ? m[a * HC + b] : m[b * HC + a];
                }
                float d = s + e;
                for (int b = 0; b < HC; b++) {
                    int k = rows ? a * HC + b : b * HC + a;
                    m[k] = m[k] / d;
                }
            }
        }
        for (int k = 0; k < HC * HC; k++) {
            out[n * HC * HC + k] = m[k];
        }
    """
    return mx.fast.metal_kernel(
        name="deepseek_v41_sinkhorn",
        input_names=["comb", "eps"],
        output_names=["out"],
        source=source,
    )


_sinkhorn_kernel = _make_sinkhorn_kernel()


def _sinkhorn(comb: mx.array, *, iters: int, eps: mx.array) -> mx.array:
    """Normalize the columns, then the rows and columns iters - 1 more times."""
    if (
        _sinkhorn_kernel is None
        or mx.default_device() != mx.gpu
        or iters < 1
        or comb.size == 0
    ):
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
        for _ in range(iters - 1):
            comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
            comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
        return comb
    hc = comb.shape[-1]
    flat = comb.reshape(-1, hc, hc)
    (out,) = _sinkhorn_kernel(
        inputs=[flat, eps],
        template=[("HC", hc), ("ITERS", iters)],
        grid=(flat.shape[0], 1, 1),
        threadgroup=(min(flat.shape[0], 256), 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[mx.float32],
    )
    return mx.reshape(out, comb.shape)


class HyperConnection(nn.Module):
    """The pre / post / comb mixes of one sublayer, from its stream (mHC)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc = args.hc_mult
        self.iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.norm_eps = args.rms_norm_eps
        self._eps = mx.array([args.hc_eps], dtype=mx.float32)
        self._scale_ids = mx.array([0] * self.hc + [1] * self.hc + [2] * self.hc**2)
        mix = (2 + self.hc) * self.hc
        self.fn = mx.zeros((mix, self.hc * args.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def mixes(self, streams: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        flat = streams.reshape(*streams.shape[:2], -1).astype(mx.float32)
        flat = mx.fast.rms_norm(flat, None, self.norm_eps)
        mix = (flat @ self.fn.T) * self.scale[self._scale_ids] + self.base
        pre, post, comb = mx.split(mix, [self.hc, 2 * self.hc], axis=-1)
        pre = mx.sigmoid(pre) + self.hc_eps
        post = 2.0 * mx.sigmoid(post)
        comb = comb.reshape(*comb.shape[:-1], self.hc, self.hc)
        comb = mx.softmax(comb, axis=-1, precise=True) + self.hc_eps
        return pre, post, _sinkhorn(comb, iters=self.iters, eps=self._eps)


def _collapse(streams: mx.array, mix: mx.array) -> mx.array:
    out = mx.sum(mix[..., None] * streams.astype(mx.float32), axis=2)
    return out.astype(streams.dtype)


def _expand(
    x: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    mixed = mx.sum(comb[..., None] * residual[..., :, None, :], axis=2)
    out = post[..., None] * x[..., None, :] + mixed
    return out.astype(residual.dtype)


def _append(cached: mx.array | None, new: mx.array) -> mx.array:
    if cached is None:
        return new
    return mx.concatenate([cached, new], axis=1)


# ---------------------------------------------------------------------------------------
# Engram


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


@dataclass(frozen=True)
class EngramLayout:
    """The reference's EngramLayout: per engram layer, a prime-sized bucket range for every
    (n-gram size, head) pair, the primes drawn in order across all the engram layers and
    never reused."""

    layer_ids: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size - 2][head]
    max_ngram_size: int

    @classmethod
    def from_args(cls, args: ModelArgs) -> "EngramLayout":
        primes: list[tuple[tuple[int, ...], ...]] = []
        seen: set[int] = set()
        for _ in args.engram_layer_ids:
            per_ngram: list[tuple[int, ...]] = []
            for _ in range(args.engram_max_ngram_size - 1):
                sizes: list[int] = []
                current = args.engram_vocab_size - 1
                for _ in range(args.engram_n_heads):
                    current += 1
                    while not _is_prime(current) or current in seen:
                        current += 1
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            tuple(args.engram_layer_ids), tuple(primes), args.engram_max_ngram_size
        )


class _Backend(Protocol):
    def get_vocab_size(self, with_added_tokens: bool) -> int: ...
    def decode(self, ids: list[int], skip_special_tokens: bool) -> str: ...
    def id_to_token(self, token_id: int) -> str | None: ...


class _Normalizer(Protocol):
    def normalize_str(self, sequence: str) -> str: ...


def build_compressed_token_map(tokenizer_json: Path) -> tuple[list[int], int]:
    """The reference's build_compressed_token_map, on the raw `tokenizers` backend: token
    ids whose decoded text normalizes alike share one compressed id."""
    from tokenizers import Regex, Tokenizer, normalizers

    # a private-use character, so a token that is exactly one space survives Strip()
    sentinel = ""
    normalizer = cast(
        _Normalizer,
        normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), sentinel),
                normalizers.Strip(),
                normalizers.Replace(sentinel, " "),
            ]
        ),
    )
    backend = cast(_Backend, Tokenizer.from_file(str(tokenizer_json)))  # pyright: ignore[reportUnknownMemberType]
    size = backend.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = [0] * size
    for token_id in range(size):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            # a partial UTF-8 byte token: key it by its raw form
            key = backend.id_to_token(token_id) or text
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


_INT64_MAX = 2**63 - 1


class NgramHasher:
    """The hash ids of one engram layer (the reference's NgramHashState, for one layer).

    Position p hashes the compressed tokens p, p-1, ..., p-(max_ngram_size-1); positions
    before the sequence start read the pad token. The XOR of the multiplied ids after each
    look-back is the hash of that n-gram, reduced into its (n-gram size, head) prime's
    bucket range of the layer's table."""

    def __init__(
        self,
        layout: EngramLayout,
        layer_index: int,
        token_map: list[int],
        vocab_size: int,
        pad_token_id: int,
    ):
        layer_id = layout.layer_ids[layer_index]
        # odd, and small enough that token id x multiplier cannot overflow int64
        bound = max(1, (_INT64_MAX // vocab_size) // 2)
        rng = np.random.default_rng(10007 * layer_id)
        multipliers = rng.integers(
            0, bound, size=(layout.max_ngram_size,), dtype=np.int64
        )
        self.multipliers: IntArray = multipliers * 2 + 1
        self.primes: IntArray = np.array(layout.primes[layer_index], dtype=np.int64)
        flat = self.primes.reshape(-1)
        self.offsets: IntArray = np.concatenate([[0], np.cumsum(flat)[:-1]]).astype(
            np.int64
        )
        self.token_map: IntArray = np.array(token_map, dtype=np.int64)
        self.pad = token_map[pad_token_id]
        self.history_length = layout.max_ngram_size - 1
        self._device: tuple[mx.array, mx.array, mx.array, mx.array] | None = None

    def initial_history(self, batch: int) -> IntArray:
        return np.full((batch, self.history_length), self.pad, dtype=np.int64)

    def hash_host(
        self, tokens: IntArray, history: IntArray
    ) -> tuple[IntArray, IntArray]:
        """tokens [B, L] (token ids) and history [B, n-1] (compressed ids) ->
        ([B, L, (n-1) * heads] hash ids, the new history)."""
        n = self.history_length
        compressed: IntArray = self.token_map[tokens]
        window: IntArray = np.concatenate([history, compressed], axis=1)
        length = np.size(tokens, 1)
        # slices rather than scalars, so numpy's types stay arrays
        rolling: IntArray = window[:, n:] * self.multipliers[0:1]
        hashes: list[IntArray] = []
        for shift in range(1, n + 1):
            back: IntArray = window[:, n - shift : n - shift + length]
            rolling = np.bitwise_xor(
                rolling, back * self.multipliers[shift : shift + 1]
            )
            hashes.append(rolling[..., None] % self.primes[shift - 1 : shift])
        ids: IntArray = np.concatenate(hashes, axis=-1) + self.offsets
        return ids, window[:, -n:]

    def hash_device(
        self, tokens: mx.array, history: mx.array
    ) -> tuple[mx.array, mx.array]:
        """hash_host on the GPU (int64 arithmetic), so decode never waits for the host."""
        if self._device is None:
            self._device = (
                mx.array(self.token_map),
                mx.array(self.multipliers),
                mx.array(self.primes),
                mx.array(self.offsets),
            )
        token_map, multipliers, primes, offsets = self._device
        n = self.history_length
        window = mx.concatenate([history, token_map[tokens]], axis=1)
        length = tokens.shape[1]
        rolling = window[:, n:] * multipliers[0]
        hashes: list[mx.array] = []
        for shift in range(1, n + 1):
            back = window[:, n - shift : n - shift + length]
            rolling = mx.bitwise_xor(rolling, back * multipliers[shift])
            hashes.append(rolling[..., None] % primes[shift - 1])
        ids = mx.concatenate(hashes, axis=-1) + offsets
        return ids, window[:, -n:]


@dataclass(frozen=True)
class HostEngramTable:
    """An engram table left in its safetensors file: the rows a step needs are read
    through a memory map. The table takes no GPU memory."""

    weight: ByteArray  # [rows, dim] E4M3 values
    scale: ByteArray  # [rows, dim / 32] E8M0 scales

    @classmethod
    def open(
        cls, model_path: Path, weight_key: str, scale_key: str
    ) -> "HostEngramTable":
        index_file = model_path / "model.safetensors.index.json"
        index = cast(dict[str, object], json.loads(index_file.read_text()))
        weight_map = cast(dict[str, str], index["weight_map"])
        maps: list[ByteArray] = []
        for key in (weight_key, scale_key):
            path = model_path / weight_map[key]
            with open(path, "rb") as fh:
                header_length = cast(int, struct.unpack("<Q", fh.read(8))[0])
                header = cast(
                    dict[str, dict[str, object]], json.loads(fh.read(header_length))
                )
            meta = header[key]
            start, end = cast(list[int], meta["data_offsets"])
            shape = tuple(cast(list[int], meta["shape"]))
            if end - start != math.prod(shape):
                raise ValueError(f"{key}: expected one byte per element, got {meta}")
            maps.append(
                np.memmap(
                    path,
                    dtype=np.uint8,
                    mode="r",
                    offset=8 + header_length + start,
                    shape=shape,
                )
            )
        return cls(maps[0], maps[1])

    def gather(self, ids: IntArray) -> tuple[mx.array, mx.array]:
        shape = [np.size(ids, axis) for axis in range(ids.ndim)]
        flat = ids.reshape(-1)
        rows = np.take(self.weight, flat, axis=0)
        scales = np.take(self.scale, flat, axis=0)
        return (
            mx.reshape(mx.array(rows), [*shape, -1]),
            mx.reshape(mx.array(scales), [*shape, -1]),
        )


def _e8m0_to_float(scale: mx.array) -> mx.array:
    """E8M0 bytes -> float32 powers of two, 2 ** (byte - 127)."""
    bits = scale.astype(mx.uint32)
    normal = (bits << 23).view(mx.float32)
    return mx.where(bits == 0, mx.array(2.0**-127, dtype=mx.float32), normal)


class EngramEmbedding(nn.Module):
    """The engram n-gram table: [rows, head_dim] E4M3 with an E8M0 scale per 32 values."""

    def __init__(self, rows: int, dim: int, on_device: bool):
        super().__init__()
        self.dim = dim
        self.on_device = on_device
        if on_device:
            self.weight = mx.zeros((rows, dim), dtype=mx.uint8)
            self.scale = mx.zeros((rows, dim // 32), dtype=mx.uint8)
        self._host: HostEngramTable | None = None

    @staticmethod
    def dequantize(rows: mx.array, scales: mx.array, dtype: mx.Dtype) -> mx.array:
        # E4M3 times a power of two is exact in bf16; dtype only matters downstream
        values = mx.from_fp8(rows, mx.float32).reshape(*rows.shape[:-1], -1, 32)
        values = values * _e8m0_to_float(scales)[..., None]
        return mx.reshape(values, rows.shape).astype(dtype)

    def lookup_device(self, ids: mx.array, dtype: mx.Dtype) -> mx.array:
        return self.dequantize(self.weight[ids], self.scale[ids], dtype)

    def lookup_host(self, ids: IntArray, dtype: mx.Dtype) -> mx.array:
        if self._host is None:
            raise RuntimeError(
                "The engram table is not attached: call Model.load_engram(model_path) "
                "after loading the weights"
            )
        rows, scales = self._host.gather(ids)
        return self.dequantize(rows, scales, dtype)


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual streams, gated per stream (the
    reference's Engram)."""

    def __init__(self, args: ModelArgs, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.dim = args.hidden_size
        self.hc = args.hc_mult
        self.eps = args.rms_norm_eps
        n_cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
        self.embed = EngramEmbedding(
            args.engram_num_embeddings[layer_index],
            args.engram_head_dim,
            on_device=args.engram_tables == "device",
        )
        self.wkv = nn.Linear(
            n_cols * args.engram_head_dim,
            args.hidden_size * (args.hc_mult + 1),
            bias=False,
        )
        self.q_weight = mx.ones((args.hc_mult, args.hidden_size))
        self.k_weight = mx.ones((args.hc_mult, args.hidden_size))
        self._hasher: NgramHasher | None = None

    def attach(self, hasher: NgramHasher, table: HostEngramTable | None) -> None:
        self._hasher = hasher
        if table is not None:
            self.embed._host = table  # pyright: ignore[reportPrivateUsage]

    def _rows(self, inputs: mx.array, cache: "LayerCache", dtype: mx.Dtype) -> mx.array:
        hasher = self._hasher
        if hasher is None:
            raise RuntimeError(
                "The engram hasher is not attached: call Model.load_engram(model_path) "
                "after loading the weights"
            )
        batch = inputs.shape[0]
        if self.embed.on_device:
            history = cache.ngram_history
            if history is None:
                history = mx.array(hasher.initial_history(batch))
            ids, cache.ngram_history = hasher.hash_device(inputs, history)
            return self.embed.lookup_device(ids, dtype)
        history_host = (
            hasher.initial_history(batch)
            if cache.ngram_history is None
            else np.array(cache.ngram_history, dtype=np.int64)
        )
        tokens = np.array(inputs, dtype=np.int64)
        ids_host, new_history = hasher.hash_host(tokens, history_host)
        cache.ngram_history = mx.array(new_history)
        return self.embed.lookup_host(ids_host, dtype)

    def __call__(
        self, streams: mx.array, inputs: mx.array, cache: "LayerCache"
    ) -> mx.array:
        batch, length = streams.shape[:2]
        rows = self._rows(inputs, cache, streams.dtype)  # [B, L, cols, head_dim]
        kv = self.wkv(rows.reshape(batch, length, -1))
        key = (
            kv[..., : self.hc * self.dim]
            .astype(mx.float32)
            .reshape(batch, length, self.hc, self.dim)
        )
        value = kv[..., self.hc * self.dim :].astype(mx.float32)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        h = streams.astype(mx.float32)
        rstd = mx.rsqrt(mx.mean(h * h, axis=-1) + self.eps) * mx.rsqrt(
            mx.mean(key * key, axis=-1) + self.eps
        )
        dot = mx.sum(h * weight * key, axis=-1) * rstd / math.sqrt(self.dim)
        # signed square root before the sigmoid
        root = mx.sqrt(mx.maximum(mx.abs(dot), 1e-6))
        gate = mx.sigmoid(mx.where(dot < 0, -root, root))
        return (h + gate[..., None] * value[..., None, :]).astype(streams.dtype)


# ---------------------------------------------------------------------------------------
# Attention


class LayerCache:
    """Cache for one layer: the sliding window, the closed compressed groups and their
    index keys, the rows of the group still filling up, and, for an engram layer, the
    n-gram history (the last compressed token ids)."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.offset = 0
        self.window: mx.array | None = None
        self.compress_kv: mx.array | None = None
        self.index_k: mx.array | None = None
        self.pending_kv: mx.array | None = None
        self.pending_gate: mx.array | None = None
        self.ngram_history: mx.array | None = None

    def update_window(self, kv: mx.array) -> mx.array:
        """The window with kv appended; keeps the last window_size rows."""
        window = _append(self.window, kv)
        self.window = window[:, -self.window_size :]
        self.offset += kv.shape[1]
        return window

    def _arrays(self) -> list[mx.array | None]:
        return [
            self.window,
            self.compress_kv,
            self.index_k,
            self.pending_kv,
            self.pending_gate,
            self.ngram_history,
        ]

    @property
    def state(self) -> list[mx.array]:
        return [a for a in self._arrays() if a is not None]

    def is_trimmable(self) -> bool:
        # Evicted window rows and closed compressed groups cannot be restored.
        return False

    def empty(self) -> bool:
        return self.offset == 0

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self._arrays() if a is not None)

    def prefix_cache_snapshot(self) -> Self:
        """An independent copy (exo's prefix cache). The arrays are immutable, so the copy
        shares them; they are evaluated so the snapshot holds no lazy graph."""
        mx.eval(*self.state)
        copy = type(self)(self.window_size)
        copy.offset = self.offset
        copy.window = self.window
        copy.compress_kv = self.compress_kv
        copy.index_k = self.index_k
        copy.pending_kv = self.pending_kv
        copy.pending_gate = self.pending_gate
        copy.ngram_history = self.ngram_history
        return copy


class Compressor(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.ratio = int(args.compress_ratios[layer_id])
        self.kv_proj = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        if self.ratio > 1:
            self.gate_proj = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(args.head_dim, eps=args.rms_norm_eps)

    def pool(
        self, x: mx.array, start_pos: int, cache: LayerCache
    ) -> tuple[mx.array | None, int]:
        """Pool each full group of ratio tokens into one latent. Returns the latents (None
        if no group closed) and the index of the first one."""
        if self.ratio == 1:
            return self.kv_norm(self.kv_proj(x)), start_pos
        h = x.astype(mx.float32)
        pending = 0 if cache.pending_kv is None else cache.pending_kv.shape[1]
        kv = _append(cache.pending_kv, self.kv_proj(h))
        gate = _append(cache.pending_gate, self.gate_proj(h))
        full = kv.shape[1] - kv.shape[1] % self.ratio
        cache.pending_kv = kv[:, full:]
        cache.pending_gate = gate[:, full:]
        # A group takes the position of its first token.
        first = (start_pos - pending) // self.ratio
        if full == 0:
            return None, first
        batch, _, dim = kv.shape
        kv = kv[:, :full].reshape(batch, -1, self.ratio, dim)
        gate = gate[:, :full].reshape(batch, -1, self.ratio, dim)
        latent = mx.sum(kv * mx.softmax(gate, axis=2, precise=True), axis=2)
        return self.kv_norm(latent.astype(x.dtype)), first


def _window_indices(*, previous: int, length: int, window: int) -> mx.array:
    """The window rows each query attends to, -1 for none. The window holds the previous
    rows, then the new ones."""
    width = min(window, previous + length)
    end = previous + mx.arange(length)
    start = mx.maximum(end - window + 1, 0)
    indices = start[:, None] + mx.arange(width)[None, :]
    return mx.where(indices > end[:, None], -1, indices)


def _gather_rows(values: mx.array, indices: mx.array) -> mx.array:
    batch, length, dim = values.shape
    safe = mx.maximum(indices, 0)
    base = (mx.arange(batch, dtype=mx.int32) * length).reshape(batch, 1, 1)
    flat = values.reshape(batch * length, dim)
    return flat[(safe + base).reshape(-1)].reshape(*indices.shape, dim)


def _sparse_attention(
    q: mx.array,
    kvs: list[mx.array],
    indices: list[mx.array],
    *,
    sinks: mx.array,
    scale: float,
) -> mx.array:
    """Attend each query to the rows its indices pick from each KV array (K = V); -1 picks
    nothing. q is [B, S, H, D], each KV array [B, K, D] and its indices [B, S, k]."""
    batch, length, heads, dim = q.shape
    rows = [_gather_rows(kv, idx) for kv, idx in zip(kvs, indices, strict=True)]
    keys = (
        mx.concatenate(rows, axis=2)
        .reshape(batch * length, 1, -1, dim)
        .astype(mx.float32)
    )
    mask = mx.concatenate(indices, axis=-1) >= 0
    # SDPA rounds the scores to the input dtype at this head size, so use fp32.
    out = mx.fast.scaled_dot_product_attention(
        q.reshape(batch * length, heads, 1, dim).astype(mx.float32),
        keys,
        keys,
        scale=scale,
        mask=mask.reshape(batch * length, 1, 1, -1),
        sinks=sinks.astype(mx.float32),
    )
    return out.reshape(batch, length, heads, dim).astype(q.dtype)


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.ratio = int(args.compress_ratios[layer_id])
        self.is_candidate_source = layer_id == args.candidate_source_layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.num_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.rope_dim = args.qk_rope_head_dim
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        if layer_id in args.kv_source_layer_ids:
            self.k_proj = nn.Linear(args.head_dim, args.index_head_dim, bias=False)
            self.k_norm = nn.RMSNorm(args.index_head_dim, eps=args.rms_norm_eps)

    def select(
        self,
        x: mx.array,
        q_residual: mx.array,
        *,
        index_k: mx.array | None,
        candidates: mx.array | None,
        rope: RoPE,
        start_pos: int,
    ) -> tuple[mx.array, mx.array | None]:
        """The top-k compressed entries of each query (-1 for none) and the candidate
        mask: the one this layer selects, or the given one."""
        batch, length, _ = x.shape
        if index_k is None or index_k.shape[1] == 0:
            return mx.zeros((batch, length, 0), dtype=mx.int32), candidates
        q = self.q_b_proj(q_residual).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        q = _apply_rope(q, rope, rope_dim=self.rope_dim, offset=start_pos)
        q = fake_quant_fp4(q, block_size=32)
        keys = index_k.astype(mx.float32)
        weights = self.weights_proj(x).astype(mx.float32) / math.sqrt(self.num_heads)
        scores = mx.matmul(q.astype(mx.float32), keys[:, None].swapaxes(-1, -2))
        scores = mx.maximum(scores, 0.0) / math.sqrt(self.head_dim)
        scores = mx.sum(scores * weights[..., None], axis=2)
        # A query sees the groups that end at or before it.
        lens = ((start_pos + 1 + mx.arange(length)) // self.ratio)[:, None]
        visible = mx.arange(index_k.shape[1])[None, :] < lens
        scores = mx.where(visible[None], scores, -mx.inf)

        if self.is_candidate_source and self.candidate_topk_blocks > 0:
            block = self.candidate_block_size
            pad = (-scores.shape[-1]) % block
            padded = mx.pad(scores, [(0, 0), (0, 0), (0, pad)], constant_values=-mx.inf)
            block_scores = padded.reshape(batch, length, -1, block).max(axis=-1)
            # Keep the newest block: it is only partly full, so an older full block can
            # outscore it.
            last = mx.where(lens > 0, (lens - 1) // block, -1)
            num_blocks = block_scores.shape[-1]
            block_scores = mx.where(mx.arange(num_blocks) == last, mx.inf, block_scores)
            k_blocks = min(self.candidate_topk_blocks, num_blocks)
            chosen = mx.argpartition(-block_scores, k_blocks - 1, axis=-1)[
                ..., :k_blocks
            ]
            # With fewer reachable blocks than k_blocks, the extra picks are -inf.
            reachable = mx.take_along_axis(block_scores, chosen, axis=-1) > -mx.inf
            candidates = mx.zeros(block_scores.shape, dtype=mx.bool_)
            candidates = mx.put_along_axis(candidates, chosen, reachable, axis=-1)
            candidates = mx.repeat(candidates, block, axis=-1)[..., : scores.shape[-1]]
        elif candidates is not None:
            scores = mx.where(candidates, scores, -mx.inf)

        k = min(self.index_topk, index_k.shape[1])
        chosen = mx.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(mx.int32)
        return mx.where(chosen < lens[None], chosen, -1), candidates


SharedState = dict[str, mx.array | None]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.o_groups = args.o_groups
        ratios = args.compress_ratios
        self.ratio = int(ratios[layer_id]) if layer_id < len(ratios) else 0
        self.is_kv_source = layer_id in args.kv_source_layer_ids
        self.is_index_source = layer_id in args.index_source_layer_ids
        self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_a_norm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, args.num_attention_heads * args.head_dim, bias=False
        )
        self.kv_proj = nn.Linear(args.hidden_size, args.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(args.head_dim, eps=args.rms_norm_eps)
        # wo_a is block diagonal: one projection per output group.
        self.o_a_proj = SwitchLinear(
            args.num_attention_heads * args.head_dim // args.o_groups,
            args.o_lora_rank,
            args.o_groups,
            bias=False,
        )
        self.o_b_proj = nn.Linear(
            args.o_groups * args.o_lora_rank, args.hidden_size, bias=False
        )
        self.sinks = mx.zeros((args.num_attention_heads,), dtype=mx.float32)
        self._group_ids = mx.arange(args.o_groups)[None]
        if self.is_kv_source:
            self.compressor = Compressor(args, layer_id)
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id)
        if self.ratio:
            # layers with compressed KV: compress_rope_theta with YaRN
            self.rope = RoPE(
                args.qk_rope_head_dim, args.compress_rope_theta, args.rope_scaling
            )
        else:
            # window-only layers: rope_theta, no YaRN
            self.rope = RoPE(args.qk_rope_head_dim, args.rope_theta, None)

    def __call__(
        self,
        x: mx.array,
        *,
        start_pos: int,
        window_idx: mx.array,
        cache: LayerCache,
        shared: SharedState,
    ) -> mx.array:
        batch, length, _ = x.shape
        q_residual = self.q_a_norm(self.q_a_proj(x))
        q = self.q_b_proj(q_residual).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        q = _apply_rope(q, self.rope, rope_dim=self.rope_dim, offset=start_pos)

        kv = self.kv_norm(self.kv_proj(x))
        kv = _apply_rope(kv, self.rope, rope_dim=self.rope_dim, offset=start_pos)
        kv = fake_quant_fp8(kv, 32)
        kvs = [cache.update_window(kv)]
        indices = [window_idx]
        if self.ratio:
            if self.is_kv_source:
                self._compress(x, start_pos=start_pos, cache=cache, shared=shared)
            if self.is_index_source:
                topk, candidates = self.indexer.select(
                    x,
                    q_residual,
                    index_k=shared.get("index_k"),
                    candidates=shared.get("candidates"),
                    rope=self.rope,
                    start_pos=start_pos,
                )
                shared["topk_idx"] = topk
                if self.indexer.is_candidate_source:
                    shared["candidates"] = candidates
            compressed = shared.get("compress_kv")
            topk_idx = shared.get("topk_idx")
            if (
                compressed is not None
                and topk_idx is not None
                and topk_idx.shape[-1] > 0
            ):
                kvs.append(compressed)
                indices.append(topk_idx)

        output = _sparse_attention(
            q, kvs, indices, sinks=self.sinks, scale=1.0 / math.sqrt(self.head_dim)
        )
        output = _apply_rope(
            output, self.rope, rope_dim=self.rope_dim, offset=start_pos, inverse=True
        )
        output = output.reshape(batch * length, self.o_groups, 1, -1)
        output = cast(mx.array, self.o_a_proj(output, self._group_ids))
        return self.o_b_proj(output.reshape(batch, length, -1))

    def _compress(
        self, x: mx.array, *, start_pos: int, cache: LayerCache, shared: SharedState
    ) -> None:
        """Add the new compressed KV and index keys to the cache, and share them."""
        latent, first = self.compressor.pool(x, start_pos, cache)
        if latent is not None:
            kv = _apply_rope(
                latent,
                self.rope,
                rope_dim=self.rope_dim,
                offset=first,
                scale=self.ratio,
            )
            kv = fake_quant_fp4(kv, block_size=16, e4m3_scale=True)
            cache.compress_kv = _append(cache.compress_kv, kv)
            if self.is_index_source:
                k = self.indexer.k_norm(self.indexer.k_proj(latent))
                k = _apply_rope(
                    k, self.rope, rope_dim=self.rope_dim, offset=first, scale=self.ratio
                )
                k = fake_quant_fp4(k, block_size=32)
                cache.index_k = _append(cache.index_k, k)
        shared["compress_kv"] = cache.compress_kv
        if self.is_index_source:
            shared["index_k"] = cache.index_k


# ---------------------------------------------------------------------------------------
# MoE


class DeepseekV41SwiGLU(nn.Module):
    """DeepSeek-V4.1's clamped SwiGLU activation."""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, up: mx.array, gate: mx.array) -> mx.array:
        dtype = up.dtype
        gate = gate.astype(mx.float32)
        up = up.astype(mx.float32)
        if self.limit > 0:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        return (nn.silu(gate) * up).astype(dtype)


class SharedMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size, args.moe_intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            args.hidden_size, args.moe_intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            args.moe_intermediate_size, args.hidden_size, bias=False
        )
        self.activation = DeepseekV41SwiGLU(args.swiglu_limit)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.activation(self.up_proj(x), self.gate_proj(x))
        return self.down_proj(hidden.astype(x.dtype))


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.weight = mx.zeros(
            (args.n_routed_experts, args.hidden_size), dtype=mx.float32
        )
        self.e_score_correction_bias = mx.zeros(
            (args.n_routed_experts,), dtype=mx.float32
        )
        self.top_k = args.num_experts_per_tok
        self.score_func = args.scoring_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.routed_scaling_factor

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        logits = logits / self.gate_temp
        if self.score_func == "sigmoid":
            scores = mx.sigmoid(logits)
        elif self.score_func == "softmax":
            scores = mx.softmax(logits, axis=-1, precise=True)
        else:
            scores = mx.sqrt(nn.softplus(logits))
        # the bias picks the experts; the weights come from the unbiased scores
        biased = scores + self.e_score_correction_bias
        chosen = mx.argpartition(-biased, self.top_k - 1, axis=-1)[..., : self.top_k]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        return chosen, weights * self.route_scale


class Experts(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=DeepseekV41SwiGLU(args.swiglu_limit),
        )

    def __call__(self, x: mx.array, indices: mx.array, weights: mx.array) -> mx.array:
        routed = self.switch_mlp(x, indices)
        return mx.sum(
            routed.astype(mx.float32) * weights[..., None].astype(mx.float32), axis=1
        )


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = Router(args)
        self.experts = Experts(args)
        self.shared_experts = SharedMLP(args)
        self.dim = args.hidden_size
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        indices, weights = self.gate.route(flat)
        routed = self.experts(flat, indices, weights)
        out = routed + self.shared_experts(flat).astype(mx.float32)
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return mx.reshape(out, shape).astype(x.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_idx = layer_id
        self.self_attn = Attention(args, layer_id)
        self.mlp = MoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)
        self.engram: Engram | None = (
            Engram(args, args.engram_layer_ids.index(layer_id))
            if layer_id in args.engram_layer_ids
            else None
        )

    def forward(
        self,
        streams: mx.array,
        pre_mix: mx.array,
        *,
        inputs: mx.array,
        start_pos: int,
        window_idx: mx.array,
        cache: LayerCache,
        shared: SharedState,
    ) -> tuple[mx.array, mx.array]:
        if self.engram is not None:
            streams = self.engram(streams, inputs, cache)
        # The mix a sublayer computes is used by the next one.
        attn_pre, attn_post, attn_comb = self.attn_hc.mixes(streams)
        h = self.input_layernorm(_collapse(streams, pre_mix))
        h = self.self_attn(
            h, start_pos=start_pos, window_idx=window_idx, cache=cache, shared=shared
        )
        streams = _expand(h, streams, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.mixes(streams)
        h = self.mlp(self.post_attention_layernorm(_collapse(streams, attn_pre)))
        streams = _expand(h, streams, ffn_post, ffn_comb)
        return streams, ffn_pre


# ---------------------------------------------------------------------------------------
# State shared between pipeline stages


def _shared_reads(args: ModelArgs, layer: int) -> dict[str, int]:
    """The shared state a layer reads, as {key: the layer that writes it}."""
    ratios = args.compress_ratios
    if layer >= len(ratios) or not ratios[layer]:
        return {}
    kv, index = set(args.kv_source_layer_ids), set(args.index_source_layer_ids)

    def last(match: Callable[[int], bool]) -> int | None:
        return max((j for j in range(layer + 1) if match(j)), default=None)

    reads = {"compress_kv": last(lambda j: j in kv and bool(ratios[j]))}
    if layer in index:
        reads["index_k"] = last(lambda j: j in kv and j in index)
        source = args.candidate_source_layer_id
        if 0 <= source < layer and args.candidate_topk_blocks > 0:
            reads["candidates"] = source
    else:
        reads["topk_idx"] = last(lambda j: j in index)
    return {key: source for key, source in reads.items() if source is not None}


def _crossing(args: ModelArgs, boundary: int) -> dict[str, int]:
    """The shared state that layers from `boundary` on read from layers before it."""
    crossing: dict[str, int] = {}
    for layer in range(boundary, args.num_hidden_layers):
        for key, source in _shared_reads(args, layer).items():
            if source < boundary:
                crossing[key] = source
    return crossing


def _entries(args: ModelArgs, source: int | None, position: int) -> int:
    return position // args.compress_ratios[source] if source is not None else 0


def _index_keys(args: ModelArgs, layer: int, position: int) -> int:
    return _entries(args, _shared_reads(args, layer).get("index_k"), position)


_CACHE_FIELDS = ("compress_kv", "index_k")
Layout = list[tuple[str, tuple[int, ...], mx.Dtype]]


def _layout(
    args: ModelArgs,
    crossing: dict[str, int],
    batch: int,
    length: int,
    start_pos: int,
    dtype: mx.Dtype,
) -> Layout:
    """The parts of the message at a stage boundary, as (key, shape, dtype). Compressed
    KV and index keys take the streams' dtype: the next stage cannot know the dtype the
    stage before makes them in."""
    end = start_pos + length
    layout: Layout = [
        ("streams", (batch, length, args.hc_mult, args.hidden_size), dtype),
        ("pre_mix", (batch, length, args.hc_mult), mx.float32),
    ]
    for key, source in crossing.items():
        if key in _CACHE_FIELDS:
            # Only the new entries: the next stage keeps the earlier ones.
            new = _entries(args, source, end) - _entries(args, source, start_pos)
            width = args.head_dim if key == "compress_kv" else args.index_head_dim
            layout.append((key, (batch, new, width), dtype))
        elif key == "candidates":
            # One flag for each candidate block.
            block = args.candidate_block_size
            blocks = (_index_keys(args, source, end) + block - 1) // block
            layout.append((key, (batch, length, blocks), mx.bool_))
        else:
            topk = min(args.index_topk, _index_keys(args, source, end))
            layout.append((key, (batch, length, topk), mx.int32))
    return layout


# A view at a byte offset that is not a multiple of its item size reads the wrong bytes,
# so each part of a message starts at a multiple of 8 bytes.
_ALIGN = 8


def _nbytes(shape: tuple[int, ...], dtype: mx.Dtype) -> int:
    size = math.prod(shape) * dtype.size
    return (size + _ALIGN - 1) // _ALIGN * _ALIGN


def _pack(parts: list[mx.array]) -> mx.array:
    """All parts in one byte array: two sends can run in a different order than the two
    receives on the other rank."""
    data: list[mx.array] = []
    for part in parts:
        flat = part.view(mx.uint8).reshape(-1)
        pad = _nbytes(part.shape, part.dtype) - flat.size
        data.append(mx.pad(flat, (0, pad)) if pad else flat)
    return mx.concatenate(data)


def _unpack(packed: mx.array, layout: Layout) -> dict[str, mx.array]:
    parts: dict[str, mx.array] = {}
    start = 0
    for key, shape, dtype in layout:
        data = packed[start : start + math.prod(shape) * dtype.size]
        parts[key] = data.reshape(*shape[:-1], shape[-1] * dtype.size).view(dtype)
        start += _nbytes(shape, dtype)
    return parts


@dataclass
class PipelineStage:
    """Where a rank sits in a pipeline: its layers [start, end) and its neighbours."""

    group: mx.distributed.Group
    start: int
    end: int
    previous_rank: int | None  # runs the layers before `start`
    next_rank: int | None  # runs the layers from `end`
    last_rank: int  # runs the final layer
    shared_in: dict[str, int] = field(default_factory=dict[str, int])
    shared_out: dict[str, int] = field(default_factory=dict[str, int])


class TextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers: list[DecoderLayer | None] = [
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args
        self._stage: PipelineStage | None = None
        # Set by Model.shard for tensor parallelism.
        self.tensor_group: mx.distributed.Group | None = None

    @property
    def pipeline_layers(self) -> list[DecoderLayer]:
        return [layer for layer in self.layers if layer is not None]

    def set_pipeline_stage(self, stage: PipelineStage) -> None:
        """Keep layers [stage.start, stage.end) only. The others become None, so the kept
        ones keep their indices for weight loading."""
        stage.shared_in = _crossing(self.args, stage.start)
        stage.shared_out = _crossing(self.args, stage.end)
        self._stage = stage
        self.layers = [
            layer if stage.start <= i < stage.end else None
            for i, layer in enumerate(self.layers)
        ]

    def make_cache(self) -> list[LayerCache]:
        return [LayerCache(self.args.sliding_window) for _ in self.pipeline_layers]

    def _receive(
        self,
        stage: PipelineStage,
        like: mx.array,
        start_pos: int,
        cache: LayerCache,
        shared: SharedState,
    ) -> tuple[mx.array, mx.array]:
        """Receive the streams and pre_mix, and put the shared state in `shared`."""
        assert stage.previous_rank is not None
        batch, length = like.shape[:2]
        layout = _layout(
            self.args, stage.shared_in, batch, length, start_pos, like.dtype
        )
        size = sum(_nbytes(shape, dtype) for _, shape, dtype in layout)
        packed = mx.distributed.recv(
            (size,), mx.uint8, stage.previous_rank, group=stage.group
        )
        # Evaluate the receive on its own: a rank waiting for the previous stage must not
        # wait inside a GPU command buffer.
        mx.eval(packed)
        parts = _unpack(packed, layout)
        streams = parts.pop("streams")
        pre_mix = parts.pop("pre_mix")
        for key, part in parts.items():
            if key in _CACHE_FIELDS:
                # The first layer never writes this field, so it keeps the entries of the
                # source in the stage before.
                if part.shape[1]:
                    current = cast(mx.array | None, getattr(cache, key))
                    setattr(cache, key, _append(current, part))
                shared[key] = cast(mx.array | None, getattr(cache, key))
            elif key == "candidates":
                keys = _index_keys(self.args, stage.shared_in[key], start_pos + length)
                mask = mx.repeat(part, self.args.candidate_block_size, axis=-1)
                shared[key] = mask[..., :keys] if keys else None
            else:
                shared[key] = part
        return streams, pre_mix

    def _send(
        self,
        stage: PipelineStage,
        streams: mx.array,
        pre_mix: mx.array,
        start_pos: int,
        shared: SharedState,
    ) -> mx.array:
        """Send the streams, pre_mix and the shared state the next stages read."""
        assert stage.next_rank is not None
        batch, length = streams.shape[:2]
        layout = _layout(
            self.args, stage.shared_out, batch, length, start_pos, streams.dtype
        )
        state: SharedState = {**shared, "streams": streams, "pre_mix": pre_mix}
        parts: list[mx.array] = []
        for key, shape, dtype in layout:
            part = state.get(key)
            if part is None:
                # Not written yet, so the layout has no entries either.
                part = mx.zeros(shape, dtype)
            elif key in _CACHE_FIELDS:
                # It holds all entries so far; send only the new ones.
                part = part[:, part.shape[1] - shape[1] :]
            elif key == "candidates":
                part = part[..., :: self.args.candidate_block_size]
            if tuple(part.shape) != shape or part.dtype != dtype:
                raise ValueError(
                    f"The next pipeline stage expects {key} as {dtype} {shape}, "
                    f"not {part.dtype} {part.shape}."
                )
            parts.append(part)
        packed = _pack(parts)
        mx.eval(packed)
        sent = mx.distributed.send(packed, stage.next_rank, group=stage.group)
        mx.eval(sent)
        return sent

    def __call__(
        self, inputs: mx.array, cache: list[LayerCache] | None = None
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        batch, length, _ = h.shape
        layers = self.pipeline_layers
        caches = cache if cache is not None else self.make_cache()
        start_pos = caches[0].offset
        # Every layer's window holds the same rows, so all use the same indices.
        window = caches[0].window
        previous = 0 if window is None else window.shape[1]
        window_idx = _window_indices(
            previous=previous, length=length, window=self.args.sliding_window
        )
        window_idx = mx.broadcast_to(window_idx, (batch, *window_idx.shape))

        streams = mx.broadcast_to(
            h[:, :, None, :], (batch, length, self.args.hc_mult, self.args.hidden_size)
        )
        pre_mix = mx.zeros((batch, length, self.args.hc_mult), dtype=mx.float32)
        pre_mix[..., 0] = 1.0
        shared: SharedState = {}
        stage = self._stage
        if stage is not None and stage.previous_rank is not None:
            streams, pre_mix = self._receive(
                stage, streams, start_pos, caches[0], shared
            )

        for layer, layer_cache in zip(layers, caches, strict=True):
            streams, pre_mix = layer.forward(
                streams,
                pre_mix,
                inputs=inputs,
                start_pos=start_pos,
                window_idx=window_idx,
                cache=layer_cache,
                shared=shared,
            )

        if self.tensor_group is not None and self.args.engram_tables == "host":
            # The host engram reads above (made while the graph is built) can take seconds
            # on one rank and not on another, e.g. when its page cache is cold. Let every
            # rank finish them before any starts the graph: otherwise its GPU sits in the
            # first all-reduce that long, and Metal times the command buffer out.
            mx.eval(
                mx.distributed.all_sum(
                    mx.array(1.0),
                    group=self.tensor_group,
                    stream=mx.default_stream(mx.Device(mx.cpu)),
                )
            )
        if stage is not None and stage.next_rank is not None:
            sent = self._send(stage, streams, pre_mix, start_pos, shared)
            streams, pre_mix = mx.depends([streams, pre_mix], sent)
            last_window = caches[-1].window
            if last_window is not None:
                caches[-1].window = mx.depends(last_window, sent)
        out = _collapse(streams, pre_mix)
        if stage is not None and stage.group.size() > 1:
            # Every rank needs the last stage's output.
            gathered = mx.distributed.all_gather(out, group=stage.group)
            out = gathered[stage.last_rank * batch : (stage.last_rank + 1) * batch]
        return self.norm(out)


# ---------------------------------------------------------------------------------------
# Checkpoint


_DROP_PREFIXES = ("vision.", "aligner.", "image_", "mtp.")
_EXPERT_KEY = r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)"
_EXPERT_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_RENAMES = (
    (".attn.", ".self_attn."),
    (".ffn.", ".mlp."),
    (".attn_norm.", ".input_layernorm."),
    (".ffn_norm.", ".post_attention_layernorm."),
    (".hc_attn_", ".attn_hc."),
    (".hc_ffn_", ".ffn_hc."),
    (".attn_sink", ".sinks"),
    (".wq_a.", ".q_a_proj."),
    (".wq_b.", ".q_b_proj."),
    (".wkv.", ".kv_proj."),
    (".wgate.", ".gate_proj."),
    (".wo_a.", ".o_a_proj."),
    (".wo_b.", ".o_b_proj."),
    (".q_norm.", ".q_a_norm."),
    (".compressor.norm.", ".compressor.kv_norm."),
    (".indexer.wk.", ".indexer.k_proj."),
    (".gate.bias", ".gate.e_score_correction_bias"),
    (".shared_experts.w1.", ".shared_experts.gate_proj."),
    (".shared_experts.w2.", ".shared_experts.down_proj."),
    (".shared_experts.w3.", ".shared_experts.up_proj."),
)


def _rename(key: str) -> str:
    """Map one DeepSeek checkpoint name to its MLX name."""
    if key == "head.weight":
        return "lm_head.weight"
    if key == "embed.weight":
        return "model.embed_tokens.weight"
    key = f"model.{key}"
    for old, new in _RENAMES:
        key = key.replace(old, new)
    # the engram's wkv was renamed with the attention's: restore it
    return key.replace(".engram.kv_proj.", ".engram.wkv.")


def _repack(weight: mx.array, scale: mx.array) -> tuple[mx.array, mx.array]:
    """An fp8 or fp4 weight and its ue8m0 scale in MLX's mxfp8 / mxfp4 layout."""
    if scale.dtype != mx.uint8:
        scale = (mx.log2(scale.astype(mx.float32)) + 127).astype(mx.uint8)
    if scale.shape[-2] != weight.shape[-2]:
        # fp8 keeps one scale per 32x32 tile, MLX one per output row.
        scale = mx.repeat(scale, 32, axis=-2)[..., : weight.shape[-2], :]
    return weight.view(mx.uint32), scale


def _shard_groups(_path: str, _weight: mx.array) -> int:
    """o_a_proj's weight is [groups, rank, in]: split it by group."""
    return 0


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        # float32, as the reference's ParallelHead: the logits come out in float32
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache: list[LayerCache] | None = None
    ) -> mx.array:
        out = self.model(inputs, cache)
        return out.astype(mx.float32) @ self.lm_head.weight.astype(mx.float32).T

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.pipeline_layers

    def make_cache(self) -> list[LayerCache]:
        return self.model.make_cache()

    def load_engram(self, model_path: Path) -> None:
        """Attach what the engram layers need besides their weights: the hashing (from the
        tokenizer's compressed token map) and, for host tables, the memory maps into the
        checkpoint."""
        engrams = [layer.engram for layer in self.layers if layer.engram is not None]
        if not engrams:
            return
        token_map, vocab = build_compressed_token_map(model_path / "tokenizer.json")
        if vocab != self.args.engram_compressed_vocab_size:
            raise ValueError(
                f"The tokenizer compresses to {vocab} ids where the config expects "
                f"{self.args.engram_compressed_vocab_size}: the engram hashes would not "
                "match the tables."
            )
        layout = EngramLayout.from_args(self.args)
        for engram in engrams:
            hasher = NgramHasher(
                layout,
                engram.layer_index,
                token_map,
                vocab,
                self.args.engram_pad_token_id,
            )
            table = None
            if not engram.embed.on_device:
                layer_id = self.args.engram_layer_ids[engram.layer_index]
                table = HostEngramTable.open(
                    model_path,
                    f"layers.{layer_id}.engram.embed.weight",
                    f"layers.{layer_id}.engram.embed.scale",
                )
            engram.attach(hasher, table)

    def shard(self, group: mx.distributed.Group) -> None:
        """Tensor parallelism: each rank keeps its whole output groups (and their heads),
        and a slice of every expert's intermediate dimension. The compressor, indexer,
        hyper-connections and engram are replicated."""
        size, rank = group.size(), group.rank()
        heads, groups = self.args.num_attention_heads, self.args.o_groups
        if heads % size or groups % size:
            raise ValueError(
                f"Tensor parallelism over {size} ranks needs {size} to divide the "
                f"{heads} attention heads and the {groups} output groups."
            )
        self.model.tensor_group = group
        for layer in self.layers:
            attn = layer.self_attn
            attn.q_b_proj = shard_linear(attn.q_b_proj, "all-to-sharded", group=group)
            # An output group reads only its own heads, so a rank keeps whole groups.
            shard_inplace(attn.o_a_proj, _shard_groups, group=group)
            attn.o_b_proj = shard_linear(attn.o_b_proj, "sharded-to-all", group=group)
            attn.num_heads //= size
            attn.o_groups //= size
            attn._group_ids = mx.arange(attn.o_groups)[None]  # pyright: ignore[reportPrivateUsage]
            attn.sinks = attn.sinks[rank * attn.num_heads : (rank + 1) * attn.num_heads]
            layer.mlp.sharding_group = group
            switch_mlp = layer.mlp.experts.switch_mlp
            for projection in (switch_mlp.gate_proj, switch_mlp.up_proj):
                shard_inplace(projection, "all-to-sharded", group=group)
            shard_inplace(switch_mlp.down_proj, "sharded-to-all", group=group)
            shared = layer.mlp.shared_experts
            shard_inplace(shared.gate_proj, "all-to-sharded", group=group)
            shard_inplace(shared.up_proj, "all-to-sharded", group=group)
            shard_inplace(shared.down_proj, "sharded-to-all", group=group)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Rename the checkpoint's keys and keep its fp8 / fp4 weights packed (as mxfp8 /
        mxfp4, which they are bit for bit). Drops what the text model does not run (vision,
        MTP, the image routing bias), layers past num_hidden_layers, and, for host engram
        tables, the tables (they are read from the files)."""
        if any(key.startswith("model.") for key in weights):
            return weights
        n_layers = self.args.num_hidden_layers
        host_tables = self.args.engram_tables == "host"

        def keep(key: str) -> bool:
            if key.startswith(_DROP_PREFIXES) or key.endswith(".gate.bias_vl"):
                return False
            match = re.match(r"layers\.(\d+)\.", key)
            if match is not None and int(match.group(1)) >= n_layers:
                return False
            return not (host_tables and ".engram.embed." in key)

        text = {key: value for key, value in weights.items() if keep(key)}
        clean: dict[str, mx.array] = {}
        experts: dict[tuple[int, str], dict[int, dict[str, mx.array]]] = {}
        for key, value in text.items():
            expert = re.fullmatch(_EXPERT_KEY, key)
            if expert is not None:
                layer, index, projection, kind = expert.groups()
                rows = experts.setdefault((int(layer), projection), {})
                rows.setdefault(int(index), {})[kind] = value
                continue
            if ".engram.embed." in key:
                # E4M3 rows and E8M0 scales (uint8), as stored
                clean[f"model.{key}"] = value
                continue
            if (
                key.endswith(".scale")
                and f"{key.removesuffix('.scale')}.weight" in text
            ):
                continue
            scale = (
                text.get(f"{key.removesuffix('.weight')}.scale")
                if key.endswith(".weight")
                else None
            )
            name = _rename(key)
            if scale is not None:
                value, scale = _repack(value, scale)
            if name.endswith(".o_a_proj.weight"):
                # wo_a is block diagonal: split it back into its groups.
                value = value.reshape(self.args.o_groups, -1, value.shape[-1])
                if scale is not None:
                    scale = scale.reshape(self.args.o_groups, -1, scale.shape[-1])
            if name == "lm_head.weight":
                value = value.astype(mx.float32)
            clean[name] = value
            if scale is not None:
                clean[f"{name.removesuffix('weight')}scales"] = scale

        n_experts = self.args.n_routed_experts
        for (layer, projection), rows in experts.items():
            if len(rows) != n_experts:
                raise ValueError(
                    f"Layer {layer} has {len(rows)} {projection} expert weights, but the "
                    f"config declares {n_experts} experts."
                )
            parts = [rows[index] for index in sorted(rows)]
            name = f"model.layers.{layer}.mlp.experts.switch_mlp.{_EXPERT_PROJ[projection]}"
            value = mx.stack([part["weight"] for part in parts])
            if "scale" in parts[0]:
                scales = mx.stack([part["scale"] for part in parts])
                value, scales = _repack(value, scales)
                clean[f"{name}.scales"] = scales
            clean[f"{name}.weight"] = value
        return clean

    @property
    def cast_predicate(self) -> Callable[[str], bool]:
        # The reference keeps these in float32 (or as stored bytes).
        keep = (
            "attn_hc.",
            "ffn_hc.",
            "sinks",
            "e_score_correction_bias",
            "compressor.",
            "lm_head",
            "engram.embed",
        )

        def predicate(key: str) -> bool:
            return not any(part in key for part in keep)

        return predicate
