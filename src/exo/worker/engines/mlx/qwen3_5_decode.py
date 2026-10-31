"""Faster single-token decode for Qwen3.5-architecture models (mlx-lm's ``qwen3_5``).

A decode step of these hybrid models is a long chain of small kernels, and on Apple GPUs each
kernel launch costs a few microseconds whatever its size. This module cuts the launches per
step after loading (and after tensor or pipeline sharding), without changing weights:

- Projections that read the same input run as one matmul: the Gated DeltaNet
  in_proj_qkv/z/b/a, the attention q/k/v and the MLP gate/up. The fused weights are the
  originals concatenated by rows; the original modules become views of their rows, so memory
  does not grow and prefill keeps running the original code bit for bit.
- A Gated DeltaNet decode step (conv1d + SiLU, q/k l2-normalisation, gate and beta, the
  recurrence and the gated RMSNorm) runs as one Metal kernel instead of about twenty ops.
  Normalisation, gate, beta and the recurrent state are float32 in the kernel, which is closer
  to the float32 reference than the op-by-op bfloat16 path.
- The attention q/k RMSNorm + RoPE (and the v copy) run as one kernel that reproduces MLX's
  rms_norm and rope arithmetic exactly; the output gate is one compiled kernel.
- Each residual add runs fused with the RMSNorm that follows it, reproducing MLX's rms_norm
  reduction exactly.

Every fused path falls back to the original code when its preconditions do not hold (prefill,
no Metal, other shapes). Set EXO_QWEN3_5_FUSED_DECODE=0 to disable all of it.
"""

import math
import os
from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, Any, Protocol, cast, final

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import AllToShardedLinear, QuantizedAllToShardedLinear
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3_5 import DecoderLayer as Qwen3_5DecoderLayer
from mlx_lm.models.qwen3_5 import GatedDeltaNet as Qwen3_5GatedDeltaNet
from mlx_lm.models.qwen3_5 import Model as Qwen3_5Model
from mlx_lm.models.qwen3_next import Qwen3NextAttention, Qwen3NextMLP

from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from mlx.core.fast import (  # pyright: ignore[reportMissingModuleSource]
        MetalKernel,
    )

_DISABLE_ENV = "EXO_QWEN3_5_FUSED_DECODE"
_FUSED_PROJECTION = "_exo_fused_projection"


# Typed views of the mlx / mlx-lm objects used here (the stubs leave these attributes out).


class _QuantizedParts(Protocol):
    weight: mx.array
    scales: mx.array
    group_size: int
    bits: int
    mode: str


class _DenseParts(Protocol):
    weight: mx.array


class _Norm(Protocol):
    weight: mx.array
    eps: float


class _Rope(Protocol):
    dims: int
    traditional: bool
    base: float
    scale: float


class _AttentionParts(Protocol):
    q_norm: _Norm
    k_norm: _Norm
    rope: nn.Module


class _RecurrentCache(Protocol):
    lengths: mx.array | None

    def __getitem__(self, index: int) -> mx.array | None: ...
    def __setitem__(self, index: int, value: mx.array) -> None: ...
    def advance(self, n: int) -> None: ...


class _AttentionCache(Protocol):
    offset: int | mx.array

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]: ...


def _make_kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
    header: str = "",
) -> "MetalKernel":
    return mx.fast.metal_kernel(
        name=name,
        input_names=input_names,
        output_names=output_names,
        source=source,
        header=header,
    )


def _metal_decode_available() -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu


# ----------------------------------------------------------------------------------------
# Fused projections


_QUANTIZED_LINEARS = (nn.QuantizedLinear, QuantizedAllToShardedLinear)
_DENSE_LINEARS = (nn.Linear, AllToShardedLinear)


def fusable_linears(modules: Sequence[nn.Module]) -> bool:
    """Bias-free linears that read the same input and store their weights the same way:
    all quantized alike, or all dense. Tensor-parallel all-to-sharded linears qualify (their
    forward is a plain matmul of the rank's rows)."""
    first = modules[0]
    if any("bias" in m for m in modules):
        return False
    if all(isinstance(m, _QUANTIZED_LINEARS) for m in modules):
        parts = [cast(_QuantizedParts, cast(object, m)) for m in modules]
        return all(
            (p.group_size, p.bits, p.mode, "biases" in m)
            == (parts[0].group_size, parts[0].bits, parts[0].mode, "biases" in first)
            for p, m in zip(parts, modules, strict=True)
        )
    return all(isinstance(m, _DENSE_LINEARS) for m in modules) and all(
        cast(_DenseParts, cast(object, m)).weight.dtype
        == cast(_DenseParts, cast(object, first)).weight.dtype
        for m in modules
    )


def fuse_linears(modules: Sequence[nn.Module]) -> nn.Module:
    """One linear computing all of ``modules`` (concatenated along the output), and every
    original turned into a view of its rows of the fused weights (no extra memory).

    Each output row of the matmul depends only on its own weight row, so the fused outputs of
    a single-token matmul are bitwise those of the separate modules."""
    if not fusable_linears(modules):
        raise ValueError("these linears can't be fused")
    first = modules[0]
    names = (
        ["weight", "scales", *(["biases"] if "biases" in first else [])]
        if isinstance(first, _QUANTIZED_LINEARS)
        else ["weight"]
    )
    fused_arrays = {
        name: mx.concatenate([cast(mx.array, m[name]) for m in modules], axis=0)
        for name in names
    }
    mx.eval(list(fused_arrays.values()))
    if isinstance(first, _QUANTIZED_LINEARS):
        quantized = cast(_QuantizedParts, cast(object, first))
        fused: nn.Module = nn.QuantizedLinear(
            input_dims=quantized.group_size,
            output_dims=1,
            bias=False,
            group_size=quantized.group_size,
            bits=quantized.bits,
            mode=quantized.mode,
        )
        if "biases" not in fused_arrays and "biases" in fused:
            del fused["biases"]
    else:
        fused = nn.Linear(input_dims=1, output_dims=1, bias=False)
    fused.update(dict(fused_arrays))
    fused.freeze()
    start = 0
    for module in modules:
        rows = cast(mx.array, module["weight"]).shape[0]
        module.update(
            {name: array[start : start + rows] for name, array in fused_arrays.items()}
        )
        start += rows
    mx.eval([m.parameters() for m in modules])
    return fused


def _attach(module: nn.Module, fused: nn.Module) -> None:
    # An instance attribute outside the module tree: parameters() and the sharding code never
    # see the fused weights next to the views of them.
    object.__setattr__(module, _FUSED_PROJECTION, fused)


def _fused(module: nn.Module) -> nn.Module | None:
    return cast(
        nn.Module | None,
        cast(dict[str, object], module.__dict__).get(_FUSED_PROJECTION),
    )


# ----------------------------------------------------------------------------------------
# Gated DeltaNet decode step

_GDN_HEADER = """
template <typename T, typename W>
inline float exo_gdn_conv4(
    const device T* conv_state, const device T* x, const device W* w, int c, int C, bool valid) {
  float acc = static_cast<float>(conv_state[c]) * static_cast<float>(w[4 * c]);
  acc += static_cast<float>(conv_state[C + c]) * static_cast<float>(w[4 * c + 1]);
  acc += static_cast<float>(conv_state[2 * C + c]) * static_cast<float>(w[4 * c + 2]);
  acc += (valid ? static_cast<float>(x[c]) : 0.0f) * static_cast<float>(w[4 * c + 3]);
  return acc;
}

// The conv1d output and the SiLU output are materialised in the model dtype, as on the op path.
template <typename T>
inline float exo_gdn_silu_rounded(float conv) {
  float x = static_cast<float>(static_cast<T>(conv));
  return static_cast<float>(static_cast<T>(x / (1.0f + metal::exp(-x))));
}
"""

# One threadgroup per (sequence, value head): 32 SIMD-groups, each owning Dv / 32 value rows,
# lanes owning Dk / 32 state elements of a row. The q/k/v prologue is computed once into
# threadgroup memory; the gated RMSNorm reduces the head's Dv outputs in threadgroup memory.
_GDN_SOURCE = """
    constexpr int Kd = Hk * Dk;
    constexpr int Vd = Hv * Dv;
    constexpr int C = 2 * Kd + Vd;
    constexpr int P = 2 * Kd + 2 * Vd + 2 * Hv;
    constexpr int n_per_t = Dk / 32;
    constexpr int NSG = 32;
    constexpr int NT = 32 * NSG;
    constexpr int RPS = Dv / NSG;

    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n = threadgroup_position_in_grid.z;
    const uint b_idx = n / Hv;
    const uint hv = n % Hv;
    const uint hk = hv / (Hv / Hk);
    const bool valid = HAS_MASK ? static_cast<bool>(mask[b_idx]) : true;

    auto x = proj + b_idx * P;
    auto cs = conv_state + b_idx * 3 * C;
    auto cso = conv_state_out + b_idx * 3 * C;

    threadgroup float qs[Dk];
    threadgroup float ks[Dk];
    threadgroup float vs[Dv];
    threadgroup float ys[Dv];

    const int cq0 = hk * Dk;
    const int ck0 = Kd + hk * Dk;
    const int cv0 = 2 * Kd + hv * Dv;
    const bool writes_qk = (hv % (Hv / Hk)) == 0;
    for (int e = tid; e < Dk; e += NT) {
      qs[e] = exo_gdn_silu_rounded<InT>(exo_gdn_conv4(cs, x, conv_w, cq0 + e, C, valid));
      ks[e] = exo_gdn_silu_rounded<InT>(exo_gdn_conv4(cs, x, conv_w, ck0 + e, C, valid));
      if (writes_qk) {
        int cq = cq0 + e;
        int ck = ck0 + e;
        cso[cq] = cs[C + cq];
        cso[C + cq] = cs[2 * C + cq];
        cso[2 * C + cq] = valid ? x[cq] : static_cast<InT>(0);
        cso[ck] = cs[C + ck];
        cso[C + ck] = cs[2 * C + ck];
        cso[2 * C + ck] = valid ? x[ck] : static_cast<InT>(0);
      }
    }
    for (int r = tid; r < Dv; r += NT) {
      int cv = cv0 + r;
      vs[r] = exo_gdn_silu_rounded<InT>(exo_gdn_conv4(cs, x, conv_w, cv, C, valid));
      cso[cv] = cs[C + cv];
      cso[C + cv] = cs[2 * C + cv];
      cso[2 * C + cv] = valid ? x[cv] : static_cast<InT>(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // q/k l2-normalisation in float32 (q also takes the Dk^-1/2 readout scale)
    float q[n_per_t];
    float k[n_per_t];
    float qq = 0.0f;
    float kk = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      q[i] = qs[n_per_t * lane + i];
      k[i] = ks[n_per_t * lane + i];
      qq += q[i] * q[i];
      kk += k[i] * k[i];
    }
    qq = simd_sum(qq);
    kk = simd_sum(kk);
    float q_scale = metal::rsqrt(qq + 1e-6f) * metal::rsqrt(static_cast<float>(Dk));
    float k_scale = metal::rsqrt(kk + 1e-6f);
    for (int i = 0; i < n_per_t; ++i) {
      q[i] *= q_scale;
      k[i] *= k_scale;
    }

    // gate and beta in float32
    float b_val = static_cast<float>(x[2 * Kd + 2 * Vd + hv]);
    float a_val = static_cast<float>(x[2 * Kd + 2 * Vd + Hv + hv]);
    float x_g = a_val + static_cast<float>(dt_bias[hv]);
    float softplus = (x_g > 20.0f) ? x_g : metal::log(1.0f + metal::exp(x_g));
    float g = metal::exp(-metal::exp(static_cast<float>(A_log[hv])) * softplus);
    float beta = 1.0f / (1.0f + metal::exp(-b_val));

    float s[RPS][n_per_t];
    for (int j = 0; j < RPS; ++j) {
      int dv = sg + j * NSG;
      auto i_state = state_in + (n * Dv + dv) * Dk + n_per_t * lane;
      for (int i = 0; i < n_per_t; ++i) {
        s[j][i] = static_cast<float>(i_state[i]);
      }
    }
    for (int j = 0; j < RPS; ++j) {
      int dv = sg + j * NSG;
      auto o_state = state_out + (n * Dv + dv) * Dk + n_per_t * lane;
      if (!valid) {
        // padding: the state passes through and the output is zero, as on the op path
        for (int i = 0; i < n_per_t; ++i) {
          o_state[i] = static_cast<StT>(s[j][i]);
        }
        if (lane == 0) {
          ys[dv] = 0.0f;
        }
        continue;
      }
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        s[j][i] = s[j][i] * g;
        kv_mem += s[j][i] * k[i];
      }
      kv_mem = simd_sum(kv_mem);
      float delta = (vs[dv] - kv_mem) * beta;
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        s[j][i] = s[j][i] + k[i] * delta;
        out += s[j][i] * q[i];
        o_state[i] = static_cast<StT>(s[j][i]);
      }
      out = simd_sum(out);
      if (lane == 0) {
        ys[dv] = static_cast<float>(static_cast<InT>(out));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // gated RMSNorm, as MLX's rms_norm (Dv / 32 reads per lane, one SIMD-group) then SiLU(z)
    constexpr int v_per_t = Dv / 32;
    float ss = 0.0f;
    for (int i = 0; i < v_per_t; ++i) {
      float yv = ys[v_per_t * lane + i];
      ss += yv * yv;
    }
    ss = simd_sum(ss);
    float inv_rms = metal::precise::rsqrt(ss / Dv + eps);
    auto z = x + 2 * Kd + Vd + hv * Dv;
    for (int e = tid; e < Dv; e += NT) {
      InT normed = static_cast<InT>(norm_w[e] * static_cast<InT>(ys[e] * inv_rms));
      float zf = static_cast<float>(z[e]);
      float gate = zf / (1.0f + metal::exp(-zf));
      y[n * Dv + e] = static_cast<InT>(gate * static_cast<float>(normed));
    }
"""


@cache
def _gdn_kernel() -> "MetalKernel":
    return _make_kernel(
        name="exo_qwen3_5_gated_delta_decode",
        input_names=[
            "proj",
            "conv_state",
            "conv_w",
            "A_log",
            "dt_bias",
            "state_in",
            "norm_w",
            "eps",
            "mask",
        ],
        output_names=["y", "state_out", "conv_state_out"],
        source=_GDN_SOURCE,
        header=_GDN_HEADER,
    )


def gated_delta_decode(
    gdn: Qwen3_5GatedDeltaNet,
    projections: mx.array,
    conv_state: mx.array,
    state: mx.array,
    mask: mx.array | None,
) -> tuple[mx.array, mx.array, mx.array]:
    """One decode token of a Gated DeltaNet mixer from its fused input projections
    (B, 1, qkv | z | b | a). Returns the gated-normed output (B, 1, value_dim) for out_proj, and
    the new conv and recurrent states."""
    batch = projections.shape[0]
    hk, hv = gdn.num_k_heads, gdn.num_v_heads
    dk, dv = gdn.head_k_dim, gdn.head_v_dim
    dtype = projections.dtype
    mask_input = (
        mask.reshape(batch) if mask is not None else mx.zeros((1,), dtype=mx.bool_)
    )
    y, state_out, conv_state_out = _gdn_kernel()(
        inputs=[
            projections,
            conv_state,
            gdn.conv1d.weight,
            gdn.A_log,
            gdn.dt_bias,
            state,
            gdn.norm.weight,
            gdn.norm.eps,
            mask_input,
        ],
        template=[
            ("InT", dtype),
            ("StT", state.dtype),
            ("Dk", dk),
            ("Dv", dv),
            ("Hk", hk),
            ("Hv", hv),
            ("HAS_MASK", mask is not None),
        ],
        grid=(1024, 1, batch * hv),
        threadgroup=(1024, 1, 1),
        output_shapes=[(batch, 1, hv * dv), state.shape, conv_state.shape],
        output_dtypes=[dtype, state.dtype, conv_state.dtype],
    )
    return y, conv_state_out, state_out


def _gdn_supported(gdn: Qwen3_5GatedDeltaNet) -> bool:
    return (
        gdn.head_k_dim % 32 == 0
        and gdn.head_v_dim % 32 == 0
        and gdn.head_v_dim <= 1024
        and gdn.num_v_heads % gdn.num_k_heads == 0
        and gdn.conv_kernel_size == 4
        and tuple(gdn.conv1d.weight.shape) == (gdn.conv_dim, 4, 1)
    )


@final
class FusedDecodeGatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        batch, length, _ = inputs.shape
        fused = _fused(self)
        recurrent = cast(_RecurrentCache | None, cache)
        if (
            length != 1
            or fused is None
            or recurrent is None
            or recurrent.lengths is not None
            or (mask is not None and tuple(mask.shape) != (batch, 1))
            or cast(dict[str, object], self.__dict__).get("sharding_group") is not None
            or not _metal_decode_available()
        ):
            return super().__call__(inputs, mask, cache)
        conv_state, state = recurrent[0], recurrent[1]
        if conv_state is None or state is None:
            return super().__call__(inputs, mask, cache)
        out, conv_state, state = gated_delta_decode(
            self, fused(inputs), conv_state, state, mask
        )
        recurrent[0] = conv_state
        recurrent[1] = state
        recurrent.advance(1)
        return self.out_proj(out)


# ----------------------------------------------------------------------------------------
# Attention decode prologue: q/k RMSNorm + RoPE and the v copy, from the fused projections

_ATTENTION_SOURCE = """
    constexpr int Nq = 2 * H * D;
    constexpr int P = Nq + 2 * Hkv * D;
    constexpr int N_READS = 4;
    constexpr int HALF = ROPE_DIMS / 2;
    const uint lid = thread_position_in_threadgroup.x;
    const uint j = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    auto x = proj + b * P;
    if (j >= H + Hkv) {
      int vh = j - H - Hkv;
      for (int i = 0; i < N_READS; i++) {
        int e = lid * N_READS + i;
        v_out[(b * Hkv + vh) * D + e] = x[Nq + Hkv * D + vh * D + e];
      }
      return;
    }
    const bool is_q = j < H;
    // the q projection holds [query | gate] per head
    auto src = is_q ? (x + j * 2 * D) : (x + Nq + (j - H) * D);
    auto w = is_q ? q_w : k_w;

    // MLX rms_single_row: D / 4 threads, 4 reads each
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    threadgroup T normed[D];
    float thread_x[N_READS];
    float acc = 0;
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = src[lid * N_READS + i];
      acc += thread_x[i] * thread_x[i];
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
      local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
      local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
      int e = lid * N_READS + i;
      normed[e] = w[e] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // MLX rope (not traditional) over the first ROPE_DIMS dims
    auto dst = is_q ? (q_out + (b * H + j) * D) : (k_out + (b * Hkv + (j - H)) * D);
    float L = scale * static_cast<float>(offset[OFFSET_PER_BATCH ? b : 0]);
    for (int i = 0; i < N_READS; i++) {
      int e = lid * N_READS + i;
      if (e < ROPE_DIMS) {
        int ii = e < HALF ? e : e - HALF;
        float d = static_cast<float>(ii) / static_cast<float>(HALF);
        float inv_freq = metal::exp2(-d * rope_base);
        float theta = L * inv_freq;
        float costheta = metal::fast::cos(theta);
        float sintheta = metal::fast::sin(theta);
        float x1 = static_cast<float>(normed[ii]);
        float x2 = static_cast<float>(normed[ii + HALF]);
        float r = e < HALF ? (x1 * costheta - x2 * sintheta) : (x1 * sintheta + x2 * costheta);
        dst[e] = static_cast<T>(r);
      } else {
        dst[e] = normed[e];
      }
    }
"""


@cache
def _attention_kernel() -> "MetalKernel":
    return _make_kernel(
        name="exo_qwen3_5_attention_decode_prologue",
        input_names=["proj", "q_w", "k_w", "eps", "offset", "scale", "rope_base"],
        output_names=["q_out", "k_out", "v_out"],
        source=_ATTENTION_SOURCE,
    )


@mx.compile
def _gated_output(output: mx.array, gate: mx.array) -> mx.array:
    return output * mx.sigmoid(gate)


def _attention_supported(attention: Qwen3NextAttention) -> bool:
    parts = cast(_AttentionParts, cast(object, attention))
    if not isinstance(parts.rope, nn.RoPE):
        return False
    rope = cast(_Rope, cast(object, parts.rope))
    return (
        not rope.traditional
        and attention.head_dim % 128 == 0
        and attention.head_dim <= 4096
        and rope.dims % 2 == 0
        and rope.dims <= attention.head_dim
        and parts.q_norm.eps == parts.k_norm.eps
    )


def attention_decode_prologue(
    attention: Qwen3NextAttention, projections: mx.array, offset: mx.array | int
) -> tuple[mx.array, mx.array, mx.array]:
    batch = projections.shape[0]
    heads, kv_heads, head_dim = (
        attention.num_attention_heads,
        attention.num_key_value_heads,
        attention.head_dim,
    )
    parts = cast(_AttentionParts, cast(object, attention))
    rope = cast(_Rope, cast(object, parts.rope))
    if isinstance(offset, mx.array):
        offsets = offset.astype(mx.int32).reshape(-1)
    else:
        offsets = mx.array([offset], dtype=mx.int32)
    per_batch = offsets.size > 1
    threads = 32 * ((head_dim // 4 + 31) // 32)
    queries, keys, values = _attention_kernel()(
        inputs=[
            projections,
            parts.q_norm.weight,
            parts.k_norm.weight,
            parts.q_norm.eps,
            offsets,
            float(rope.scale),
            math.log2(rope.base),
        ],
        template=[
            ("T", projections.dtype),
            ("H", heads),
            ("Hkv", kv_heads),
            ("D", head_dim),
            ("ROPE_DIMS", rope.dims),
            ("OFFSET_PER_BATCH", per_batch),
        ],
        grid=(threads, heads + 2 * kv_heads, batch),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (batch, heads, 1, head_dim),
            (batch, kv_heads, 1, head_dim),
            (batch, kv_heads, 1, head_dim),
        ],
        output_dtypes=[projections.dtype] * 3,
    )
    return queries, keys, values


@final
class FusedDecodeAttention(Qwen3NextAttention):
    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        fused = _fused(self)
        kv_cache = cast(_AttentionCache | None, cache)
        if (
            length != 1
            or fused is None
            or kv_cache is None
            or not _metal_decode_available()
        ):
            return super().__call__(x, mask, cache)
        projections = fused(x)
        queries, keys, values = attention_decode_prologue(
            self, projections, kv_cache.offset
        )
        keys, values = kv_cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        heads, head_dim = self.num_attention_heads, self.head_dim
        gate = projections[..., : 2 * heads * head_dim].reshape(
            batch, 1, heads, 2 * head_dim
        )[..., head_dim:]
        gated = _gated_output(output.reshape(batch, 1, heads, head_dim), gate)
        return self.o_proj(gated.reshape(batch, 1, heads * head_dim))


# ----------------------------------------------------------------------------------------
# MLP gate/up


@final
class FusedGateUpMLP(Qwen3NextMLP):
    def __call__(self, x: mx.array) -> mx.array:
        fused = _fused(self)
        # Single tokens only: prefill keeps the original matmuls (and their exact results).
        if fused is None or x.shape[-2] != 1 or not _metal_decode_available():
            return super().__call__(x)
        gate_up = fused(x)
        rows = self.gate_proj.weight.shape[0]
        return self.down_proj(swiglu(gate_up[..., :rows], gate_up[..., rows:]))


# ----------------------------------------------------------------------------------------
# Residual add + RMSNorm

# MLX's rms_looped (threadgroup 1024, 4 reads per thread) with the add in front: h and the
# normed output are bitwise those of `h = x + r; mx.fast.rms_norm(h, w, eps)`.
_ADD_RMS_NORM_SOURCE = """
    constexpr int N_READS = 4;
    const uint gid = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lsize = threads_per_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];

    auto xa = x + gid * size_t(D) + lid * N_READS;
    auto xr = r + gid * size_t(D) + lid * N_READS;
    auto hh = h + gid * size_t(D) + lid * N_READS;
    auto ww = w + lid * N_READS;
    float acc = 0;
    for (uint rr = 0; rr < D; rr += lsize * N_READS) {
      for (int i = 0; i < N_READS; i++) {
        if ((rr + lid * N_READS + i) < D) {
          T hv = xa[rr + i] + xr[rr + i];
          hh[rr + i] = hv;
          float xi = hv;
          acc += xi * xi;
        }
      }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
      local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
      local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto oo = out + gid * size_t(D) + lid * N_READS;
    for (uint rr = 0; rr < D; rr += lsize * N_READS) {
      for (int i = 0; i < N_READS; i++) {
        if ((rr + lid * N_READS + i) < D) {
          T hv = xa[rr + i] + xr[rr + i];
          oo[rr + i] = ww[i + rr] * static_cast<T>(hv * local_inv_mean[0]);
        }
      }
    }
"""

# MLX uses its looped rms_norm kernel above this row size, the one reproduced here.
_RMS_LOOPED_LIMIT = 4096


@cache
def _add_rms_norm_kernel() -> "MetalKernel":
    return _make_kernel(
        name="exo_add_rms_norm",
        input_names=["x", "r", "w", "eps"],
        output_names=["h", "out"],
        source=_ADD_RMS_NORM_SOURCE,
    )


def add_rms_norm(
    x: mx.array, residual: mx.array, weight: mx.array, eps: float
) -> tuple[mx.array, mx.array]:
    """``h = x + residual`` and ``rms_norm(h, weight, eps)``, in one kernel when exact."""
    size = x.shape[-1]
    if (
        size <= _RMS_LOOPED_LIMIT
        or x.dtype != residual.dtype
        or x.dtype != weight.dtype
        or x.shape != residual.shape
        or not _metal_decode_available()
    ):
        h = x + residual
        return h, mx.fast.rms_norm(h, weight, eps)
    rows = x.size // size
    h, out = _add_rms_norm_kernel()(
        inputs=[x, residual, weight, eps],
        template=[("T", x.dtype), ("D", size)],
        grid=(1024 * rows, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return h, out


_PRENORMED = "_exo_prenormed"
_NEXT_LAYER = "_exo_next_layer"


@final
class AddNormDecoderLayer(Qwen3_5DecoderLayer):
    """Runs both residual adds fused with the RMSNorm after them: the post-attention norm, and
    the next layer's input norm. The next layer takes the stashed normed input only when it is
    called with exactly the array this layer returned, so a pipeline wrapper that sends or
    receives in between falls back to its own norm."""

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        state = cast(dict[str, object], self.__dict__)
        stash = cast(tuple[mx.array, mx.array] | None, state.get(_PRENORMED))
        state[_PRENORMED] = None
        if stash is not None and stash[0] is x:
            normed_input = stash[1]
        else:
            normed_input = self.input_layernorm(x)
        if self.is_linear:
            mixed = self.linear_attn(normed_input, mask, cache)
        else:
            mixed = self.self_attn(normed_input, mask, cache)
        post_norm = cast(_Norm, cast(object, self.post_attention_layernorm))
        h, normed = add_rms_norm(x, mixed, post_norm.weight, post_norm.eps)
        mlp_out = self.mlp(normed)
        next_layer = cast(Qwen3_5DecoderLayer | None, state.get(_NEXT_LAYER))
        if next_layer is None:
            return h + mlp_out
        next_norm = cast(_Norm, cast(object, next_layer.input_layernorm))
        out, normed_next = add_rms_norm(h, mlp_out, next_norm.weight, next_norm.eps)
        cast(dict[str, object], next_layer.__dict__)[_PRENORMED] = (out, normed_next)
        return out


# ----------------------------------------------------------------------------------------
# Installing


def _unwrap(layer: nn.Module) -> nn.Module:
    # exo's pipeline wrappers keep the original layer
    inner = cast(dict[str, object], layer).get("_original_layer")
    return inner if isinstance(inner, nn.Module) else layer


def _swap_class(module: nn.Module, base: type, fused_class: type) -> bool:
    if type(module) is base:
        module.__class__ = fused_class
        return True
    return isinstance(module, fused_class)


def prepare_qwen3_5_decode(model: nn.Module) -> bool:
    """Install the fused decode paths on a loaded (and possibly sharded) Qwen3.5 model.
    Returns whether anything was installed."""
    if os.environ.get(_DISABLE_ENV, "1") == "0":
        logger.info(f"{_DISABLE_ENV}=0: fused Qwen3.5 decode disabled")
        return False
    # Only the dense model: the MoE variants share these classes but are not verified here.
    if type(model) is not Qwen3_5Model:
        return False
    layers = [
        cast(Qwen3_5DecoderLayer, _unwrap(layer))
        for layer in cast(list[nn.Module], model.layers)
    ]
    counts = {"gated_delta": 0, "attention": 0, "mlp": 0, "layers": 0}
    for layer in layers:
        if layer.is_linear:
            gdn = layer.linear_attn
            projections = [gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a]
            if (
                type(gdn) is Qwen3_5GatedDeltaNet
                and _gdn_supported(gdn)
                and fusable_linears(projections)
            ):
                _attach(gdn, fuse_linears(projections))
                counts["gated_delta"] += _swap_class(
                    gdn, Qwen3_5GatedDeltaNet, FusedDecodeGatedDeltaNet
                )
        else:
            attention = layer.self_attn
            projections = [attention.q_proj, attention.k_proj, attention.v_proj]
            if (
                type(attention) is Qwen3NextAttention
                and _attention_supported(attention)
                and fusable_linears(projections)
            ):
                _attach(attention, fuse_linears(projections))
                counts["attention"] += _swap_class(
                    attention, Qwen3NextAttention, FusedDecodeAttention
                )
        mlp = cast(object, layer.mlp)
        if type(mlp) is Qwen3NextMLP and fusable_linears([mlp.gate_proj, mlp.up_proj]):
            _attach(mlp, fuse_linears([mlp.gate_proj, mlp.up_proj]))
            counts["mlp"] += _swap_class(mlp, Qwen3NextMLP, FusedGateUpMLP)
    for index, layer in enumerate(layers):
        if _swap_class(layer, Qwen3_5DecoderLayer, AddNormDecoderLayer):
            counts["layers"] += 1
            object.__setattr__(layer, _PRENORMED, None)
            next_layer = layers[index + 1] if index + 1 < len(layers) else None
            object.__setattr__(layer, _NEXT_LAYER, next_layer)
    mx.clear_cache()
    logger.info(f"Fused Qwen3.5 decode installed: {counts}")
    return True
