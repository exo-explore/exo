"""Nemotron-H (`nemotron_h`) model classes: mlx-lm's, computing in float32 where NVIDIA's
reference implementation (the `modeling_nemotron_h.py` shipped with the checkpoints) does.

1. The MoE router logits. The reference (and Hugging Face transformers) computes them as a
   float32 matmul, `F.linear(hidden_states.float(), weight.float())`. mlx-lm computes
   `x @ W.T` in the weight's dtype, bf16 in most conversions, and only the sigmoid in
   float32, so expert choices among near-tied scores flip. The router weight is kept in
   float32 from loading on (`Model.sanitize`), which is exact.
2. The Mamba-2 mixer between `in_proj` and `out_proj`. The reference's causal conv1d (+SiLU)
   accumulates in float32 and rounds once; the selective scan, the D skip connection and the
   gated RMSNorm run in float32; only `out_proj`'s input is rounded back to the model's
   dtype. mlx-lm rounds to bf16 after the conv, after the SiLU and after the scan.

On Metal these run as fused kernels (kernels.py) with the same float32 math: the mixer in
three dispatches (conv, scan, norm) for any number of tokens and the expert selection in
one. Two more changes cut the number of GPU dispatches of a decode step:

- a shared expert that is a whole number of routed experts wide is folded into them
  (`fold_shared_experts`), so all of a layer's expert weights are read in two gathered
  matmuls;
- each boundary between two blocks is one kernel (`Backbone`): the residual add, with the
  MoE's weighted sum of experts, and the next block's RMSNorm. Every RMSNorm of the model
  uses this kernel (`RMSNorm`), so blocks called whole (at a pipeline shard's edges) give
  the same bits.

Where a kernel does not apply (no Metal, masked or variable-length batches, grouped
routing) the MLX-op path below runs instead. Everything else is mlx-lm's model unchanged.
"""

from collections.abc import Callable
from typing import Any, Optional, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.nemotron_h import Model as MlxLmModel
from mlx_lm.models.nemotron_h import (
    ModelArgs,
    MoEGate,
    NemotronHBlock,
    NemotronHMamba2Mixer,
    NemotronHModel,
    NemotronHMoE,
    group_expert_select,
)
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

from exo.worker.engines.mlx.models.nemotron_h import kernels

__all__ = [
    "Backbone",
    "Float32Conv1d",
    "Float32Mamba2Mixer",
    "Float32MoE",
    "Float32RouterGate",
    "Model",
    "ModelArgs",
    "MoEBlock",
    "RMSNorm",
    "fold_shared_experts",
]


class Float32RouterGate(MoEGate):
    """mlx-lm's Nemotron-H MoE gate with the router logits computed in float32.

    With `always_on` > 0 (a shared expert folded into the routed experts, see
    `fold_shared_experts`), experts n_routed_experts, ... follow the top-k with weight 1.
    """

    always_on: int = 0

    def __call__(
        self, x: mx.array, x32: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        """`x32`, if given, is `x` in float32 (exactly)."""
        logits = (x.astype(mx.float32) if x32 is None else x32) @ self.weight.astype(
            mx.float32
        ).T
        if kernels.metal_available() and (self.n_group or 1) == 1:
            return kernels.select_experts(
                logits,
                self.e_score_correction_bias,
                self.top_k,
                self.routed_scaling_factor,
                bool(self.norm_topk_prob),
                self.always_on,
            )
        inds, scores = group_expert_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )
        return kernels.with_always_on_experts(
            inds, scores, self.n_routed_experts, self.always_on
        )


_LHS_INDICES: dict[tuple[int, ...], mx.array] = {}


def _lhs_indices(shape: tuple[int, ...]) -> mx.array:
    """gather_qmm's default `lhs_indices` (arange over the input's batch), made once per
    shape rather than by an arange kernel at every call."""
    if shape not in _LHS_INDICES:
        size = 1
        for n in shape:
            size *= n
        indices = mx.reshape(mx.arange(size, dtype=mx.uint32), shape)
        mx.eval(indices)
        _LHS_INDICES[shape] = indices
    return _LHS_INDICES[shape]


class Float32MoE(NemotronHMoE):
    """mlx-lm's Nemotron-H MoE with the float32 router, and its output combined in one
    kernel: the routed experts' weighted sum (float32, rounded once), the shared expert
    and, given `residual`, the block's residual connection, each rounded as mlx-lm does.
    Once `fold_shared_experts` has run, the shared expert is part of `switch_mlp`."""

    def __init__(self, config: ModelArgs) -> None:
        super().__init__(config)
        self.gate = Float32RouterGate(config)

    @property
    def router(self) -> Float32RouterGate:
        return cast(Float32RouterGate, self.gate)

    def expert_outputs(
        self, x: mx.array, x32: mx.array | None = None
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        """The routed experts' outputs (..., K, D), their weights (..., K) and the
        shared expert's output (unless folded). Not for latent MoEs."""
        inds, scores = self.router(x, x32)
        shared = (
            cast(nn.Module, self["shared_experts"])(x)
            if "shared_experts" in self
            else None
        )
        return self._experts(x, inds), scores, shared

    def _experts(self, x: mx.array, inds: mx.array) -> mx.array:
        """switch_mlp(x, inds); for a few tokens, its two gathered matmuls with constant
        `lhs_indices` (the same matmuls, without two arange kernels)."""
        fc1 = self.switch_mlp.fc1
        fc2 = self.switch_mlp.fc2
        if (
            inds.size >= 64
            or not isinstance(fc1, QuantizedSwitchLinear)
            or not isinstance(fc2, QuantizedSwitchLinear)
            or "bias" in fc1
            or "bias" in fc2
            or getattr(fc1, "global_scale", None) is not None
            or getattr(fc2, "global_scale", None) is not None
        ):
            return self.switch_mlp(x, inds)
        lead = inds.shape[:-1]
        h = mx.gather_qmm(
            mx.expand_dims(x, (-2, -3)),
            fc1.weight,
            fc1.scales,
            cast(mx.array | None, fc1.get("biases")),
            lhs_indices=_lhs_indices((*lead, 1)),
            rhs_indices=inds,
            transpose=True,
            group_size=fc1.group_size,
            bits=fc1.bits,
            mode=fc1.mode,
        )
        h = self.switch_mlp.activation(h)
        y = mx.gather_qmm(
            h,
            fc2.weight,
            fc2.scales,
            cast(mx.array | None, fc2.get("biases")),
            lhs_indices=_lhs_indices(inds.shape),
            rhs_indices=inds,
            transpose=True,
            group_size=fc2.group_size,
            bits=fc2.bits,
            mode=fc2.mode,
        )
        return y.squeeze(-2)

    def __call__(self, x: mx.array, residual: mx.array | None = None) -> mx.array:
        latent = self.moe_latent_size is not None
        if not latent:
            y, scores, shared = self.expert_outputs(x)
        else:
            inds, scores = self.gate(x)
            shared = (
                cast(nn.Module, self["shared_experts"])(x)
                if "shared_experts" in self
                else None
            )
            y = self.switch_mlp(self.fc1_latent_proj(x), inds)
        if kernels.metal_available():
            if not latent:
                return kernels.combine_experts(y, scores, shared, residual)
            routed = kernels.combine_experts(y, scores, None, None)
        else:
            routed = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        out = self.fc2_latent_proj(routed) if latent else routed
        if shared is not None:
            out = out + shared
        return out if residual is None else residual + out


def fold_shared_experts(moe: Float32MoE) -> None:
    """Make the shared expert part of the routed experts' stacks, in place: a shared
    expert of m times the routed width becomes m extra experts (its hidden units in m
    slices, exact since the activation is elementwise) that every token uses with weight
    1. Decode then reads all its expert weights in two gathered matmuls instead of four
    matmuls. Only when the shared expert is stored like the routed ones (same
    quantization, no biases) and there is no latent projection."""
    if moe.moe_latent_size is not None or "shared_experts" not in moe:
        return
    shared = cast(nn.Module, moe["shared_experts"])
    up = cast(nn.Module, shared["up_proj"])
    down = cast(nn.Module, shared["down_proj"])
    fc1 = moe.switch_mlp.fc1
    fc2 = moe.switch_mlp.fc2
    if any(
        "bias" in m or getattr(m, "global_scale", None) is not None
        for m in (up, down, fc1, fc2)
    ):
        return
    if isinstance(up, nn.QuantizedLinear) and isinstance(fc1, QuantizedSwitchLinear):
        if not isinstance(down, nn.QuantizedLinear) or not isinstance(
            fc2, QuantizedSwitchLinear
        ):
            return
        layouts = {(m.bits, m.group_size, m.mode) for m in (up, down, fc1, fc2)}
        if len(layouts) != 1:
            return
        names = ["weight", "scales"] + (["biases"] if "biases" in fc1 else [])
    elif type(up) is nn.Linear and type(fc1) is SwitchLinear:
        if type(down) is not nn.Linear or type(fc2) is not SwitchLinear:
            return
        names = ["weight"]
    else:
        return
    up_params = {n: cast(mx.array, up[n]) for n in names}
    down_params = {n: cast(mx.array, down[n]) for n in names}
    fc1_params = {n: cast(mx.array, fc1[n]) for n in names}
    fc2_params = {n: cast(mx.array, fc2[n]) for n in names}
    hidden = fc1_params["weight"].shape[1]
    copies, remainder = divmod(up_params["weight"].shape[0], hidden)
    if remainder or any(
        up_params[n].dtype != fc1_params[n].dtype
        or up_params[n].shape[-1] != fc1_params[n].shape[-1]
        or down_params[n].shape[-1] != copies * fc2_params[n].shape[-1]
        or down_params[n].shape[0] != fc2_params[n].shape[1]
        for n in names
    ):
        return
    for n in names:
        fc1[n] = mx.concatenate(
            [fc1_params[n], up_params[n].reshape(copies, hidden, -1)]
        )
        rows, columns = fc2_params[n].shape[1:]
        fc2[n] = mx.concatenate(
            [
                fc2_params[n],
                down_params[n].reshape(rows, copies, columns).transpose(1, 0, 2),
            ]
        )
    del moe["shared_experts"]
    moe.router.always_on = copies


class RMSNorm(nn.RMSNorm):
    """mlx's RMSNorm computed by kernels.residual_norm on Metal: the kernel the model's
    fused block boundaries use, so that a norm computed on its own (e.g. at a pipeline
    shard's edge) rounds exactly as a fused one."""

    eps: float

    def __call__(self, x: mx.array) -> mx.array:
        if not kernels.metal_available():
            return mx.fast.rms_norm(x, self.weight, self.eps)
        return kernels.residual_norm(x, self.weight, self.eps)[1]


def _call_block_mixer(
    mixer: object,
    block_type: str,
    x: mx.array,
    mask: object = None,
    cache: object = None,
) -> mx.array:
    """A block's mixer: Mamba ("M") and attention ("*") mixers take a mask and a cache
    entry, MoE and MLP mixers neither."""
    if block_type in ("M", "*"):
        return cast(Callable[..., mx.array], mixer)(x, mask=mask, cache=cache)
    return cast(Callable[[mx.array], mx.array], mixer)(x)


class MoEBlock(NemotronHBlock):
    """A MoE block whose residual connection is fused into the MoE's output kernel (when
    the mixer is not wrapped, e.g. by tensor parallelism's all-reduce)."""

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        mixer = cast(object, self.mixer)
        if isinstance(mixer, Float32MoE):
            return mixer(self.norm(x), residual=x)
        return x + _call_block_mixer(mixer, self.block_type, self.norm(x))


class Backbone(NemotronHModel):
    """mlx-lm's Nemotron-H backbone. On Metal, each boundary between two plain blocks is
    one kernel (kernels.residual_norm): the residual add of one block (with the MoE's
    expert combination) and the RMSNorm of the next, plus the float32 copy a MoE router
    reads. The values are those of the blocks called one by one, so a pipeline shard,
    whose edge layers are wrapped and called whole, computes the same bits."""

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        if not kernels.metal_available():
            return super().__call__(inputs, cache=cache)
        # Layers of a pipeline shard may be wrapped (and are then called whole).
        layers = cast(list[object], self.layers)
        hidden = self.embeddings(inputs)
        entries = (
            list(cast(list[object], cache))
            if cache is not None
            else [None] * len(layers)
        )
        attn_mask = create_attention_mask(hidden, entries[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden, entries[self.ssm_idx])

        normed: mx.array | None = None
        normed32: mx.array | None = None
        cache_counter = 0
        for i, layer in enumerate(layers):
            block_type = cast(str, getattr(layer, "block_type"))  # noqa: B009
            entry: object = None
            if block_type in ("M", "*"):
                entry = entries[cache_counter]
                cache_counter += 1
            mask = attn_mask if block_type == "*" else ssm_mask
            if not _plain_block(layer):
                hidden = cast(Callable[..., mx.array], layer)(
                    hidden, mask=mask, cache=entry
                )
                normed = normed32 = None
                continue
            block = cast(NemotronHBlock, layer)
            if normed is None:
                normed = block.norm(hidden)
            mixer = cast(object, block.mixer)
            scores: mx.array | None = None
            shared: mx.array | None = None
            if isinstance(mixer, Float32MoE) and mixer.moe_latent_size is None:
                y, scores, shared = mixer.expert_outputs(normed, normed32)
            else:
                y = _call_block_mixer(mixer, block_type, normed, mask, entry)
            following = layers[i + 1] if i + 1 < len(layers) else None
            if following is None:
                norm: RMSNorm | None = cast(RMSNorm, self.norm_f)
            elif _plain_block(following):
                norm = cast(RMSNorm, cast(NemotronHBlock, following).norm)
            else:
                norm = None
            if norm is None:
                if scores is not None:
                    hidden = kernels.combine_experts(y, scores, shared, hidden)
                else:
                    hidden = hidden + y
                normed = normed32 = None
                continue
            next_is_moe = following is not None and isinstance(
                cast(object, cast(NemotronHBlock, following).mixer), Float32MoE
            )
            hidden, normed, normed32 = kernels.residual_norm(
                hidden,
                norm.weight,
                norm.eps,
                y=y,
                scores=scores,
                shared=shared,
                want_float32=next_is_moe,
            )
            if following is None:
                return normed
        return self.norm_f(hidden)


def _plain_block(layer: object) -> bool:
    """A block of this model, not wrapped (e.g. by a pipeline shard's send or receive)."""
    return isinstance(layer, NemotronHBlock) and isinstance(layer.norm, RMSNorm)


class Float32Conv1d(nn.Conv1d):
    """A depthwise causal conv computed in float32; returns float32."""

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv1d(
            x.astype(mx.float32), self.weight.astype(mx.float32), groups=self.groups
        )
        bias = cast(mx.array | None, self.get("bias"))
        if bias is not None:
            y = y + bias.astype(mx.float32)
        return y


class Float32Mamba2Mixer(NemotronHMamba2Mixer):
    """mlx-lm's Mamba-2 mixer with the conv, the scan and the gated norm in float32.

    With a float32 conv, mlx-lm's SiLU, scan (`D` is cast to the scan input's dtype) and
    gated norm (given a float32 gate) all run in float32; the result is rounded once, for
    `out_proj`. On Metal the same math runs as three kernels (`_fused`).
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__(args)
        self.conv1d = Float32Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            padding=0,
            groups=self.conv_dim,
            bias=args.use_conv_bias,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array],
        cache: Optional[ArraysCache] = None,
    ) -> mx.array:
        projected = self.in_proj(hidden_states)
        if (
            mask is None
            and (cache is None or cache.lengths is None)
            and self.ssm_state_size % 32 == 0
            and kernels.metal_available()
        ):
            return self.out_proj(self._fused(projected, cache, hidden_states.dtype))
        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )
        conv_output = self._conv(conv_input, cache, mask)
        hidden_states_ssm, b, c = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )
        y = self._ssm(hidden_states_ssm, b, c, dt, cache, mask)
        if cache:
            cache.advance(y.shape[1])
        y = self.norm(y, gate.astype(mx.float32))
        return self.out_proj(y.astype(hidden_states.dtype))

    def _fused(
        self, projected: mx.array, cache: Optional[ArraysCache], dtype: mx.Dtype
    ) -> mx.array:
        """conv + SiLU, the scan with dt, D and the SiLU gate, then the grouped RMSNorm:
        the input of `out_proj`, rounded once to `dtype`."""
        batch, length, _ = projected.shape
        conv_state = cast(mx.array | None, cache[0]) if cache is not None else None
        ssm_state = cast(mx.array | None, cache[1]) if cache is not None else None
        if conv_state is None:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim), dtype=projected.dtype
            )
        if ssm_state is None:
            ssm_state = mx.zeros(
                (batch, self.num_heads, self.head_dim, self.ssm_state_size),
                dtype=mx.float32,
            )
        xbc, conv_state = kernels.causal_conv_silu(
            projected,
            self.intermediate_size,
            self.conv_dim,
            conv_state,
            self.conv1d.weight,
            cast(mx.array | None, self.conv1d.get("bias")),
        )
        time_step_limit = self.time_step_limit
        y, ssm_state = kernels.selective_scan(
            xbc,
            projected,
            self.intermediate_size + self.conv_dim,
            self.dt_bias,
            self.A_log,
            self.D,
            (float(time_step_limit[0]), float(time_step_limit[1])),
            ssm_state,
            self.num_heads,
            self.head_dim,
            self.n_groups,
            self.ssm_state_size,
        )
        if cache is not None:
            cache[0] = conv_state
            cache[1] = ssm_state
            cache.advance(length)
        return kernels.grouped_rms_norm(
            y, self.norm.weight, self.norm.group_size, self.norm.eps, dtype
        )


class Model(MlxLmModel):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__(args)
        self.backbone = Backbone(args)
        layers = self.backbone.layers
        for i, layer in enumerate(layers):
            if isinstance(layer.mixer, NemotronHMoE):
                # Same parameter names (norm.weight, mixer.gate.weight, ...).
                block = MoEBlock(args, layer.block_type)
                block.mixer = Float32MoE(args)
                layers[i] = block
            elif isinstance(layer.mixer, NemotronHMamba2Mixer):
                # Same parameter names (in_proj, conv1d, dt_bias, A_log, D, norm, out_proj).
                layer.mixer = Float32Mamba2Mixer(args)
            layers[i].norm = RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.backbone.norm_f = RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def load_weights(
        self,
        file_or_weights: str | list[tuple[str, mx.array]],
        strict: bool = True,
    ) -> "Model":
        super().load_weights(file_or_weights, strict=strict)
        for layer in self.layers:
            if isinstance(layer.mixer, Float32MoE):
                fold_shared_experts(layer.mixer)
        return self

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        weights = super().sanitize(weights)
        # The router computes in float32 (exactly the reference's `weight.float()`):
        # convert its weight once instead of at every call.
        for name in weights:
            if name.endswith(".mixer.gate.weight"):
                weights[name] = cast(mx.array, weights[name]).astype(mx.float32)
        return weights
