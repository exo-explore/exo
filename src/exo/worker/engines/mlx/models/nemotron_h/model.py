"""Nemotron-H (`nemotron_h`) model classes: mlx-lm's, computing in float32 where NVIDIA's
reference implementation (the `modeling_nemotron_h.py` shipped with the checkpoints) does.

1. The MoE router logits. The reference (and Hugging Face transformers) computes them as a
   float32 matmul, `F.linear(hidden_states.float(), weight.float())`. mlx-lm computes
   `x @ W.T` in the weight's dtype, bf16 in most conversions, and only the sigmoid in
   float32, so expert choices among near-tied scores flip.
2. The Mamba-2 mixer between `in_proj` and `out_proj`. The reference's causal conv1d (+SiLU)
   accumulates in float32 and rounds once; the selective scan, the D skip connection and the
   gated RMSNorm run in float32; only `out_proj`'s input is rounded back to the model's
   dtype. mlx-lm rounds to bf16 after the conv, after the SiLU and after the scan.

Everything else is mlx-lm's model unchanged.
"""

from typing import Optional, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.nemotron_h import Model as MlxLmModel
from mlx_lm.models.nemotron_h import (
    ModelArgs,
    MoEGate,
    NemotronHMamba2Mixer,
    NemotronHMoE,
    group_expert_select,
)

__all__ = [
    "Float32Conv1d",
    "Float32Mamba2Mixer",
    "Float32RouterGate",
    "Model",
    "ModelArgs",
]


class Float32RouterGate(MoEGate):
    """mlx-lm's Nemotron-H MoE gate with the router logits computed in float32."""

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = x.astype(mx.float32) @ self.weight.T.astype(mx.float32)
        return group_expert_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


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
    `out_proj`.
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


class Model(MlxLmModel):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__(args)
        for layer in self.layers:
            if isinstance(layer.mixer, NemotronHMoE):
                # Same parameter names (gate.weight, gate.e_score_correction_bias).
                layer.mixer.gate = Float32RouterGate(args)
            elif isinstance(layer.mixer, NemotronHMamba2Mixer):
                # Same parameter names (in_proj, conv1d, dt_bias, A_log, D, norm, out_proj).
                layer.mixer = Float32Mamba2Mixer(args)
