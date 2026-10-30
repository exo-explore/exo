"""A faster forward pass for mlx-lm's Laguna, set up once the model is loaded and sharded.

The model, its weights and its arithmetic are mlx-lm's. ``fuse_for_inference`` (the
family's post_load step) only rearranges how they run:

- the q, k and v projections of each attention layer become one matmul (their weights
  concatenated), and the gate and up projections of the dense MLP, the shared expert and
  the routed experts likewise;
- ``kernels.qk_norm_rope`` normalises and rotates the queries and keys straight from that
  matmul's output;
- ``kernels.route`` does the router's sigmoid, bias, top-k and weights in one dispatch;
- ``kernels.combine_experts`` weights and sums the experts, adds the shared expert and the
  residual;
- ``kernels.add_rms_norm`` adds the attention output to the residual and normalises it;
- the attention gate (softplus) is one compiled elementwise kernel.

Each piece reproduces mlx-lm's results bit for bit (the tests check it), so a fused model
computes the same logits as an unfused one, on one device and sharded.

A layer whose weights don't allow a fusion (e.g. its projections are quantized
differently) keeps mlx-lm's code for that part.
"""

import re
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import activations
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.laguna import MLP, Attention, MoE, TransformerBlock
from mlx_lm.models.rope_utils import YarnRoPE
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from exo.worker.engines.mlx.models.laguna import kernels
from exo.worker.engines.mlx.sharding.blocks import CustomMlxLayer, ShardedMoE

_HEAD_DIM = 128
# Fused projections serve decode-sized inputs: up to this many rows (batch x tokens).
_MAX_FUSED_ROWS = 8

# mlx-lm's compiled silu(gate) * up.
_swiglu = cast(
    Callable[[mx.array, mx.array], mx.array],
    getattr(activations, "swiglu"),  # noqa: B009
)


def _quantization(module: nn.Module) -> tuple[int, int, str] | None:
    """(group_size, bits, mode) of a quantized linear layer, None for a plain one."""
    if "scales" not in module:
        return None
    return (
        cast(int, getattr(module, "group_size")),  # noqa: B009
        cast(int, getattr(module, "bits")),  # noqa: B009
        cast(str, getattr(module, "mode", "affine")),
    )


def _fusable(modules: list[nn.Module]) -> bool:
    first = modules[0]
    quantization = _quantization(first)
    for module in modules:
        if "bias" in module or getattr(module, "global_scale", None) is not None:
            return False
        if type(module) is not type(first) or _quantization(module) != quantization:
            return False
    return True


def _concatenate(arrays: list[mx.array], axis: int) -> mx.array:
    return arrays[0] if len(arrays) == 1 else mx.concatenate(arrays, axis=axis)


def _gpu_architecture() -> tuple[int, str]:
    """(generation, size letter) of the GPU, as MLX reads them (e.g. applegpu_g15d)."""
    architecture = str(mx.device_info().get("architecture", ""))
    match = re.fullmatch(r"applegpu_g(\d+)(\w)", architecture)
    if match is None:
        return 0, "?"
    return int(match.group(1)), match.group(2)


def _qmv_rows_limit(in_dims: int, out_dims: int) -> int:
    """MLX's get_qmv_batch_limit: below this many rows a transposed quantized matmul runs
    its vector kernel, which computes every output row the same way whatever the number of
    output rows; from it on, a tiled (split-K) kernel."""
    generation, size = _gpu_architecture()
    small = in_dims <= 2048 and out_dims <= 2048
    medium = in_dims <= 4096 and out_dims <= 4096
    if generation >= 17 and size != "d":
        return 33 if small else (25 if medium else 13)
    if generation >= 15 and size != "d":
        return 13 if small else (15 if medium else 13)
    if size == "d":
        return 32 if small else (18 if medium else 12)
    return 14 if small else (10 if medium else 6)


def _fused_rows(modules: list[nn.Module], fused_out_dims: int) -> int:
    """The most rows (batch x tokens) for which the fused matmul gives exactly what the
    separate ones give: as long as MLX runs all of them with its vector kernels (gemv,
    gemv_wide, qmv), whose results don't depend on the number of output rows. Larger
    inputs run tiled kernels, which split K by the output size; those use the separate
    weights (views into the fused ones)."""
    outs = [int(cast(mx.array, m["weight"]).shape[0]) for m in modules]
    quantization = _quantization(modules[0])
    if quantization is None:
        # One row runs gemv. Two to fifteen run gemv_wide (Apple GPU generation 15 on),
        # tiled by the number of rows alone unless the output is tiny or vocabulary-wide.
        generation, _ = _gpu_architecture()
        if generation >= 15 and all(64 < n < 65536 for n in [*outs, fused_out_dims]):
            return _MAX_FUSED_ROWS
        return 1
    packed_in_dims = int(cast(mx.array, modules[0]["weight"]).shape[-1])
    in_dims = packed_in_dims * 32 // quantization[1]
    limit = min(_qmv_rows_limit(in_dims, n) for n in [*outs, fused_out_dims])
    return min(_MAX_FUSED_ROWS, limit - 1)


def _stackable(experts: list[nn.Module], shared: list[nn.Module]) -> bool:
    """Whether the shared expert's (gate, up, down) can be stacked with the routed experts'
    as one more expert: the same quantization and the same shapes."""
    for stacked, single in zip(experts, shared, strict=True):
        if not isinstance(stacked, QuantizedSwitchLinear):
            return False
        if "bias" in single or getattr(single, "global_scale", None) is not None:
            return False
        if _quantization(stacked) != _quantization(single):
            return False
        for name in ("weight", "scales", "biases"):
            if (name in stacked) != (name in single):
                return False
            if name in stacked and tuple(
                cast(mx.array, stacked[name]).shape[1:]
            ) != tuple(cast(mx.array, single[name]).shape):
                return False
    return True


class FusedLinear(nn.Module):
    """Several linear layers on the same input as one matmul, split back afterwards. The
    output rows are the layers' rows one after the other, so each output is computed exactly
    as its own layer computes it."""

    def __init__(self, modules: list[nn.Module], axis: int = 0) -> None:
        super().__init__()
        quantization = _quantization(modules[0])
        self.quantization = quantization
        weights = [cast(mx.array, m["weight"]) for m in modules]
        self.weight = _concatenate(weights, axis)
        if quantization is not None:
            self.scales = _concatenate(
                [cast(mx.array, m["scales"]) for m in modules], axis
            )
            if "biases" in modules[0]:
                self.biases = _concatenate(
                    [cast(mx.array, m["biases"]) for m in modules], axis
                )
        # Output sizes, from the packed weights' rows (axis 0 for linear layers, axis 1
        # for stacked experts).
        self.splits = [int(w.shape[axis]) for w in weights]
        self.fused_rows = (
            _fused_rows(modules, sum(self.splits)) if axis == 0 else 1 << 30
        )

    def append_expert(self, other: "FusedLinear") -> None:
        """Stack ``other`` (one expert's weights, as a linear layer) after these stacked
        experts."""
        names = ["weight"] + (["scales"] if self.quantization is not None else [])
        names += ["biases"] if "biases" in self else []
        for name in names:
            stacked = cast(mx.array, self[name])
            extra = cast(mx.array, other[name])
            self[name] = mx.concatenate([stacked, extra[None]], axis=0)

    def view_expert(self, stacked: "FusedLinear", index: int) -> None:
        """Read the weights from expert ``index`` of ``stacked`` (views, no copy)."""
        for name in ("weight", "scales", "biases"):
            if name in self:
                self[name] = cast(mx.array, stacked[name])[index]

    def use_fused(self, x: mx.array) -> bool:
        return x.size // x.shape[-1] <= self.fused_rows

    def parts(self, x: mx.array) -> list[mx.array]:
        """Each layer's output on its own, from views of the fused weights: exactly the
        separate layers' matmuls."""
        outputs: list[mx.array] = []
        start = 0
        for size in self.splits:
            rows = slice(start, start + size)
            start += size
            if self.quantization is None:
                outputs.append(x @ self.weight[rows].T)
                continue
            group_size, bits, mode = self.quantization
            biases = cast(mx.array | None, self.get("biases"))
            outputs.append(
                mx.quantized_matmul(
                    x,
                    self.weight[rows],
                    scales=self.scales[rows],
                    biases=None if biases is None else biases[rows],
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                )
            )
        return outputs

    def __call__(self, x: mx.array) -> mx.array:
        if self.quantization is None:
            return x @ self.weight.T
        group_size, bits, mode = self.quantization
        return mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=cast(mx.array | None, self.get("biases")),
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    def gather(
        self,
        x: mx.array,
        lhs_indices: mx.array | None,
        rhs_indices: mx.array,
        sorted_indices: bool,
        n_experts: int | None = None,
    ) -> mx.array:
        """As mlx-lm's SwitchLinear / QuantizedSwitchLinear (stacked experts). With
        ``n_experts``, only the first ``n_experts`` stacked experts (a view): MLX picks its
        gathered-matmul kernel by the number of experts, so a call that doesn't need the
        stacked shared expert must see exactly the routed experts' array shape."""
        weight = self.weight
        scales = cast(mx.array | None, self.get("scales"))
        biases = cast(mx.array | None, self.get("biases"))
        if n_experts is not None and n_experts != int(weight.shape[0]):
            weight = weight[:n_experts]
            scales = None if scales is None else scales[:n_experts]
            biases = None if biases is None else biases[:n_experts]
        if self.quantization is None:
            return mx.gather_mm(
                x,
                weight.swapaxes(-1, -2),
                lhs_indices=lhs_indices,
                rhs_indices=rhs_indices,
                sorted_indices=sorted_indices,
            )
        assert scales is not None
        group_size, bits, mode = self.quantization
        return mx.gather_qmm(
            x,
            weight,
            scales,
            biases,
            lhs_indices=lhs_indices,
            rhs_indices=rhs_indices,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
            sorted_indices=sorted_indices,
        )


class FastMLP(nn.Module):
    """mlx-lm's Laguna MLP with the gate and up projections as one matmul."""

    def __init__(self, mlp: MLP) -> None:
        super().__init__()
        self.gate_up_proj = FusedLinear([mlp.gate_proj, mlp.up_proj])
        self.down_proj = mlp.down_proj

    def __call__(self, x: mx.array) -> mx.array:
        if self.gate_up_proj.use_fused(x):
            gate_up = self.gate_up_proj(x)
            hidden = self.gate_up_proj.splits[0]
            gate, up = gate_up[..., :hidden], gate_up[..., hidden:]
        else:
            gate, up = self.gate_up_proj.parts(x)
        return self.down_proj(_swiglu(gate, up))


# Query rows per attention call in a sliding-window prefill, and the key alignment that
# keeps MLX's attention kernel on the same key blocks (its block is 16 or 32 keys).
_SLIDING_QUERY_BLOCK = 256
_KEY_ALIGNMENT = 64


def _sliding_window_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    mask: mx.array,
    scale: float,
    window: int,
) -> mx.array:
    """A sliding-window layer's prefill attention, a block of queries at a time over only
    the keys the block's window can reach, instead of every query over every key.

    MLX's attention kernel walks the keys in fixed blocks with an online softmax. A key
    block that is masked for every query row leaves that row's result exactly as it was:
    before the first unmasked block its contribution is multiplied by exp(-huge) = 0, after
    it its weights are exp(-huge) = 0. Starting the keys at a multiple of the block size
    before the first key any row of the query block can see, and ending them after the last
    one, keeps the remaining blocks where they were, so each row is computed exactly as the
    whole call computes it."""
    n_queries = queries.shape[2]
    n_keys = keys.shape[2]
    # The cache returns the window's earlier keys and then the chunk's: query row r is at
    # key index n_keys - n_queries + r, and sees keys (that - window, that].
    first_row_key = n_keys - n_queries
    outputs: list[mx.array] = []
    start = 0
    while start < n_queries:
        end = start + _SLIDING_QUERY_BLOCK
        if n_queries - end < _SLIDING_QUERY_BLOCK // 2:
            # No short last block: MLX runs a different kernel for a few query rows.
            end = n_queries
        first_key = first_row_key + start - window + 1
        key_start = max(0, first_key // _KEY_ALIGNMENT * _KEY_ALIGNMENT)
        key_end = first_row_key + end
        key_end = min(
            n_keys, (key_end + _KEY_ALIGNMENT - 1) // _KEY_ALIGNMENT * _KEY_ALIGNMENT
        )
        outputs.append(
            mx.fast.scaled_dot_product_attention(
                queries[:, :, start:end],
                keys[:, :, key_start:key_end],
                values[:, :, key_start:key_end],
                scale=scale,
                mask=mask[..., start:end, key_start:key_end],
            )
        )
        start = end
    return mx.concatenate(outputs, axis=2)


@partial(mx.compile, shapeless=True)
def _gate(output: mx.array, gate_logits: mx.array) -> mx.array:
    # mlx-lm: softplus(g.astype(float32)).astype(output.dtype), broadcast over head_dim.
    gate = nn.softplus(gate_logits.astype(mx.float32)).astype(output.dtype)
    return output * gate


class FastAttention(nn.Module):
    """mlx-lm's Laguna attention with one q/k/v matmul and one norm+RoPE kernel."""

    def __init__(self, attention: Attention) -> None:
        super().__init__()
        self.n_heads = attention.n_heads
        self.n_kv_heads = attention.n_kv_heads
        self.head_dim = attention.head_dim
        self.scale = attention.scale
        self.use_sliding = attention.use_sliding
        self.gating = bool(attention.gating)
        self.qkv_proj = FusedLinear(
            [attention.q_proj, attention.k_proj, attention.v_proj]
        )
        if self.gating:
            self.g_proj = attention.g_proj
        self.o_proj = attention.o_proj
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.eps = cast(float, getattr(attention.q_norm, "eps"))  # noqa: B009
        rope = attention.rope
        self.rotary_dims = cast(int, getattr(rope, "dims"))  # noqa: B009
        if isinstance(rope, YarnRoPE):
            freqs = cast(mx.array, getattr(rope, "_freqs"))  # noqa: B009
            self._freqs = freqs.astype(mx.float32)
            self._base = 0.0
            mscale = cast(float, getattr(rope, "mscale"))  # noqa: B009
            self._magnitude_scale = None if mscale == 1.0 else mscale
        else:
            self._freqs = None
            self._base = cast(float, getattr(rope, "base"))  # noqa: B009
            self._magnitude_scale = None

    @staticmethod
    def supports(attention: Attention) -> bool:
        rope = attention.rope
        if attention.head_dim != _HEAD_DIM or bool(getattr(rope, "traditional", True)):
            return False
        # The kernel rotates dims d and d + dims/2 within a lane group of 4.
        dims = cast(int, getattr(rope, "dims"))  # noqa: B009
        if dims % 8 != 0 or not 0 < dims <= _HEAD_DIM:
            return False
        if cast(float, getattr(attention.q_norm, "eps")) != cast(  # noqa: B009
            float,
            getattr(attention.k_norm, "eps"),  # noqa: B009
        ):
            return False
        if isinstance(rope, YarnRoPE):
            supported_rope = True
        elif isinstance(rope, nn.RoPE):
            supported_rope = cast(float, getattr(rope, "scale")) == 1.0  # noqa: B009
        else:
            supported_rope = False
        return supported_rope and _fusable(
            [attention.q_proj, attention.k_proj, attention.v_proj]
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        q_width, kv_width, _ = self.qkv_proj.splits
        if self.qkv_proj.use_fused(x):
            qkv = self.qkv_proj(x)
            sources = (qkv, qkv, qkv)
            columns = (0, q_width, q_width + kv_width)
        else:
            q, k, v = self.qkv_proj.parts(x)
            sources = (q, k, v)
            columns = (0, 0, 0)
        if cache is None:
            offsets = mx.zeros((batch,), dtype=mx.int32)
        else:
            offset = cast(int | mx.array, cache.offset)
            if isinstance(offset, mx.array):
                offsets = offset.astype(mx.int32)
            else:
                offsets = mx.array([offset] * batch, dtype=mx.int32)
        queries, keys, values = kernels.qk_norm_rope(
            sources,
            columns,
            self.q_norm.weight,
            self.k_norm.weight,
            n_q_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            eps=self.eps,
            rotary_dims=self.rotary_dims,
            freqs=self._freqs,
            base=self._base,
            magnitude_scale=self._magnitude_scale,
            offsets=offsets,
        )
        if cache is not None:
            keys, values = cast(
                tuple[mx.array, mx.array],
                cache.update_and_fetch(keys, values),  # pyright: ignore[reportAny]
            )
        window = cast(int | None, getattr(cache, "max_size", None))
        if (
            self.use_sliding
            and batch == 1
            and window is not None
            and isinstance(mask, mx.array)
            and seq_len > 2 * _SLIDING_QUERY_BLOCK
            and not hasattr(cache, "bits")
        ):
            output = _sliding_window_attention(
                queries, keys, values, mask, self.scale, window
            )
        else:
            # mlx-lm passes the mask through to MLX, which also takes "causal".
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=cast(mx.array | None, mask),
            )
        output = output.transpose(0, 2, 1, 3)
        if self.gating:
            gate_logits = self.g_proj(x)
            output = _gate(output, gate_logits[..., None])
        return self.o_proj(output.reshape(batch, seq_len, -1))


class FastMoE(nn.Module):
    """mlx-lm's Laguna MoE: the router in one kernel, the gate and up projections of the
    routed and the shared experts as one matmul each, the combination in one kernel.

    When the shared expert is quantized like the routed ones (Laguna XS's NVFP4 build), it
    is stacked after them as one more expert: a decode step then runs it in the same two
    gathered matmuls as the routed experts. More tokens run it on its own, through views of
    the stacked weights."""

    def __init__(self, moe: MoE) -> None:
        super().__init__()
        gate = moe.gate
        self.router = gate.proj
        self.bias = gate.e_score_correction_bias
        self.top_k = gate.top_k
        self.n_experts = gate.num_experts
        self.routed_scaling_factor = moe.routed_scaling_factor
        switch = moe.switch_mlp
        self.experts_gate_up = FusedLinear([switch.gate_proj, switch.up_proj], axis=1)
        self.experts_down = FusedLinear([switch.down_proj], axis=1)
        self.shared_expert = FastMLP(moe.shared_expert)
        self.shared_stacked = _stackable(
            [switch.gate_proj, switch.up_proj, switch.down_proj],
            [
                moe.shared_expert.gate_proj,
                moe.shared_expert.up_proj,
                moe.shared_expert.down_proj,
            ],
        )
        if self.shared_stacked:
            self.experts_gate_up.append_expert(self.shared_expert.gate_up_proj)
            self.experts_down.append_expert(FusedLinear([moe.shared_expert.down_proj]))
            # The shared expert on its own reads its rows of the stacked weights.
            self.shared_expert.gate_up_proj.view_expert(
                self.experts_gate_up, self.n_experts
            )
            down = cast(nn.Module, self.shared_expert.down_proj)
            for name in ("weight", "scales", "biases"):
                if name in down:
                    down[name] = cast(mx.array, self.experts_down[name])[self.n_experts]
        # Not a module attribute: MLX would treat the arrays as parameters.
        object.__setattr__(self, "_lhs_indices", {})

    @staticmethod
    def supports(moe: MoE) -> bool:
        gate = moe.gate
        switch = moe.switch_mlp
        return (
            gate.softcap == 0.0
            and gate.score_func == "sigmoid"
            and gate.num_experts <= 1024
            and _fusable([switch.gate_proj, switch.up_proj])
            and _fusable([switch.down_proj])
            and _fusable([moe.shared_expert.gate_proj, moe.shared_expert.up_proj])
        )

    def _lhs(self, shape: tuple[int, ...]) -> mx.array:
        """gather_qmm's default lhs indices for an input of this batch shape, kept so that
        each call doesn't launch an arange."""
        cached = cast(dict[tuple[int, ...], mx.array], self._lhs_indices)
        indices = cached.get(shape)
        if indices is None:
            size = 1
            for dim in shape:
                size *= dim
            indices = mx.reshape(mx.arange(size, dtype=mx.uint32), shape)
            mx.eval(indices)
            cached[shape] = indices
        return indices

    def _experts(
        self, x: mx.array, indices: mx.array, do_sort: bool, with_shared: bool = False
    ) -> mx.array:
        # mlx-lm's SwitchGLU with the gate and up projections fused. Without the shared
        # expert's slot, the gathers see only the routed experts (n_experts of them).
        n_experts = None if with_shared else self.n_experts
        x = mx.expand_dims(x, (-2, -3))
        if do_sort:
            *_, k = indices.shape
            flat = indices.flatten()
            order = mx.argsort(flat)
            inverse_order = mx.argsort(order)
            x = x.flatten(0, -3)[order // k]
            idx = flat[order]
        else:
            inverse_order = None
            idx = indices
        gate_up = self.experts_gate_up.gather(
            x,
            None if do_sort else self._lhs(tuple(x.shape[:-2])),
            idx,
            do_sort,
            n_experts,
        )
        hidden = self.experts_gate_up.splits[0]
        activated = _swiglu(gate_up[..., :hidden], gate_up[..., hidden:])
        y = self.experts_down.gather(
            activated,
            None if do_sort else self._lhs(tuple(activated.shape[:-2])),
            idx,
            do_sort,
            n_experts,
        )
        if inverse_order is not None:
            y = mx.unflatten(y[inverse_order], 0, indices.shape)
        return y.squeeze(-2)

    def __call__(self, x: mx.array, residual: mx.array | None = None) -> mx.array:
        """The MoE output, plus ``residual`` when given."""
        return self.forward(x, residual, None)[0]

    def forward(
        self,
        x: mx.array,
        residual: mx.array | None,
        next_norm: tuple[mx.array, float] | None,
    ) -> tuple[mx.array, mx.array | None]:
        """The MoE output (plus ``residual``), and with ``next_norm`` (weight, eps) also its
        RMSNorm (the next layer's input norm)."""
        rows = x.size // x.shape[-1]
        # mlx-lm sorts the tokens by expert from 64 (token, expert) pairs on.
        do_sort = rows * self.top_k >= 64
        logits = self.router(x)
        # For a single token the stacked shared expert runs in the same gathered matmuls,
        # which compute a single row exactly as its own matmul does (for more rows MLX
        # runs a multi-row vector kernel for it instead).
        if self.shared_stacked and rows == 1:
            indices, weights = kernels.route(
                logits, self.bias, self.top_k, extra_index=self.n_experts
            )
            experts = self._experts(x, indices, do_sort, with_shared=True)
            shared = None
        else:
            indices, weights = kernels.route(logits, self.bias, self.top_k)
            experts = self._experts(x, indices, do_sort)
            shared = self.shared_expert(x)
        return kernels.combine_experts(
            experts, weights, shared, self.routed_scaling_factor, residual, next_norm
        )


class FastTransformerBlock(TransformerBlock):
    """mlx-lm's Laguna block with the post-attention residual add and RMSNorm fused, and
    the MoE adding the residual itself (unless tensor parallelism sums its output). The
    MoE also computes the next block's input RMSNorm, which that block takes when it is
    given exactly this output."""

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        prenormed = cast(
            tuple[mx.array, mx.array] | None, self.__dict__.get("_prenormed")
        )
        if prenormed is not None and prenormed[0] is x:
            normed_input = prenormed[1]
        else:
            normed_input = self.input_layernorm(x)
        object.__setattr__(self, "_prenormed", None)
        r = self.self_attn(normed_input, mask, cache)
        norm = self.post_attention_layernorm
        h, normed = kernels.add_rms_norm(
            x,
            r,
            norm.weight,
            cast(float, getattr(norm, "eps")),  # noqa: B009
        )
        mlp = cast(nn.Module, self.mlp)
        if not isinstance(mlp, FastMoE):
            return h + mlp(normed)
        next_block = cast(
            "FastTransformerBlock | None", self.__dict__.get("_next_block")
        )
        if next_block is None:
            return mlp(normed, residual=h)
        next_norm = next_block.input_layernorm
        out, next_normed = mlp.forward(
            normed,
            h,
            (
                next_norm.weight,
                cast(float, getattr(next_norm, "eps")),  # noqa: B009
            ),
        )
        assert next_normed is not None
        object.__setattr__(next_block, "_prenormed", (out, next_normed))
        return out


def _innermost(layer: nn.Module) -> nn.Module:
    while isinstance(layer, CustomMlxLayer):
        layer = cast(nn.Module, layer.original_layer)
    return layer


def fuse_block(block: TransformerBlock) -> None:
    """Switch one (possibly sharded) Laguna block to the fused forward pass, in place."""
    hidden_size = int(block.post_attention_layernorm.weight.shape[0])
    if hidden_size > kernels.MAX_ROW or hidden_size % 4 != 0:
        return  # the fused RMSNorm kernels keep a row in registers, as MLX's does
    attention = block.self_attn
    if FastAttention.supports(attention):
        block["self_attn"] = FastAttention(attention)
    mlp = cast(nn.Module, block.mlp)
    if isinstance(mlp, ShardedMoE):
        inner = _innermost(mlp)
        if isinstance(inner, MoE) and FastMoE.supports(inner):
            dict.__setitem__(mlp, "_original_layer", FastMoE(inner))  # pyright: ignore[reportUnknownMemberType]
    elif isinstance(mlp, MoE):
        if FastMoE.supports(mlp):
            block["mlp"] = FastMoE(mlp)
    elif isinstance(mlp, MLP) and _fusable([mlp.gate_proj, mlp.up_proj]):
        block["mlp"] = FastMLP(mlp)
    block.__class__ = FastTransformerBlock
    mx.eval(block.parameters())


def fuse_for_inference(model: nn.Module, layers: list[nn.Module]) -> None:
    """Switch every Laguna block of the (possibly sharded) model to the fused forward."""
    del model
    blocks: list[nn.Module] = []
    for layer in layers:
        block = _innermost(layer)
        if isinstance(block, TransformerBlock) and not isinstance(
            block, FastTransformerBlock
        ):
            fuse_block(block)
            mx.clear_cache()
        blocks.append(block)
    # Each block's MoE also normalises its output for the next block of this shard. Only
    # between blocks the model calls one after the other with nothing in between: not
    # into a pipeline stage's wrapped first layer, which receives its input instead.
    for index, block in enumerate(blocks):
        if not isinstance(block, FastTransformerBlock):
            continue
        following = layers[index + 1] if index + 1 < len(layers) else None
        if (
            following is not None
            and not isinstance(following, CustomMlxLayer)
            and isinstance(blocks[index + 1], FastTransformerBlock)
        ):
            object.__setattr__(block, "_next_block", blocks[index + 1])
