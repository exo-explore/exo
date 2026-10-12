"""DeepSeek V4 multi-token prediction (MTP) block.

DeepSeek V4 checkpoints ship one MTP block (``mtp.0.*``) that predicts the
token after next from the main model's final hyper-connection state and the
embedding of the next token. mlx_lm drops these weights, so they are
converted separately (see ``convert_mtp_weights``) and loaded next to the
main model to draft tokens for speculative decoding.

Reference (DeepSeek-V4 ``inference/model.py``, ``MTPBlock.forward``)::

    e = enorm(embed(next_ids))
    x = hnorm(h)                                  # h: [B, S, hc, D]
    x = e_proj(e).unsqueeze(2) + h_proj(x)
    x = Block(x)                                  # layer_id = num_hidden_layers
    logits = head(hc_head(x), mtp.norm)           # main model's lm_head
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.deepseek_v4 import (
    DeepseekV4Block,
    DeepseekV4Cache,
    DeepseekV4Model,
    HyperHead,
    Model,
    ModelArgs,
)

MTP_WEIGHTS_FILE = "mtp.safetensors"


class DeepseekV4MTP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.e_proj = nn.Linear(dim, dim, bias=False)
        self.h_proj = nn.Linear(dim, dim, bias=False)
        self.enorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.norm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.hc_head = HyperHead(dim, args.hc_mult, args.rms_norm_eps, args.hc_eps)
        self.block = DeepseekV4Block(args, args.num_hidden_layers)

    def __call__(
        self,
        hidden: mx.array,
        next_embeddings: mx.array,
        next_ids: mx.array,
        cache: DeepseekV4Cache,
    ) -> mx.array:
        """Final normed hidden state ``[B, S, D]``; apply the main lm_head to it.

        ``hidden`` is the main model's pre-head hyper-connection state at
        positions ``p`` and ``next_ids`` are the tokens at ``p + 1``.
        """
        return self.forward_with_state(hidden, next_embeddings, next_ids, cache)[0]

    def forward_with_state(
        self,
        hidden: mx.array,
        next_embeddings: mx.array,
        next_ids: mx.array,
        cache: DeepseekV4Cache,
    ) -> tuple[mx.array, mx.array]:
        """Like ``__call__`` but also returns the block's hyper-connection
        output, which stands in for the main hidden state when chaining the
        block to draft further ahead."""
        e = self.enorm(next_embeddings)
        x = self.hnorm(hidden)
        x = self.e_proj(e)[:, :, None, :] + self.h_proj(x)
        x = self.block(x, cache, next_ids)
        return self.norm(self.hc_head(x)), x


def _mtp_key(key: str) -> str:
    """Map an ``mtp.0.*`` checkpoint key into the main model's key scheme so
    mlx_lm's sanitize converts it like a regular layer."""
    rest = key[len("mtp.0.") :]
    if rest.split(".")[0] in ("e_proj", "h_proj", "enorm", "hnorm", "norm"):
        return "mtp_top." + rest
    if rest.startswith("hc_head_"):
        return "mtp_top.hc_head." + rest[len("hc_head_") :]
    return f"layers.{_MTP_LAYER}." + rest


_MTP_LAYER = 10_000


def convert_mtp_weights(checkpoint: Path, args: ModelArgs) -> dict[str, mx.array]:
    """Convert the ``mtp.0.*`` tensors of an original DeepSeek V4 shard into
    ``DeepseekV4MTP`` parameters: FP8 linears dequantized and re-quantized to
    8-bit affine (group 64, like mlx-community's conversion), FP4 experts kept
    as mxfp4."""
    from mlx_lm.utils import (
        _load_safetensors_with_e8m0,  # pyright: ignore[reportPrivateUsage]
    )

    raw = _load_safetensors_with_e8m0(str(checkpoint))
    weights = {_mtp_key(k): v for k, v in raw.items() if k.startswith("mtp.0.")}
    sanitize_self = SimpleNamespace(
        args=SimpleNamespace(
            num_hidden_layers=_MTP_LAYER + 1,
            n_routed_experts=args.n_routed_experts,
        )
    )
    sanitized = cast(
        dict[str, mx.array],
        Model.sanitize(sanitize_self, weights),  # pyright: ignore[reportArgumentType]
    )

    params: dict[str, mx.array] = {}
    for key, value in sanitized.items():
        if key.startswith("model.layers."):
            key = "block." + key.split(".", 3)[3]
        elif key.startswith("mtp_top."):
            key = key[len("mtp_top.") :]
        else:
            raise ValueError(f"unexpected MTP tensor {key}")
        params[key] = value

    mtp = DeepseekV4MTP(args)
    mtp.load_weights(list(params.items()), strict=False)
    nn.quantize(
        mtp,
        group_size=64,
        bits=8,
        class_predicate=lambda _, module: type(module) is nn.Linear,
    )
    keep_fp32 = (".hc_", "hc_head.", "attn_sink", "e_score_correction_bias")
    converted: dict[str, mx.array] = {}
    for key, value in cast(list[tuple[str, mx.array]], tree_flatten(mtp.parameters())):
        if mx.issubdtype(value.dtype, mx.floating):
            fp32 = any(marker in key for marker in keep_fp32)
            value = value.astype(mx.float32 if fp32 else mx.bfloat16)
        converted[key] = value
    missing = {
        key
        for key, _ in cast(list[tuple[str, mx.array]], tree_flatten(mtp.parameters()))
    } - set(converted)
    assert not missing, missing
    return converted


def load_mtp(weights_path: Path, args: ModelArgs) -> DeepseekV4MTP:
    mtp = DeepseekV4MTP(args)
    nn.quantize(
        mtp,
        group_size=64,
        bits=8,
        class_predicate=lambda _, module: type(module) is nn.Linear,
    )
    mtp.load_weights(str(weights_path), strict=True)
    mx.eval(mtp.parameters())
    return mtp


MTP_ROOT = Path.home() / ".exo" / "mtp"


def mtp_weights_path(model_path: Path) -> Path:
    return MTP_ROOT / model_path.name / MTP_WEIGHTS_FILE


def attach_mtp(
    model: object, model_path: Path, group: mx.distributed.Group | None
) -> None:
    """Load the converted MTP block for a DeepSeek V4 model, if present.

    With a tensor-parallel ``group`` the block is sharded like the main
    layers, and each rank drafts from its own slice of the vocabulary.
    """
    if not isinstance(model, Model):
        return
    path = mtp_weights_path(model_path)
    if not path.exists():
        return
    mtp = load_mtp(path, model.args)
    if group is not None and group.size() > 1:
        from functools import partial

        from mlx.nn.layers.distributed import shard_inplace

        from exo.worker.engines.mlx.auto_parallel import shard_deepseek_v4_block

        shard_deepseek_v4_block(
            mtp.block,
            group,
            partial(shard_inplace, sharding="all-to-sharded", group=group),
            partial(shard_inplace, sharding="sharded-to-all", group=group),
        )
        mx.eval(mtp.parameters())
        mx.clear_cache()
    object.__setattr__(mtp, "exo_group", group)
    object.__setattr__(model, "exo_mtp", mtp)


def draft_argmax(model: Model, mtp: "DeepseekV4MTP", hidden: mx.array) -> mx.array:
    """Greedy draft token from the MTP output ``[B, D]``.

    Under tensor parallelism each rank scores only its slice of the
    vocabulary and the ranks exchange their best (score, token) pair, so no
    rank reads the whole lm_head.
    """
    group = cast(mx.distributed.Group | None, getattr(mtp, "exo_group", None))
    head = model.lm_head
    if group is None or group.size() == 1 or not isinstance(head, nn.QuantizedLinear):
        return mx.argmax(head(hidden), axis=-1)
    world_size = group.size()
    vocab = head.weight.shape[0]
    part = (vocab + world_size - 1) // world_size
    start = group.rank() * part
    end = min(vocab, start + part)
    biases = head.get("biases")
    logits = mx.quantized_matmul(
        hidden,
        head.weight[start:end],
        cast(mx.array, head.scales)[start:end],
        None if biases is None else cast(mx.array, biases)[start:end],
        transpose=True,
        group_size=head.group_size,
        bits=head.bits,
        mode=head.mode,
    )
    best = mx.max(logits, axis=-1).astype(mx.float32)
    token = (mx.argmax(logits, axis=-1) + start).astype(mx.float32)
    candidates = mx.distributed.all_gather(
        mx.stack([best, token], axis=-1), group=group
    ).reshape(world_size, -1, 2)
    winner = mx.argmax(candidates[..., 0], axis=0)
    return mx.take_along_axis(candidates[..., 1], winner[None], axis=0)[0].astype(
        mx.uint32
    )


def get_mtp(model: object) -> DeepseekV4MTP | None:
    return cast(DeepseekV4MTP | None, getattr(model, "exo_mtp", None))


_original_inner_call = DeepseekV4Model.__call__
# Layer after which a forward submits what it has built so far (None: never).
early_submit_layer: int | None = None


def _capturing_inner_call(
    self: DeepseekV4Model, inputs: mx.array, cache: list[Any] | None = None
) -> mx.array:
    """DeepseekV4Model.__call__ that also keeps the pre-head hidden state."""
    batch, sequence = inputs.shape
    h = self.embed_tokens(inputs)
    h = mx.broadcast_to(
        h[:, :, None, :], (batch, sequence, self.args.hc_mult, h.shape[-1])
    )
    h = mx.contiguous(h)
    layer_caches = (
        cast(list[DeepseekV4Cache | None], cache)
        if cache is not None
        else [None] * len(self.layers)
    )
    split = early_submit_layer
    for i, (layer, layer_cache) in enumerate(
        zip(self.layers, layer_caches, strict=True)
    ):
        h = layer(h, layer_cache, inputs)
        if i == split:
            # Start the GPU on the first layers while the rest of the graph is
            # still being built (used for speculative verification, whose
            # graph is built while the GPU would otherwise sit idle).
            mx.async_eval(h)
    object.__setattr__(self, "exo_last_hidden", h)
    return self.norm(self.hc_head(h))


def patch_deepseek_v4_hidden_capture() -> None:
    DeepseekV4Model.__call__ = _capturing_inner_call
