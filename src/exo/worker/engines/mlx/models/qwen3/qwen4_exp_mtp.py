"""Qwen3.8-Flash-Next's multi-token-prediction (MTP) head, for speculative decoding.

The released checkpoint (Qwen/Qwen3.8-Flash-Next) has one MTP layer, ``mtp.*``, which the
mlx-community conversions drop (and qwen4_exp.Model.sanitize skips).
``scripts/convert_qwen4_exp_mtp.py`` fetches those tensors, quantizes them like the trunk
and writes ``~/.exo/mtp/<model dir>/mtp.safetensors``. The head is attached at load when
that file exists, on the node that runs the model's last pipeline stage (or the whole
model); ``EXO_QWEN4_EXP_MTP=0`` leaves it off. Without the file nothing changes.

The head drafts token t+2 from the trunk's streams at position t (before the final mixer)
and the embedding of token t+1:
- the four streams are normed at full width (``pre_fc_norm_hidden``) and each goes through
  ``fc_hidden``; the normed, projected token embedding (``pre_fc_norm_embedding``,
  ``fc_embedding``) is added to every stream;
- one full-attention decoder layer (Qwen Sparse Attention and the MoE, with its own tensors
  and KV cache) runs on that;
- the head's own mixer collapses the streams for the trunk's lm_head.
No public implementation of this head was available; the wiring follows MTPLX's reading of
the shipped tensors (https://github.com/youssofal/MTPLX, mtplx/models/qwen4_exp.py,
Apache-2.0, read 2026-11-10). The code here is written against this repository's qwen4_exp
classes. The norms are zero-centred like the trunk's, so the checkpoint's gains load as
they are.

The head's KV cache is the last entry of the request's cache list on its stage
(Model.make_cache), so the prefix cache saves, restores and trims it with the trunk's.
Pair i (streams h_i, token x_{i+1}) is the head's position i: while the head is in step it
holds one position less than the trunk after a plain forward, and the same number after a
draft (generator/mtp_speculative.py).

An optional ``draft_vocab.npy`` next to the weights (token ids) makes the head draft only
among those tokens: its output layer then reads only their rows of the lm_head. A token
outside them is never drafted, which costs acceptance, never correctness.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache

from exo.worker.engines.mlx.models.qwen3.qwen4_exp import (
    DecoderLayer,
    GatedResidual,
    Model,
    QSAKVCache,
    TextArgs,
    TextModel,
    ZeroCenteredRMSNorm,
)

MTP_ROOT = Path.home() / ".exo" / "mtp"
MTP_WEIGHTS_FILE = "mtp.safetensors"
DRAFT_VOCAB_FILE = "draft_vocab.npy"


def mtp_weights_path(model_path: Path) -> Path:
    return MTP_ROOT / model_path.name / MTP_WEIGHTS_FILE


class Qwen4ExpMTP(nn.Module):
    last_streams: mx.array | None

    def __init__(self, args: TextArgs):
        super().__init__()
        d, hc = args.hidden_size, args.hc_count
        self.pre_fc_norm_embedding = ZeroCenteredRMSNorm(d, args.rms_norm_eps)
        self.pre_fc_norm_hidden = ZeroCenteredRMSNorm(d * hc, args.rms_norm_eps)
        self.fc_embedding = nn.Linear(d, d, bias=False)
        self.fc_hidden = nn.Linear(d, d, bias=False)
        # The head's layer is a full-attention layer without the n-gram block
        fa_idx = next(
            i
            for i, kind in enumerate(args.layer_types)
            if kind != "linear_attention" and (i + 1) not in args.ple_layer_ids
        )
        self.layers = [DecoderLayer(args, fa_idx)]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.hc_count = hc
        # The streams before the mixer of the last call (a plain attribute): a second draft
        # is chained from them, as vLLM chains this family's MTP layer
        object.__setattr__(self, "last_streams", None)

    def __call__(
        self, streams: mx.array, token_embeddings: mx.array, cache: QSAKVCache
    ) -> mx.array:
        """The collapsed output [B, S, d] (for the lm_head) of S (streams, next token)
        pairs, which are appended to ``cache``."""
        batch, length, width = streams.shape
        hidden = self.pre_fc_norm_hidden(streams).reshape(
            batch, length, self.hc_count, -1
        )
        embedded = self.fc_embedding(self.pre_fc_norm_embedding(token_embeddings))
        h = (self.fc_hidden(hidden) + embedded[:, :, None, :]).reshape(
            batch, length, width
        )
        mask = create_attention_mask(h, cache)
        h = self.layers[0](h, mask, cache)
        object.__setattr__(self, "last_streams", h)
        mixed, _ = self.hyper_connection_mixer.mix(h)
        return mixed


def load_mtp_head(args: TextArgs, path: Path) -> Qwen4ExpMTP:
    """The head with its converted weights; each linear's bits and group size are read off
    its packed shapes (the router gates are 8-bit, the rest 4-bit in the mlx-community
    conversion)."""
    head = Qwen4ExpMTP(args)
    raw = cast(dict[str, mx.array], mx.load(str(path)))
    weights = {
        k.removeprefix("mtp."): v for k, v in raw.items() if k.startswith("mtp.")
    }
    inputs = {
        name.removesuffix(".weight"): int(value.shape[-1])
        for name, value in cast(
            list[tuple[str, mx.array]], tree_flatten(head.parameters())
        )
        if name.endswith(".weight") and value.ndim >= 2
    }
    recipes: dict[str, dict[str, int]] = {}
    for module, width in inputs.items():
        packed, scales = (
            weights.get(module + ".weight"),
            weights.get(module + ".scales"),
        )
        if packed is not None and scales is not None and packed.dtype == mx.uint32:
            recipes[module] = {
                "bits": int(packed.shape[-1]) * 32 // width,
                "group_size": width // int(scales.shape[-1]),
            }
    nn.quantize(
        head,
        class_predicate=lambda name, _module: recipes.get(name, False),
    )
    head.load_weights(list(weights.items()), strict=True)
    head.eval()
    mx.eval(head.parameters())
    return head


def _draft_output_layer(text: TextModel, ids: mx.array) -> nn.Module:
    """An output layer for only the tokens ``ids``: their rows of the trunk's lm_head."""
    lm = cast(nn.Module, text.lm_head)
    weight = cast(mx.array, lm["weight"])
    rows = int(ids.shape[0])
    if isinstance(lm, nn.QuantizedLinear):
        bits, group_size = cast(int, lm.bits), cast(int, lm.group_size)
        out = nn.QuantizedLinear(
            int(weight.shape[1]) * 32 // bits,
            rows,
            bias=False,
            group_size=group_size,
            bits=bits,
        )
        parts = {"weight": weight[ids], "scales": cast(mx.array, lm["scales"])[ids]}
        if "biases" in lm:
            parts["biases"] = cast(mx.array, lm["biases"])[ids]
        out.update(parts)
    else:
        out = nn.Linear(int(weight.shape[1]), rows, bias=False)
        out.update({"weight": weight[ids]})
    mx.eval(out.parameters())
    return out


class MTPTracker:
    """The head on its stage: drafts for the speculative loop (``draft``, ``chain``), and
    keeps the head's cache in step with the trunk on the forwards that the loop doesn't
    drive itself (prefill, the prompt's last tokens, plain decode steps: ``observe``).

    For a forward over trunk positions [start, end), observe forms the pairs it makes
    possible: (h_{start-1}, x_start) from the streams it kept of the previous forward (the
    last RING positions), then (h_i, x_{i+1}) within the forward. A head ahead of
    ``start`` (exo trims after prefill, a prefix-cache restore) is trimmed back first. A
    restore can leave the head at start - 1 with no streams kept for that position: one
    stand-in pair (h_start, x_start) keeps it in step, one approximate key among the
    context's."""

    RING = 8

    def __init__(self, head: Qwen4ExpMTP, vocab: tuple[mx.array, nn.Module] | None):
        self.head = head
        self.vocab = vocab
        self.current: QSAKVCache | None = None
        self.recent: list[tuple[int, mx.array]] = []
        # Set while the speculative loop drives the head itself
        self.paused = False

    def observe(
        self,
        text: TextModel,
        inputs: mx.array,
        streams: mx.array,
        trunk_cache: Sequence[ArraysCache | QSAKVCache | None] | None,
    ) -> None:
        if (
            self.paused
            or trunk_cache is None
            or inputs.ndim != 2
            or inputs.shape[0] != 1
        ):
            return
        layers = len(text.model.layers)
        if len(trunk_cache) <= layers:
            return
        head_cache = trunk_cache[layers]
        trunk_kv = next(
            (c for c in trunk_cache[:layers] if isinstance(c, QSAKVCache)), None
        )
        if not isinstance(head_cache, QSAKVCache) or trunk_kv is None:
            return
        length = int(inputs.shape[1])
        end = int(trunk_kv.offset)
        start = end - length
        if head_cache is not self.current:  # one cache list per request
            self.current = head_cache
            self.recent = []
        at = int(head_cache.offset)
        if at > start:
            head_cache.trim(at - start)
            at = start
        previous = (
            next((h for p, h in reversed(self.recent) if p == start - 1), None)
            if at <= start - 1
            else None
        )
        embedded = text.model.embed_tokens(inputs)
        if previous is None and at == start - 1:
            self.head(streams[:, :1], embedded[:, :1], head_cache)
        if previous is not None:
            pair_streams = mx.concatenate([previous, streams[:, :-1]], axis=1)
            pair_tokens = embedded
        else:
            pair_streams, pair_tokens = streams[:, :-1], embedded[:, 1:]
        if pair_streams.shape[1] > 0:
            # Only the cache is used: evaluating it skips the layer's MoE and the mixer
            self.head(pair_streams, pair_tokens, head_cache)
        kept = range(max(0, length - self.RING), length)
        self.recent = [(p, h) for p, h in self.recent if p < start]
        self.recent = (
            self.recent + [(start + i, streams[:, i : i + 1]) for i in kept]
        )[-self.RING :]
        mx.async_eval(
            *[
                a
                for a in (head_cache.keys, head_cache.values, head_cache.index_keys)
                if a is not None
            ]
        )

    def draft(
        self,
        text: TextModel,
        streams: mx.array,
        tokens: mx.array,
        head_cache: QSAKVCache,
    ) -> tuple[mx.array, mx.array]:
        """The head on the pairs (streams [1, n, hc*d], tokens (n,)), appended to its
        cache: each row's greedy draft and the head's probability of it (lazy, (n,))."""
        rows = int(tokens.shape[0])
        embedded = text.model.embed_tokens(tokens.reshape(1, rows))
        mixed = self.head(streams, embedded, head_cache)[0]
        if self.vocab is not None:
            ids, out = self.vocab
            logits = out(mixed).astype(mx.float32)
            drafts = ids[mx.argmax(logits, axis=-1)]
        else:
            logits = (
                text.model.embed_tokens.as_linear(mixed)
                if text.args.tie_word_embeddings
                else text.lm_head(mixed)
            ).astype(mx.float32)
            drafts = mx.argmax(logits, axis=-1)
        probs = mx.exp(mx.max(logits, axis=-1) - mx.logsumexp(logits, axis=-1))
        return drafts, probs

    def chain(
        self, text: TextModel, draft: int, row: int, head_cache: QSAKVCache
    ) -> tuple[mx.array, mx.array]:
        """A second draft: the head again, on its own pre-mixer streams of row ``row`` of
        the last ``draft`` call (standing in for the trunk's, not computed yet) and that
        row's draft. Appends one pair, which the caller trims before the head's next pairs.
        """
        last = self.head.last_streams
        assert last is not None
        return self.draft(text, last[:, row : row + 1], mx.array([draft]), head_cache)


def attach_mtp(model: Model, model_path: Path) -> bool:
    """Attach the converted head on this rank if it holds the last pipeline stage (or the
    whole model) and the weights exist. Tensor parallelism is left alone."""
    if os.environ.get("EXO_QWEN4_EXP_MTP", "1").strip().lower() in ("0", "false", "no"):
        return False
    path = mtp_weights_path(model_path)
    if not path.exists() or model.model.tensor_group is not None:
        return False
    if not model.spec_is_last_stage():
        return False
    from exo.worker.runner.bootstrap import logger

    text = model.language_model
    head = load_mtp_head(text.args, path)
    vocab: tuple[mx.array, nn.Module] | None = None
    vocab_path = path.with_name(DRAFT_VOCAB_FILE)
    if vocab_path.exists() and not text.args.tie_word_embeddings:
        ids = mx.array(cast(np.ndarray, np.load(vocab_path)).astype(np.int32))
        vocab = (ids, _draft_output_layer(text, ids))
        logger.info(f"MTP head drafts from {int(ids.shape[0])} tokens ({vocab_path})")
    object.__setattr__(text, "mtp", MTPTracker(head, vocab))
    object.__setattr__(text.model, "keep_streams", True)
    logger.info(f"MTP speculative decoding: head from {path}")
    return True
