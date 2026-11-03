"""Speculative decoding with a draft model, one request at a time on one node.

A small draft model that shares the target's tokenizer proposes `num_draft_tokens`
tokens; the target checks them in one forward and keeps the longest prefix that matches
its own choices (mlx-lm's `speculative_generate_step`, reached through `stream_generate`'s
`draft_model=`). Every emitted token is the target's: greedy output is the target's greedy
output, and a sampled token follows the target's distribution (mlx-lm samples draft and
target independently and accepts on equality).

Off unless `EXO_DRAFT_MODEL` is set. The draft is used only by instances that already
decode one request at a time on one node (`EXO_NO_BATCH`, or a model family that cannot
batch), and only for requests whose caches can rewind. Everything else decodes as before.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import mlx.core as mx
from mlx_lm.generate import generation_stream
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.download.download_utils import resolve_existing_model
from exo.shared.constants import (
    EXO_DRAFT_MODEL_ENV,
    EXO_MAX_NUM_DRAFT_TOKENS,
    EXO_NUM_DRAFT_TOKENS_ENV,
)
from exo.shared.types.common import ModelId
from exo.worker.engines.mlx.cache import cache_length, is_non_trimmable_cache_entry
from exo.worker.engines.mlx.types import KVCacheType, Model, SnapshotCache
from exo.worker.engines.mlx.utils_mlx import (
    load_model_checked,
    load_tokenizer_for_model_id,
    run_post_load,
)
from exo.worker.runner.bootstrap import logger

DEFAULT_NUM_DRAFT_TOKENS = 3
DRAFT_PREFILL_STEP = 2048


class DraftConfigError(ValueError):
    """EXO_DRAFT_MODEL or EXO_NUM_DRAFT_TOKENS is malformed."""


@dataclass(frozen=True)
class DraftConfig:
    # A model id (looked up in the EXO model directories) or a local directory
    source: str
    num_draft_tokens: int


@dataclass(eq=False)
class DraftModel:
    model: Model
    name: str
    num_draft_tokens: int


def _parse_num_draft_tokens(raw: str | None) -> int:
    if raw is None or raw.strip() == "":
        return DEFAULT_NUM_DRAFT_TOKENS
    try:
        value = int(raw.strip())
    except ValueError:
        raise DraftConfigError(
            f"{EXO_NUM_DRAFT_TOKENS_ENV}={raw!r} is not an integer"
        ) from None
    if not 1 <= value <= EXO_MAX_NUM_DRAFT_TOKENS:
        raise DraftConfigError(
            f"{EXO_NUM_DRAFT_TOKENS_ENV}={value} is outside 1 to {EXO_MAX_NUM_DRAFT_TOKENS}"
        )
    return value


def _parse_draft_model(raw: str, target_model_id: ModelId) -> str | None:
    """`draft` for every instance, or `target=draft[,target=draft...]` per target model."""
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not entries:
        return None
    if len(entries) == 1 and "=" not in entries[0]:
        return entries[0]
    mapping: dict[str, str] = {}
    for entry in entries:
        target, sep, draft = (part.strip() for part in entry.partition("="))
        if not sep or not target or not draft:
            raise DraftConfigError(
                f"{EXO_DRAFT_MODEL_ENV}: {entry!r} is not 'target=draft' (use one draft "
                "for every instance, or a comma-separated list of target=draft pairs)"
            )
        if target in mapping:
            raise DraftConfigError(f"{EXO_DRAFT_MODEL_ENV}: {target!r} is listed twice")
        mapping[target] = draft
    return mapping.get(str(target_model_id))


def resolve_draft_config(
    target_model_id: ModelId, env: Mapping[str, str] = os.environ
) -> DraftConfig | None:
    """The draft configured for `target_model_id`, or None when there is none.

    Raises DraftConfigError for a malformed setting, also when no draft applies to this
    target, so a typo is reported on every node rather than silently ignored."""
    raw = env.get(EXO_DRAFT_MODEL_ENV, "")
    num_draft_tokens = _parse_num_draft_tokens(env.get(EXO_NUM_DRAFT_TOKENS_ENV))
    source = _parse_draft_model(raw, target_model_id)
    if source is None:
        return None
    return DraftConfig(source=source, num_draft_tokens=num_draft_tokens)


def resolve_draft_path(source: str) -> Path | None:
    """A local directory as given, or the model id's complete copy in the EXO model dirs."""
    if source.startswith(("/", "~", ".")):
        path = Path(source).expanduser()
        return path if (path / "config.json").is_file() else None
    return resolve_existing_model(ModelId(source))


def logits_width(model: Model) -> int | None:
    args = getattr(model, "args", None)
    vocab_size = getattr(args, "vocab_size", None)
    return vocab_size if isinstance(vocab_size, int) else None


def draft_incompatibility(
    target_model: Model,
    target_tokenizer: TokenizerWrapper,
    draft_model: Model,
    draft_tokenizer: TokenizerWrapper,
) -> str | None:
    """Why the draft cannot propose tokens for the target, or None if it can.

    Exact-match acceptance needs the same token for the same id, so the vocabularies must
    be identical (special and added tokens included). The draft's logits may be narrower
    than the target's (padded embeddings differ) but not wider, or it could propose an id
    the target cannot embed. EOS ids are not compared: the target's tokenizer decides
    when to stop."""
    target_vocab = target_tokenizer.get_vocab()
    draft_vocab = draft_tokenizer.get_vocab()
    if target_vocab != draft_vocab:
        differing = sum(
            1
            for token in target_vocab.keys() | draft_vocab.keys()
            if target_vocab.get(token) != draft_vocab.get(token)
        )
        return (
            f"tokenizer vocabularies differ ({len(target_vocab)} and {len(draft_vocab)} "
            f"tokens, {differing} token ids differ)"
        )
    target_width = logits_width(target_model)
    draft_width = logits_width(draft_model)
    if (
        target_width is not None
        and draft_width is not None
        and draft_width > target_width
    ):
        return (
            f"draft logits are wider than the target's ({draft_width} > {target_width})"
        )
    return None


def load_draft(
    config: DraftConfig,
    target_model: Model,
    target_tokenizer: TokenizerWrapper,
) -> DraftModel | None:
    """Load the draft next to the target, or return None (decode without it) when it is
    missing, fails to load or does not share the target's tokenizer."""
    path = resolve_draft_path(config.source)
    if path is None:
        logger.warning(
            f"Draft model {config.source} is not on this node (it must already be "
            "downloaded or be a local directory); decoding without a draft model"
        )
        return None
    try:
        model = cast(Model, load_model_checked(path))
        mx.eval(model)
        run_post_load(model, path)
        tokenizer = load_tokenizer_for_model_id(ModelId(config.source), path)
    except Exception:
        logger.opt(exception=True).warning(
            f"Draft model {config.source} failed to load; decoding without a draft model"
        )
        return None
    reason = draft_incompatibility(target_model, target_tokenizer, model, tokenizer)
    if reason is not None:
        logger.warning(
            f"Draft model {config.source} cannot draft for this model: {reason}; "
            "decoding without a draft model"
        )
        del model
        mx.clear_cache()
        return None
    logger.info(
        f"Loaded draft model {config.source} from {path} "
        f"(num_draft_tokens={config.num_draft_tokens})"
    )
    return DraftModel(
        model=model, name=config.source, num_draft_tokens=config.num_draft_tokens
    )


def _rewind_blocker(cache: KVCacheType, which: str) -> str | None:
    for c in cache:
        if (
            is_non_trimmable_cache_entry(c)
            or isinstance(c, SnapshotCache)
            or not c.is_trimmable()  # type: ignore[reportUnknownMemberType]
        ):
            return (
                f"the {which} cache ({type(c).__name__}) cannot rewind rejected tokens"
            )
    return None


def speculation_blocker(
    model: Model,
    caches: KVCacheType,
    draft: DraftModel,
    draft_caches: KVCacheType,
    group: mx.distributed.Group | None,
    has_vision: bool,
    num_draft_tokens: int,
) -> str | None:
    """Why this request must decode without the draft, or None if it can speculate."""
    if group is not None:
        return "the instance is distributed"
    if num_draft_tokens <= 0:
        return "the request set num_draft_tokens to 0"
    if has_vision:
        return "the request has images"
    if len(caches) != len(model.layers):
        # mlx-lm splits target and draft caches at len(model.layers)
        return (
            f"the target has {len(caches)} cache entries for {len(model.layers)} layers"
        )
    if len(draft_caches) != len(draft.model.layers):
        return f"the draft has {len(draft_caches)} cache entries for {len(draft.model.layers)} layers"
    return _rewind_blocker(caches, "target") or _rewind_blocker(draft_caches, "draft")


def prefill_draft(draft_model: Model, tokens: mx.array, cache: KVCacheType) -> None:
    """Run the draft over `tokens` so its cache ends where the target's does."""
    with mx.stream(generation_stream):
        for start in range(0, len(tokens), DRAFT_PREFILL_STEP):
            chunk = tokens[start : start + DRAFT_PREFILL_STEP]
            draft_model(chunk[None], cache=cache)
            mx.eval([c.state for c in cache])  # type: ignore


def align_draft(
    draft: DraftModel,
    draft_caches: KVCacheType,
    all_prompt_tokens: mx.array,
    caches: KVCacheType,
    decode_prompt_length: int,
) -> str | None:
    """Prefill the draft to the target's position. Returns why it could not, or None.

    The target may have been filled from the prefix cache (partly or exactly), remotely or
    locally, so the draft follows the target's actual length rather than what was
    prefilled this request. Decoding then starts both from the last
    `decode_prompt_length` prompt tokens."""
    target_length = cache_length(caches)
    expected = len(all_prompt_tokens) - decode_prompt_length
    if target_length != expected:
        return f"the target cache holds {target_length} tokens, expected {expected}"
    prefill_draft(draft.model, all_prompt_tokens[:target_length], draft_caches)
    draft_length = cache_length(draft_caches)
    if draft_length != target_length:
        return f"the draft cache holds {draft_length} tokens, expected {target_length}"
    return None


@dataclass
class SpeculativeStats:
    """Acceptance counted from `from_draft` on each emitted token.

    A round emits the accepted draft tokens (from_draft) and then one token of the
    target's own (a correction, or the bonus token after a fully accepted draft); only the
    last round may end on a draft token, when max_tokens cuts it. mlx-lm proposes
    min(k, max_tokens - emitted) tokens in each round."""

    num_draft_tokens: int
    max_tokens: int
    proposed: int = 0
    accepted: int = 0
    rounds: int = 0
    emitted: int = 0
    _round_open: bool = field(default=False, repr=False)

    def observe(self, from_draft: bool) -> None:
        if not self._round_open:
            self.rounds += 1
            self.proposed += max(
                min(self.num_draft_tokens, self.max_tokens - self.emitted), 0
            )
            self._round_open = True
        self.emitted += 1
        if from_draft:
            self.accepted += 1
        else:
            self._round_open = False

    @property
    def acceptance(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    def summary(self) -> str:
        per_round = self.emitted / self.rounds if self.rounds else 0.0
        return (
            f"accepted {self.accepted}/{self.proposed} draft tokens "
            f"({100 * self.acceptance:.1f}%), {self.rounds} rounds, "
            f"{per_round:.2f} tokens per round, num_draft_tokens={self.num_draft_tokens}"
        )
