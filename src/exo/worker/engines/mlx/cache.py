import gc
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

import mlx.core as mx
import numpy as np
import psutil
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType, Model
from exo.worker.engines.mlx.constants import (
    CACHE_GROUP_SIZE,
    KV_CACHE_BITS,
    TURBOQUANT_BITS,
    TURBOQUANT_ENABLED,
    TURBOQUANT_RESIDUAL,
    TURBOQUANT_SKETCH_DIM,
)
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import MediaRegion


# Fraction of device memory above which LRU eviction kicks in.
# Smaller machines need more aggressive eviction.
def _default_memory_threshold() -> float:
    total_gb = Memory.from_bytes(psutil.virtual_memory().total).in_gb
    if total_gb >= 128:
        return 0.85
    if total_gb >= 64:
        return 0.80
    if total_gb >= 32:
        return 0.75
    return 0.70


_MEMORY_THRESHOLD = float(
    os.environ.get("EXO_MEMORY_THRESHOLD", _default_memory_threshold())
)


class CacheSnapshot:
    """Snapshot of states at a known token position."""

    def __init__(
        self, states: list[RotatingKVCache | ArraysCache | None], token_count: int
    ):
        self.states = states
        self.token_count = token_count


def _detached_copy(a: mx.array) -> mx.array:
    dtype = a.dtype
    if dtype == mx.bfloat16:
        return mx.array(np.array(a.astype(mx.float32))).astype(mx.bfloat16)
    return mx.array(np.array(a))


def copy_rotating_kv_cache(cache: RotatingKVCache) -> RotatingKVCache | None:
    """
    Deepcopy copies the metadata associated with an mx array.
    Specifically, it shares a shared_ptr to the underlying data and
    the mlx graph inputs of the array. This causes a memory leak for rotating
    kv cache. By creating an np array, no metadata is stored so the old cache
    can be cleaned up nicely.
    """
    if cache.keys is None or cache.values is None:
        return None
    n = min(cache.max_size, cache.keys.shape[2])
    k_slice = _detached_copy(cache.keys[..., -n:, :])
    v_slice = _detached_copy(cache.values[..., -n:, :])
    mx.eval(k_slice, v_slice)
    snap = RotatingKVCache.__new__(RotatingKVCache)
    snap.keys = k_slice
    snap.values = v_slice
    snap.offset = cache.offset
    snap._idx = n
    snap.keep = cache.keep
    snap.max_size = cache.max_size
    return snap


def snapshot_ssm_states(cache: KVCacheType) -> CacheSnapshot:
    states: list[ArraysCache | RotatingKVCache | None] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            states.append(deepcopy(c))
        elif isinstance(c, RotatingKVCache):
            states.append(copy_rotating_kv_cache(c))
        else:
            states.append(None)
    token_count = cache_length(cache)
    return CacheSnapshot(states=states, token_count=token_count)


def _find_nearest_snapshot(
    snapshots: list[CacheSnapshot],
    target_token_count: int,
) -> CacheSnapshot | None:
    best: CacheSnapshot | None = None
    for snap in snapshots:
        if snap.token_count <= target_token_count and (
            best is None or snap.token_count > best.token_count
        ):
            best = snap
    return best


def has_non_kv_caches(cache: KVCacheType) -> bool:
    """Check if a cache contains any ArraysCache (SSM) entries."""
    return any(isinstance(c, (ArraysCache, RotatingKVCache)) for c in cache)


def _array_like_nbytes(value: Any) -> int:
    """Sum nbytes across an mx.array, a list of mx.arrays, or None-ish values."""
    if value is None:
        return 0
    if isinstance(value, list):
        total = 0
        for item in value:  # type: ignore[reportUnknownVariableType]
            nb = getattr(item, "nbytes", None)  # type: ignore[reportUnknownArgumentType]
            if nb is not None:
                total += int(nb)  # type: ignore[reportAny]
        return total
    nb = getattr(value, "nbytes", None)
    if nb is None:
        return 0
    return int(nb)  # type: ignore[reportAny]


def _cache_object_nbytes(cache_entry: Any) -> int:
    """Sum nbytes across a single per-layer cache object's K/V/state tensors."""
    total = 0
    for attr in ("keys", "values", "state"):
        total += _array_like_nbytes(getattr(cache_entry, attr, None))
    return total


class _TrieNode:
    """Radix-trie node. Owns a contiguous edge of tokens and their per-layer KV slice.

    The root is a sentinel with depth 0, no parent, and no edge. Every non-root
    node carries `edge_tokens` (the range from parent.depth to self.depth) and,
    for positionally-sliceable layers, per-layer key/value slices covering that
    range. Non-sliceable layers (ArraysCache/RotatingKVCache) aren't stored in
    the trie — their state lives on the leaf, recovered via snapshot.

    Sharing: a single node is reached by multiple leaves iff their prompts all
    share its full prefix. `ref_count` = number of leaves whose terminal node is
    this node or a descendant.
    """

    __slots__ = (
        "parent",
        "edge_tokens",
        "depth",
        "edge_keys",
        "edge_values",
        "snapshot",
        "media_regions",
        "children",
        "ref_count",
    )

    def __init__(
        self,
        parent: "_TrieNode | None",
        edge_tokens: mx.array,
        depth: int,
        edge_keys: "list[Any] | None",
        edge_values: "list[Any] | None",
        snapshot: CacheSnapshot | None,
        media_regions: list["MediaRegion"],
    ):
        self.parent = parent
        self.edge_tokens = edge_tokens
        self.depth = depth
        # Per-layer key/value slices covering the edge's token range, or None
        # for layers whose cache type isn't positionally sliceable (those live
        # on the leaf). Entries may also be a list[mx.array] for QuantizedKVCache
        # layers, where each layer holds a list of (quantized, scales, biases).
        self.edge_keys: list[Any] | None = edge_keys
        self.edge_values: list[Any] | None = edge_values
        self.snapshot = snapshot
        self.media_regions = media_regions
        self.children: dict[int, "_TrieNode"] = {}
        self.ref_count = 0

    @property
    def edge_length(self) -> int:
        return int(self.edge_tokens.shape[0])

    def edge_nbytes(self) -> int:
        total = 0
        if self.edge_keys is not None:
            for k in self.edge_keys:
                total += _array_like_nbytes(k)
        if self.edge_values is not None:
            for v in self.edge_values:
                total += _array_like_nbytes(v)
        if self.snapshot is not None:
            for state in self.snapshot.states:
                if state is None:
                    continue
                total += _cache_object_nbytes(state)
        return total


class _Leaf:
    """Terminal session entry. Anchored at a trie node representing its full prompt."""

    __slots__ = (
        "leaf_id",
        "node",
        "full_tokens",
        "prefill_tps",
        "last_used",
        "pinned",
        "leaf_layer_caches",
        "leaf_snapshots",
    )

    def __init__(
        self,
        leaf_id: int,
        node: _TrieNode,
        full_tokens: mx.array,
        prefill_tps: float,
        last_used: int,
        pinned: bool,
        leaf_layer_caches: list[object | None],
        leaf_snapshots: list[CacheSnapshot] | None,
    ):
        self.leaf_id = leaf_id
        self.node = node
        self.full_tokens = full_tokens
        self.prefill_tps = prefill_tps
        self.last_used = last_used
        self.pinned = pinned
        # Full per-layer cache for non-sliceable layer types (ArraysCache/RotatingKVCache).
        # Entry is None for sliceable layers (those live in the trie).
        self.leaf_layer_caches = leaf_layer_caches
        # Snapshots supplied by the generator, used to restore ArraysCache/RotatingKVCache
        # layers at a given depth during partial-prefix hits.
        self.leaf_snapshots = leaf_snapshots


class _CacheProxy:
    """Dict-like proxy giving per-leaf access to a materialized full KV cache.

    Materializes on __getitem__ so callers doing `prefix_cache.caches[leaf_id]`
    get the combined trie-path + leaf-only layers as a single KVCacheType.
    """

    def __init__(
        self, leaf_ids: list[int], materialize: "Callable[[int], KVCacheType]"
    ):
        self._leaf_ids = leaf_ids
        self._materialize = materialize

    def __len__(self) -> int:
        return len(self._leaf_ids)

    def __contains__(self, leaf_id: object) -> bool:
        return leaf_id in self._leaf_ids

    def __getitem__(self, leaf_id: int) -> KVCacheType:
        return self._materialize(leaf_id)

    def keys(self) -> list[int]:
        return list(self._leaf_ids)


class KVPrefixCache:
    """Radix-trie KV prefix cache with shared storage across sessions.

    Sessions that share a token prefix share the underlying per-layer K/V
    tensors for that prefix. A session is a leaf; an internal node holds
    storage for a range of tokens common to ≥1 descendant leaves.

    Sliceable layers (KVCache/QuantizedKVCache/non-rotated RotatingKVCache)
    are dedup'd via trie edges. Non-sliceable layers (ArraysCache/SSM,
    post-rotation RotatingKVCache) keep a full-copy on each leaf and restore
    via the caller-supplied snapshot list — same as before.

    External contract preserved:
      - pin(leaf_id) / clear() / add_kv_cache / update_kv_cache / get_kv_cache
      - .prompts (dict[leaf_id, mx.array])
      - .caches (dict-like proxy, materializes on access)
      - .prefill_tps (dict[leaf_id, float])
      - get_memory_used_percentage()
    """

    def __init__(
        self,
        group: mx.distributed.Group | None,
        max_sessions: int | None = None,
        max_bytes: int | None = None,
        max_kv_tokens: int | None = None,
    ):
        self._root = _TrieNode(
            parent=None,
            edge_tokens=mx.array([], dtype=mx.int32),
            depth=0,
            edge_keys=None,
            edge_values=None,
            snapshot=None,
            media_regions=[],
        )
        self._leaves: dict[int, _Leaf] = {}
        self._next_leaf_id: int = 0
        self._access_counter: int = 0
        self._group = group
        self._max_sessions = max_sessions
        self._max_bytes = max_bytes
        self._max_kv_tokens = max_kv_tokens

    # ── Dict-like views for external callers / tests ───────────────────────
    @property
    def prompts(self) -> dict[int, mx.array]:
        return {lid: leaf.full_tokens for lid, leaf in self._leaves.items()}

    @property
    def prefill_tps(self) -> dict[int, float]:
        return {lid: leaf.prefill_tps for lid, leaf in self._leaves.items()}

    @property
    def caches(self) -> _CacheProxy:
        return _CacheProxy(list(self._leaves.keys()), self._materialize_full_leaf_cache)

    # ── Public API ─────────────────────────────────────────────────────────
    def pin(self, leaf_id: int) -> None:
        leaf = self._leaves.get(leaf_id)
        if leaf is not None:
            leaf.pinned = True

    def clear(self) -> None:
        self._leaves.clear()
        self._root.children.clear()
        self._root.ref_count = 0
        self._next_leaf_id = 0

    def add_kv_cache(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        ssm_snapshots: list[CacheSnapshot] | None = None,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ) -> int:
        """Insert a new session. Splits existing edges at divergence points.

        Returns the newly assigned leaf id.
        """
        self._evict_if_needed()

        sliceable_mask = _sliceable_layer_mask(cache)
        node = self._insert_path(
            prompt_tokens=prompt_tokens,
            cache=cache,
            sliceable_mask=sliceable_mask,
            ssm_snapshots=ssm_snapshots,
            media_regions=media_regions or [],
        )

        # Build leaf-only per-layer cache for non-sliceable layers; sliceable
        # layers live in the trie and are deepcopy'd on materialize.
        leaf_layer_caches = _extract_non_sliceable_layers(cache, sliceable_mask)

        leaf_id = self._next_leaf_id
        self._next_leaf_id += 1
        self._access_counter += 1
        leaf = _Leaf(
            leaf_id=leaf_id,
            node=node,
            full_tokens=prompt_tokens,
            prefill_tps=prefill_tps,
            last_used=self._access_counter,
            pinned=False,
            leaf_layer_caches=leaf_layer_caches,
            leaf_snapshots=list(ssm_snapshots) if ssm_snapshots else None,
        )
        self._leaves[leaf_id] = leaf
        self._increment_ref_count(node)

        logger.info(f"KV cache added: {len(prompt_tokens)} tokens (leaf {leaf_id})")
        return leaf_id

    def update_kv_cache(
        self,
        leaf_id: int,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        snapshots: list[CacheSnapshot] | None,
        restore_pos: int,
        media_regions: list["MediaRegion"] | None = None,
        prefill_tps: float = 0.0,
    ) -> None:
        """Extend an existing leaf with new suffix tokens and refreshed cache.

        Detaches the old leaf's path (decrementing ref counts, freeing orphaned
        ancestors) and re-inserts with the new prompt. This covers the common
        case of a continued conversation where the same leaf grows.
        """
        leaf = self._leaves.get(leaf_id)
        if leaf is None:
            logger.info(
                f"KV cache update: leaf {leaf_id} not found, falling back to add"
            )
            self.add_kv_cache(
                prompt_tokens=prompt_tokens,
                cache=cache,
                ssm_snapshots=snapshots,
                media_regions=media_regions,
                prefill_tps=prefill_tps,
            )
            return

        # Merge snapshots: keep old ones at or before the restore point, add new.
        merged: list[CacheSnapshot] = []
        if leaf.leaf_snapshots:
            merged = [s for s in leaf.leaf_snapshots if s.token_count <= restore_pos]
        if snapshots:
            merged.extend(snapshots)

        # Release old path.
        was_pinned = leaf.pinned
        self._decrement_ref_count(leaf.node)
        del self._leaves[leaf_id]

        # Re-insert under the same leaf_id so external references stay valid.
        sliceable_mask = _sliceable_layer_mask(cache)
        node = self._insert_path(
            prompt_tokens=prompt_tokens,
            cache=cache,
            sliceable_mask=sliceable_mask,
            ssm_snapshots=merged or None,
            media_regions=media_regions or [],
        )
        leaf_layer_caches = _extract_non_sliceable_layers(cache, sliceable_mask)
        self._access_counter += 1
        new_leaf = _Leaf(
            leaf_id=leaf_id,
            node=node,
            full_tokens=prompt_tokens,
            prefill_tps=prefill_tps,
            last_used=self._access_counter,
            pinned=was_pinned,
            leaf_layer_caches=leaf_layer_caches,
            leaf_snapshots=merged or None,
        )
        self._leaves[leaf_id] = new_leaf
        self._increment_ref_count(node)

        logger.info(f"KV cache updated (leaf {leaf_id}): {len(prompt_tokens)} tokens")

    def get_kv_cache(
        self,
        model: Model,
        prompt_tokens: mx.array,
        media_regions: list["MediaRegion"] | None = None,
    ) -> tuple[KVCacheType, mx.array, int | None, bool]:
        """Find the longest shared prefix in the trie and return a cache for it.

        Returns (cache, remaining_tokens, matched_leaf_id, is_exact).
        """
        max_length = int(prompt_tokens.shape[0])
        query_regions = media_regions or []

        match_node, match_length = self._longest_prefix_match(
            prompt_tokens, query_regions
        )
        if match_length == 0:
            return (
                make_kv_cache(model, max_kv_size=self._max_kv_tokens),
                prompt_tokens,
                None,
                False,
            )

        # Pick a leaf in/under the matched node for non-sliceable layer recovery.
        # If the match truncated mid-edge (match_node.depth < match_length), any
        # leaf in match_node's subtree whose path passes through the truncation
        # point works — pick the most recent.
        donor_leaf = self._pick_leaf_under(match_node)
        if donor_leaf is None:
            # Shouldn't happen: every node has ref_count > 0 ⇒ at least one leaf.
            return (
                make_kv_cache(model, max_kv_size=self._max_kv_tokens),
                prompt_tokens,
                None,
                False,
            )

        is_exact = match_length >= max_length - 1
        # For exact hits on non-SSM models we keep the last token out so the
        # generator always has ≥1 token to feed. For non-sliceable layers this
        # is constrained by snapshot availability.
        sliceable_mask = donor_leaf.leaf_layer_caches
        has_non_sliceable = any(c is not None for c in sliceable_mask)
        target = (
            (max_length - 1) if is_exact and not has_non_sliceable else match_length
        )

        restore_pos, snapshot = self._resolve_restore_position(
            donor_leaf, target, has_non_sliceable
        )
        if has_non_sliceable and snapshot is None:
            # No usable snapshot for SSM/rotating layers — force a full prefill.
            return (
                make_kv_cache(model, max_kv_size=self._max_kv_tokens),
                prompt_tokens,
                None,
                False,
            )

        # Materialize a cache up to restore_pos: concat trie slices for sliceable
        # layers, deepcopy + trim for leaf-held non-sliceable layers.
        materialized = self._materialize_cache_to_depth(
            donor_leaf=donor_leaf,
            target_depth=restore_pos,
            snapshot=snapshot,
        )

        self._access_counter += 1
        donor_leaf.last_used = self._access_counter
        remaining = prompt_tokens[restore_pos:]
        return materialized, remaining, donor_leaf.leaf_id, is_exact

    def get_memory_used_percentage(self) -> float:
        local_pressure: float = get_memory_used_percentage()

        if self._group is None:
            return local_pressure

        all_pressure = mx.distributed.all_gather(
            mx.array([local_pressure], dtype=mx.float32),
            group=self._group,
        )
        max_pressure = float(mx.max(all_pressure).item())
        return max_pressure

    @staticmethod
    def _validate_media_match(
        match_length: int,
        cached_regions: list["MediaRegion"],
        query_regions: list["MediaRegion"],
    ) -> int:
        """Truncate a token-level prefix match at the first cached media region
        whose content_hash diverges from the query's region at the same
        position. Kept as a static method so existing unit tests can exercise
        this logic directly without constructing a full trie.
        """
        if not cached_regions:
            return match_length

        query_by_start: dict[int, "MediaRegion"] = {
            r.start_pos: r for r in query_regions
        }
        for cached_r in cached_regions:
            if cached_r.start_pos >= match_length:
                break
            query_r = query_by_start.get(cached_r.start_pos)
            if query_r is None:
                continue
            if query_r.content_hash != cached_r.content_hash:
                return cached_r.start_pos
        return match_length

    # ── Trie mechanics ─────────────────────────────────────────────────────
    def _insert_path(
        self,
        prompt_tokens: mx.array,
        cache: KVCacheType,
        sliceable_mask: list[bool],
        ssm_snapshots: list[CacheSnapshot] | None,
        media_regions: list["MediaRegion"],
    ) -> _TrieNode:
        """Walk the trie from root, splitting edges at divergence, and attach a
        terminal node at depth == len(prompt_tokens). Returns that terminal.
        """
        prompt_np = _tokens_to_np(prompt_tokens)
        total_length = int(prompt_np.shape[0])
        node = self._root
        pos = 0

        while pos < total_length:
            next_token = int(prompt_np[pos])
            child = node.children.get(next_token)
            if child is None:
                new_node = self._build_edge_node(
                    parent=node,
                    prompt_tokens=prompt_tokens,
                    start=pos,
                    end=total_length,
                    cache=cache,
                    sliceable_mask=sliceable_mask,
                    ssm_snapshots=ssm_snapshots,
                    media_regions=media_regions,
                )
                node.children[next_token] = new_node
                return new_node

            child_edge_np = _tokens_to_np(child.edge_tokens)
            match_in_edge = _np_prefix_length(child_edge_np, prompt_np, pos)

            if match_in_edge == child.edge_length:
                node = child
                pos += match_in_edge
                continue

            split_node = self._split_edge(
                child, match_in_edge, cache, sliceable_mask, ssm_snapshots
            )
            new_depth = node.depth + match_in_edge
            pos = new_depth
            if pos >= total_length:
                return split_node
            new_leaf_node = self._build_edge_node(
                parent=split_node,
                prompt_tokens=prompt_tokens,
                start=pos,
                end=total_length,
                cache=cache,
                sliceable_mask=sliceable_mask,
                ssm_snapshots=ssm_snapshots,
                media_regions=media_regions,
            )
            next_token = int(prompt_np[pos])
            split_node.children[next_token] = new_leaf_node
            return new_leaf_node

        return node

    def _build_edge_node(
        self,
        parent: _TrieNode,
        prompt_tokens: mx.array,
        start: int,
        end: int,
        cache: KVCacheType,
        sliceable_mask: list[bool],
        ssm_snapshots: list[CacheSnapshot] | None,
        media_regions: list["MediaRegion"] | None = None,
    ) -> _TrieNode:
        edge_tokens = prompt_tokens[start:end]
        edge_keys, edge_values = _slice_sliceable_layers(
            cache, sliceable_mask, start, end
        )
        snapshot = _snapshot_at(ssm_snapshots, end) if ssm_snapshots else None
        regions = _media_regions_in_range(media_regions or [], start, end)
        return _TrieNode(
            parent=parent,
            edge_tokens=edge_tokens,
            depth=end,
            edge_keys=edge_keys,
            edge_values=edge_values,
            snapshot=snapshot,
            media_regions=regions,
        )

    def _split_edge(
        self,
        child: _TrieNode,
        split_offset: int,
        cache: KVCacheType,
        sliceable_mask: list[bool],
        ssm_snapshots: list[CacheSnapshot] | None,
    ) -> _TrieNode:
        """Split `child` so that the first `split_offset` tokens of its edge
        become a new internal node, and the original child becomes a grandchild
        carrying the remaining tokens.

        Returns the new internal node.
        """
        assert child.parent is not None
        parent = child.parent
        child_edge_np = _tokens_to_np(child.edge_tokens)
        first_head_token = int(child_edge_np[0])

        head_tokens = child.edge_tokens[:split_offset]
        tail_tokens = child.edge_tokens[split_offset:]

        head_keys, head_values, tail_keys, tail_values = _split_edge_slices(
            child.edge_keys, child.edge_values, split_offset
        )

        split_depth = child.depth - child.edge_length + split_offset
        new_snapshot = (
            _snapshot_at(ssm_snapshots, split_depth) if ssm_snapshots else None
        )

        head_regions = [r for r in child.media_regions if r.end_pos <= split_depth]
        tail_regions = [r for r in child.media_regions if r.start_pos >= split_depth]

        new_internal = _TrieNode(
            parent=parent,
            edge_tokens=head_tokens,
            depth=split_depth,
            edge_keys=head_keys,
            edge_values=head_values,
            snapshot=new_snapshot,
            media_regions=head_regions,
        )
        new_internal.ref_count = child.ref_count

        first_tail_token = int(child_edge_np[split_offset])
        parent.children[first_head_token] = new_internal
        new_internal.children[first_tail_token] = child
        child.parent = new_internal
        child.edge_tokens = tail_tokens
        child.edge_keys = tail_keys
        child.edge_values = tail_values
        child.media_regions = tail_regions
        return new_internal

    def _longest_prefix_match(
        self,
        prompt_tokens: mx.array,
        query_regions: list["MediaRegion"],
    ) -> tuple[_TrieNode, int]:
        """Walk from root, return (deepest matching node, match length). The
        match is truncated by any media-region content-hash mismatch found
        along the way.
        """
        prompt_np = _tokens_to_np(prompt_tokens)
        total_length = int(prompt_np.shape[0])
        if total_length == 0:
            return self._root, 0

        node = self._root
        matched = 0
        query_by_start: dict[int, "MediaRegion"] = {
            r.start_pos: r for r in query_regions
        }

        while matched < total_length:
            next_token = int(prompt_np[matched])
            child = node.children.get(next_token)
            if child is None:
                break

            child_edge_np = _tokens_to_np(child.edge_tokens)
            edge_match = _np_prefix_length(child_edge_np, prompt_np, matched)
            if edge_match == 0:
                break

            new_matched = matched + edge_match

            for cached_r in child.media_regions:
                if cached_r.start_pos >= new_matched:
                    break
                query_r = query_by_start.get(cached_r.start_pos)
                if query_r is None:
                    continue
                if query_r.content_hash != cached_r.content_hash:
                    logger.info(
                        f"Media region mismatch at pos {cached_r.start_pos}: "
                        f"cached={cached_r.content_hash[:12]}... "
                        f"query={query_r.content_hash[:12]}... — "
                        f"truncating match to {cached_r.start_pos}"
                    )
                    return node, cached_r.start_pos

            if edge_match < child.edge_length:
                return node, new_matched

            node = child
            matched = new_matched

        return node, matched

    def _pick_leaf_under(self, node: _TrieNode) -> _Leaf | None:
        """Return the most-recently-used leaf whose node is in node's subtree."""
        best: _Leaf | None = None
        for leaf in self._leaves.values():
            if not _is_ancestor(node, leaf.node):
                continue
            if best is None or leaf.last_used > best.last_used:
                best = leaf
        return best

    def _resolve_restore_position(
        self,
        donor_leaf: _Leaf,
        target: int,
        has_non_sliceable: bool,
    ) -> tuple[int, CacheSnapshot | None]:
        """For models with non-sliceable layers, pick a snapshot at or before
        target and return its depth. For purely sliceable models, return target
        as-is.
        """
        if not has_non_sliceable:
            return target, None
        snaps = donor_leaf.leaf_snapshots
        if not snaps:
            return 0, None
        snap = _find_nearest_snapshot(snaps, target)
        if snap is None:
            return 0, None
        return snap.token_count, snap

    def _materialize_cache_to_depth(
        self,
        donor_leaf: _Leaf,
        target_depth: int,
        snapshot: CacheSnapshot | None,
    ) -> KVCacheType:
        """Build a fresh per-layer KV cache whose contents cover tokens
        [0, target_depth) for sliceable layers and match the snapshot (at
        snapshot.token_count <= target_depth) for non-sliceable layers.
        """
        # Walk root→donor_leaf.node collecting edges up to target_depth.
        path = _collect_path(self._root, donor_leaf.node)
        num_layers = len(donor_leaf.leaf_layer_caches)

        per_layer_keys: list[list[Any]] = [[] for _ in range(num_layers)]
        per_layer_values: list[list[Any]] = [[] for _ in range(num_layers)]

        depth_so_far = 0
        for edge in path:
            if depth_so_far >= target_depth:
                break
            take = min(edge.edge_length, target_depth - depth_so_far)
            if edge.edge_keys is not None and edge.edge_values is not None:
                for layer_idx in range(num_layers):
                    k = edge.edge_keys[layer_idx]
                    v = edge.edge_values[layer_idx]
                    if not _has_tokens(k):
                        continue
                    if take < edge.edge_length:
                        k = _slice_seq_axis(k, 0, take)
                        v = _slice_seq_axis(v, 0, take)
                    per_layer_keys[layer_idx].append(k)
                    per_layer_values[layer_idx].append(v)
            depth_so_far += take

        # Build per-layer caches.
        new_cache: KVCacheType = []
        for layer_idx in range(num_layers):
            leaf_layer = donor_leaf.leaf_layer_caches[layer_idx]
            if leaf_layer is not None:
                # Non-sliceable layer: deepcopy leaf's full cache, then trim to snapshot.
                c = deepcopy(leaf_layer)
                if snapshot is not None:
                    snap_state = snapshot.states[layer_idx]
                    if snap_state is not None:
                        c = deepcopy(snap_state)
                new_cache.append(c)  # type: ignore[arg-type]
                continue

            ks = per_layer_keys[layer_idx]
            vs = per_layer_values[layer_idx]
            cache_entry = KVCache()
            if ks:
                concat_k = _concat_seq_axis(ks)
                concat_v = _concat_seq_axis(vs)
                # Detach to break MLX graph references; keeps the stored slice
                # safe from caller mutation (generation writes in place).
                cache_entry.keys = _detached_copy(concat_k)
                cache_entry.values = _detached_copy(concat_v)
                cache_entry.offset = int(cache_entry.keys.shape[2])
            new_cache.append(cache_entry)

        return new_cache

    def _materialize_full_leaf_cache(self, leaf_id: int) -> KVCacheType:
        """Materialize the full cache for a leaf (depth = leaf.node.depth).

        Used by the .caches proxy (tests + diagnostics).
        """
        leaf = self._leaves[leaf_id]
        target_depth = leaf.node.depth
        return self._materialize_cache_to_depth(
            donor_leaf=leaf, target_depth=target_depth, snapshot=None
        )

    # ── Ref counting + eviction ────────────────────────────────────────────
    def _increment_ref_count(self, node: _TrieNode) -> None:
        cur: _TrieNode | None = node
        while cur is not None:
            cur.ref_count += 1
            cur = cur.parent

    def _decrement_ref_count(self, node: _TrieNode) -> None:
        cur: _TrieNode | None = node
        while cur is not None:
            cur.ref_count -= 1
            parent = cur.parent
            if cur.ref_count <= 0 and parent is not None:
                if cur.edge_length > 0:
                    first_token = int(_tokens_to_np(cur.edge_tokens)[0])
                    parent.children.pop(first_token, None)
                cur.edge_keys = None
                cur.edge_values = None
                cur.snapshot = None
                cur.children.clear()
            cur = parent

    def _total_bytes(self) -> int:
        total = 0
        stack: list[_TrieNode] = [self._root]
        while stack:
            n = stack.pop()
            total += n.edge_nbytes()
            stack.extend(n.children.values())
        for leaf in self._leaves.values():
            for layer in leaf.leaf_layer_caches:
                if layer is None:
                    continue
                for attr in ("keys", "values", "state"):
                    t = getattr(layer, attr, None)
                    if t is None:
                        continue
                    if isinstance(t, list):
                        for a in t:
                            nb = getattr(a, "nbytes", None)
                            if nb is not None:
                                total += int(nb)
                    else:
                        nb = getattr(t, "nbytes", None)
                        if nb is not None:
                            total += int(nb)
        return total

    def _evict_lru_once(self, reason: str) -> bool:
        candidates = [leaf for leaf in self._leaves.values() if not leaf.pinned]
        if not candidates:
            return False
        lru = min(candidates, key=lambda leaf: leaf.last_used)
        evicted_tokens = int(lru.full_tokens.shape[0])
        self._decrement_ref_count(lru.node)
        del self._leaves[lru.leaf_id]
        logger.info(
            f"KV cache evicted leaf {lru.leaf_id} ({evicted_tokens} tokens) — {reason}"
        )
        return True

    def _evict_if_needed(self, *, reserve_slot: bool = True) -> None:
        """Evict LRU leaves until caps are satisfied.

        `reserve_slot` is True when called at the start of add_kv_cache so
        that a new leaf fits within max_sessions — i.e. we evict down to
        max_sessions - 1 before the caller appends.
        """
        if not self._leaves:
            return
        evicted_any = False
        while self._leaves and self.get_memory_used_percentage() > _MEMORY_THRESHOLD:
            if not self._evict_lru_once("memory pressure"):
                break
            evicted_any = True
        if self._max_sessions is not None:
            limit = self._max_sessions - (1 if reserve_slot else 0)
            limit = max(limit, 0)
            while len(self._leaves) > limit:
                if not self._evict_lru_once(f"session cap {self._max_sessions}"):
                    break
                evicted_any = True
        if self._max_bytes is not None:
            while self._leaves and self._total_bytes() > self._max_bytes:
                if not self._evict_lru_once(f"byte cap {self._max_bytes}"):
                    break
                evicted_any = True
        if evicted_any:
            gc.collect()
            mx.clear_cache()


# ── Module-level helpers used by the trie ────────────────────────────────────


def _tokens_to_np(tokens: mx.array) -> np.ndarray:
    """Convert a 1-D mx.array of token ids to a numpy array on the host.

    Single `.item()` calls inside inner loops hammer the MLX evaluator; we
    materialise the whole array once per traversal instead. Stored edge token
    arrays are short and repeated traversal is CPU-only work.
    """
    return np.asarray(tokens)


def _np_prefix_length(
    edge_tokens: np.ndarray, query_tokens: np.ndarray, query_offset: int
) -> int:
    """Length of the common prefix between `edge_tokens` and the query starting
    at `query_offset`, on host-side numpy arrays.
    """
    edge_len = int(edge_tokens.shape[0])
    query_remaining = int(query_tokens.shape[0]) - query_offset
    n = min(edge_len, query_remaining)
    if n <= 0:
        return 0
    equal = edge_tokens[:n] == query_tokens[query_offset : query_offset + n]
    # First False position; if all match, it's n.
    if bool(equal.all()):
        return n
    first_mismatch = int(np.argmin(equal))
    return first_mismatch


def _sliceable_layer_mask(cache: KVCacheType) -> list[bool]:
    """True at index i iff layer i is a positionally-sliceable KV cache type
    whose K/V tensors can be stored as edge slices in the trie.
    """
    mask: list[bool] = []
    for c in cache:
        if isinstance(c, (ArraysCache, RotatingKVCache)):
            mask.append(False)
        elif isinstance(c, KVCache):
            mask.append(True)
        elif isinstance(c, QuantizedKVCache):
            # QuantizedKVCache holds group-quantized tensors; safe to slice at
            # token offsets because the sequence axis is independent of the
            # quant groups along the last dim.
            mask.append(True)
        else:
            # Unknown cache type: keep it on the leaf to be safe.
            mask.append(False)
    return mask


_EMPTY_PLACEHOLDER: Any = mx.array([], dtype=mx.float32)


def _has_tokens(value: Any) -> bool:
    """True if this per-layer slice placeholder actually carries any tokens.

    Non-sliceable layers receive an empty mx.array as a sentinel to keep the
    per-layer list aligned with model layer indices.
    """
    if value is None:
        return False
    if isinstance(value, list):
        return bool(value) and _has_tokens(value[0])  # type: ignore[reportUnknownArgumentType]
    shape = getattr(value, "shape", None)
    if shape is None:
        return False
    # Real slices are [B, H, L, D]; placeholder is shape (0,).
    if len(shape) < 3:  # type: ignore[reportUnknownArgumentType]
        return False
    return int(shape[-2]) > 0  # type: ignore[reportAny]


def _slice_seq_axis(value: Any, start: int, end: int) -> Any:
    """Slice either an mx.array [B, H, S, D] or a list of them (QuantizedKVCache)
    along the sequence axis. `end` larger than the actual sequence is clamped
    to the tensor length — MLX requires 32-bit slice bounds.
    """
    if isinstance(value, list):
        return [_slice_seq_axis(a, start, end) for a in value]  # type: ignore[reportUnknownArgumentType]
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:  # type: ignore[reportUnknownArgumentType]
        seq_len = int(shape[-2])  # type: ignore[reportAny]
        end = min(end, seq_len)
    return value[..., start:end, :]


def _concat_seq_axis(values: "list[Any]") -> Any:
    """Concatenate slices along the sequence axis. Scalar-list of length 1 just
    returns the sole entry without an mx.concatenate call.
    """
    if len(values) == 1:
        return values[0]
    if isinstance(values[0], list):
        # QuantizedKVCache: concatenate per (quant, scale, bias) slot.
        num_slots = len(values[0])  # type: ignore[reportUnknownArgumentType]
        out: list[Any] = []
        for slot in range(num_slots):
            out.append(
                mx.concatenate([v[slot] for v in values], axis=-2)  # type: ignore[reportUnknownVariableType,reportUnknownArgumentType]
            )
        return out
    return mx.concatenate(values, axis=2)


def _slice_sliceable_layers(
    cache: KVCacheType,
    sliceable_mask: list[bool],
    start: int,
    end: int,
) -> tuple[list[Any] | None, list[Any] | None]:
    """Extract the [start:end) slice of keys/values per layer for sliceable layers.

    Returns (None, None) if no layers are sliceable (pure SSM model).
    """
    if not any(sliceable_mask):
        return None, None
    keys: list[Any] = []
    values: list[Any] = []
    for i, entry in enumerate(cache):
        if not sliceable_mask[i]:
            keys.append(_EMPTY_PLACEHOLDER)
            values.append(_EMPTY_PLACEHOLDER)
            continue
        k, v = _slice_layer_kv(entry, start, end)
        keys.append(k)
        values.append(v)
    return keys, values


def _slice_layer_kv(cache_entry: Any, start: int, end: int) -> tuple[Any, Any]:
    """Slice a single layer's K/V along the sequence axis to [start:end).

    Returns detached (numpy-round-tripped) copies so mutation of the source
    cache by the generator doesn't corrupt the stored slice.
    """
    k_full = getattr(cache_entry, "keys", None)
    v_full = getattr(cache_entry, "values", None)
    if k_full is None or v_full is None:
        return _EMPTY_PLACEHOLDER, _EMPTY_PLACEHOLDER

    offset = getattr(cache_entry, "offset", None)
    if offset is not None:
        end = min(end, int(offset))  # type: ignore[reportAny]
    start = max(0, min(start, end))

    if isinstance(k_full, list):
        k_slice = [_detached_copy(a[..., start:end, :]) for a in k_full]  # type: ignore[reportUnknownVariableType]
        v_slice = [_detached_copy(a[..., start:end, :]) for a in v_full]  # type: ignore[reportUnknownVariableType]
        return k_slice, v_slice
    k_slice = _detached_copy(k_full[..., start:end, :])
    v_slice = _detached_copy(v_full[..., start:end, :])
    return k_slice, v_slice


def _split_edge_slices(
    edge_keys: list[Any] | None,
    edge_values: list[Any] | None,
    offset: int,
) -> tuple[
    list[Any] | None,
    list[Any] | None,
    list[Any] | None,
    list[Any] | None,
]:
    """Split per-layer K/V edge slices at `offset` along the sequence axis.

    Returns (head_keys, head_values, tail_keys, tail_values). Placeholder
    entries (for non-sliceable layers) are passed through on both sides.
    """
    if edge_keys is None or edge_values is None:
        return None, None, None, None

    head_keys: list[Any] = []
    head_values: list[Any] = []
    tail_keys: list[Any] = []
    tail_values: list[Any] = []
    for k, v in zip(edge_keys, edge_values, strict=True):
        if not _has_tokens(k):
            head_keys.append(k)
            head_values.append(v)
            tail_keys.append(k)
            tail_values.append(v)
            continue
        head_keys.append(_detached_from(_slice_seq_axis(k, 0, offset)))
        head_values.append(_detached_from(_slice_seq_axis(v, 0, offset)))
        tail_keys.append(_detached_from(_slice_seq_axis(k, offset, int(1 << 31))))
        tail_values.append(_detached_from(_slice_seq_axis(v, offset, int(1 << 31))))
    return head_keys, head_values, tail_keys, tail_values


def _detached_from(value: Any) -> Any:
    """Detach either an mx.array or a list of mx.arrays."""
    if isinstance(value, list):
        return [_detached_copy(a) for a in value]  # type: ignore[reportUnknownVariableType]
    return _detached_copy(value)


def _extract_non_sliceable_layers(
    cache: KVCacheType, sliceable_mask: list[bool]
) -> list[object | None]:
    """Deepcopy only the non-sliceable layers onto the leaf; sliceable layers
    return None (their storage lives in the trie).
    """
    out: list[object | None] = []
    for i, c in enumerate(cache):
        if sliceable_mask[i]:
            out.append(None)
        else:
            out.append(deepcopy(c))
    return out


def _snapshot_at(
    snapshots: list[CacheSnapshot] | None, depth: int
) -> CacheSnapshot | None:
    """Return the snapshot whose token_count == depth, if any."""
    if not snapshots:
        return None
    for s in snapshots:
        if s.token_count == depth:
            return s
    return None


def _media_regions_in_range(
    regions: list["MediaRegion"], start: int, end: int
) -> list["MediaRegion"]:
    """Regions entirely within [start, end). Regions that straddle the boundary
    are retained on whichever side holds their start_pos — the split logic
    handles boundary-straddling regions explicitly.
    """
    return [r for r in regions if r.start_pos >= start and r.end_pos <= end]


def _is_ancestor(ancestor: _TrieNode, descendant: _TrieNode) -> bool:
    cur: _TrieNode | None = descendant
    while cur is not None:
        if cur is ancestor:
            return True
        cur = cur.parent
    return False


def _collect_path(root: _TrieNode, node: _TrieNode) -> list[_TrieNode]:
    """Return the chain of edges from root's first child down to `node`,
    excluding the root sentinel itself.
    """
    path: list[_TrieNode] = []
    cur: _TrieNode | None = node
    while cur is not None and cur is not root:
        path.append(cur)
        cur = cur.parent
    path.reverse()
    return path


def trim_cache(
    cache: KVCacheType,
    num_tokens: int,
    snapshot: CacheSnapshot | None = None,
) -> None:
    for i, c in enumerate(cache):
        if isinstance(c, (ArraysCache, RotatingKVCache)):
            if snapshot is not None and snapshot.states[i] is not None:
                cache[i] = deepcopy(snapshot.states[i])  # type: ignore
            else:
                c.state = [None] * len(c.state)
        else:
            c.trim(num_tokens)


def encode_prompt(tokenizer: TokenizerWrapper, prompt: str) -> mx.array:
    """Encode a prompt string to token array.

    For chat-templated prompts (which have their own structure markers like
    <|im_user|>, <|im_middle|>, etc.), we should NOT add BOS/EOS tokens as
    that would corrupt the prompt structure.
    """
    # Chat templates define their own structure - don't add BOS/EOS
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return mx.array(prompt_tokens)


def _entry_length(
    c: KVCache | RotatingKVCache | QuantizedKVCache | ArraysCache | CacheList,
) -> int:
    # Use .offset attribute which KVCache types have (len() not implemented in older QuantizedKVCache).
    if hasattr(c, "offset"):
        return c.offset
    # For CacheList
    if hasattr(c, "size"):
        return int(c.size())  # type: ignore
    return 0


def cache_length(cache: KVCacheType) -> int:
    """Get the number of tokens in a KV cache."""
    return max((_entry_length(c) for c in cache), default=0)


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0

    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)  # stays 1 until first mismatch, then 0 forever
    return int(mx.sum(prefix_mask).item())


def get_available_memory() -> Memory:
    mem: int = psutil.virtual_memory().available
    return Memory.from_bytes(mem)


def get_memory_used_percentage() -> float:
    mem = psutil.virtual_memory()
    # percent is 0-100
    return float(mem.percent / 100)


def _model_is_pipeline_parallel(model: Model) -> bool:
    """True iff the model has pipeline-parallel layer wrappers installed.

    Only the PP path is safe to combine with QuantizedKVCache right now:
    the single-node BatchGenerator code path in mlx-lm calls
    ``_merge_caches`` on every step (even for a single in-flight request),
    and QuantizedKVCache does not implement ``merge``. Attempting to use
    a quantized cache in that path crashes with::

        <class 'mlx_lm.models.cache.QuantizedKVCache'> does not yet
        support batching with history

    Detecting PP mode by layer type is cheap and avoids threading the
    distributed group through every cache call site.
    """
    try:
        from exo.worker.engines.mlx.auto_parallel import (
            PipelineFirstLayer,
            PipelineLastLayer,
        )
    except Exception:
        return False
    layers = getattr(model, "layers", None)
    if layers is None:
        return False
    for layer in layers:  # type: ignore[reportUnknownVariableType]
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer)):
            return True
    return False


def make_kv_cache(
    model: Model, max_kv_size: int | None = None, keep: int = 0
) -> KVCacheType:
    assert hasattr(model, "layers")

    if hasattr(model, "make_cache"):
        caches: KVCacheType = model.make_cache()  # type: ignore
        # Per-instance per-request token cap: wrap plain KVCache entries in
        # RotatingKVCache so a single request can't grow KV memory unbounded.
        # Skips quantization/turboquant/step-size paths since they target full
        # KVCache only and we're replacing those entries.
        if max_kv_size is not None:
            replaced = 0
            for i, c in enumerate(caches):
                if isinstance(c, KVCache):
                    caches[i] = RotatingKVCache(max_size=max_kv_size, keep=keep)
                    replaced += 1
            if replaced:
                logger.info(
                    f"Capped KV cache at max_kv_size={max_kv_size} for {replaced}/{len(caches)} layers"
                )
            return caches
        if TURBOQUANT_ENABLED:
            from exo.worker.engines.mlx.turboquant_cache import (
                TurboQuantConfig,
                TurboQuantKVCache,
                patch_attention_dispatch,
            )

            patch_attention_dispatch()
            tq_config = TurboQuantConfig(
                bits=TURBOQUANT_BITS,
                group_size=CACHE_GROUP_SIZE,
                sketch_dim=TURBOQUANT_SKETCH_DIM,
                use_residual=TURBOQUANT_RESIDUAL,
            )
            replaced = 0
            for i, c in enumerate(caches):
                if isinstance(c, KVCache):
                    caches[i] = TurboQuantKVCache(tq_config)
                    replaced += 1
            logger.info(
                f"Using TurboQuant KV cache (bits={TURBOQUANT_BITS}, sketch_dim={TURBOQUANT_SKETCH_DIM}) "
                f"for {replaced}/{len(caches)} layers"
            )
        elif KV_CACHE_BITS is not None and _model_is_pipeline_parallel(model):
            # Replace KVCache entries with QuantizedKVCache, but keep
            # ArraysCache (DeltaNet/SSM) and other cache types unchanged.
            # Restricted to PP mode: the single-node BatchGenerator path
            # calls _merge_caches which requires cache.merge(), and
            # QuantizedKVCache does not implement it.
            quantized = 0
            for i, c in enumerate(caches):
                if isinstance(c, KVCache):
                    qc = QuantizedKVCache(
                        group_size=CACHE_GROUP_SIZE, bits=KV_CACHE_BITS
                    )
                    qc.step = 16384
                    caches[i] = qc
                    quantized += 1
            logger.info(
                f"Using quantized KV cache (bits={KV_CACHE_BITS}, group_size={CACHE_GROUP_SIZE}) for {quantized}/{len(caches)} layers"
            )
        else:
            if KV_CACHE_BITS is not None:
                logger.info(
                    f"EXO_KV_CACHE_BITS={KV_CACHE_BITS} ignored in single-node mode "
                    f"(QuantizedKVCache has no merge() support, required by BatchGenerator)"
                )
            else:
                logger.info("Using MLX LM's make cache")
            # Increase KVCache step size to reduce Metal allocator fragmentation.
            # Default step=256 causes a mx.concatenate expansion every prefill chunk,
            # fragmenting memory (~11 GB overhead at 24k tokens). A larger step lets
            # the cache pre-allocate and write in-place for most of the prefill.
            for c in caches:
                if isinstance(c, KVCache):
                    c.step = 16384
        return caches

    if max_kv_size is None:
        logger.info("Using default KV cache")
        return [KVCache() for _ in model.layers]
    else:
        logger.info(f"Using rotating KV cache with {max_kv_size=} with {keep=}")
        return [RotatingKVCache(max_size=max_kv_size, keep=keep) for _ in model.layers]
