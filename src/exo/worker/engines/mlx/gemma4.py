from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.gemma4_text import Gemma4TextModel
from mlx_lm.models.gemma4_text import ModelArgs as Gemma4TextModelArgs

type _IntermediateEntry = tuple[tuple[mx.array, mx.array] | None, mx.array | None]
type _SourceKvs = tuple[mx.array, mx.array, mx.array]


# TODO: Really really ugly code that needs refactoring ASAP (but it works)


def is_gemma4_inner_model(inner: nn.Module) -> bool:
    return isinstance(inner, Gemma4TextModel)


def is_gemma4_pipeline_model(inner: nn.Module) -> bool:
    return isinstance(inner, Gemma4TextModel) and hasattr(
        inner, "_gemma4_pipeline_group"
    )


def try_set_gemma4_pipeline_prefill(inner: nn.Module, is_prefill: bool) -> bool:
    if isinstance(inner, Gemma4TextModel) and hasattr(inner, "_gemma4_is_prefill"):
        inner._gemma4_is_prefill = is_prefill
        return True
    return False


def try_set_gemma4_pipeline_queue_sends(inner: nn.Module, queue_sends: bool) -> bool:
    if isinstance(inner, Gemma4TextModel) and hasattr(inner, "_gemma4_queue_sends"):
        inner._gemma4_queue_sends = queue_sends
        return True
    return False


def _offset_template() -> mx.array:
    return mx.array(0, dtype=mx.int32)


def _kv_template(
    args: Gemma4TextModelArgs,
    layer_types_global: list[str],
    h: mx.array,
    source_global_idx: int,
    seq_len: int,
) -> mx.array:
    is_full = (
        source_global_idx < len(layer_types_global)
        and layer_types_global[source_global_idx] == "full_attention"
    )
    head_dim = (
        int(args.global_head_dim)
        if is_full and args.global_head_dim
        else int(args.head_dim)
    )
    n_kv_heads = int(args.num_key_value_heads)
    if (
        is_full
        and args.attention_k_eq_v
        and args.num_global_key_value_heads is not None
    ):
        n_kv_heads = int(args.num_global_key_value_heads)
    if not is_full and args.sliding_window:
        seq_len = min(seq_len, int(args.sliding_window))
    return mx.zeros((int(h.shape[0]), n_kv_heads, seq_len, head_dim), dtype=h.dtype)


def patch_gemma4_pipeline(
    model: nn.Module,
    inner: nn.Module,
    start_layer: int,
    end_layer: int,
    device_rank: int,
    world_size: int,
    group: mx.distributed.Group,
    pending_prefill_sends: list[tuple[mx.array, int, mx.distributed.Group]],
) -> None:
    assert isinstance(inner, Gemma4TextModel)
    args = inner.config
    num_hidden_layers_global = int(args.num_hidden_layers)
    num_kv_shared_layers_global = int(args.num_kv_shared_layers)
    first_kv_shared_global = num_hidden_layers_global - num_kv_shared_layers_global
    layer_types_global = list(args.layer_types or [])

    previous_kvs_global = list(inner.previous_kvs)

    consumed_global_sources: list[int] = sorted(
        set(previous_kvs_global[first_kv_shared_global:])
    )
    # Source layers we own locally to local layer index.
    local_owned_sources: dict[int, int] = {
        g_idx: g_idx - start_layer
        for g_idx in consumed_global_sources
        if start_layer <= g_idx < end_layer
    }
    # Local shared-layer slot → the source's global index.
    local_shared_to_global_source: dict[int, int] = {
        g_idx - start_layer: previous_kvs_global[g_idx]
        for g_idx in range(max(start_layer, first_kv_shared_global), end_layer)
    }

    new_previous_kvs: list[int] = []
    for g_idx in range(start_layer, end_layer):
        local_idx = g_idx - start_layer
        source_g = previous_kvs_global[g_idx]
        if start_layer <= source_g < end_layer:
            new_previous_kvs.append(source_g - start_layer)
        else:
            new_previous_kvs.append(local_idx)
    inner.previous_kvs = new_previous_kvs

    sliding_window = int(args.sliding_window)

    def _make_cache() -> list[KVCache | RotatingKVCache]:
        local_source_end = min(first_kv_shared_global, end_layer)
        caches: list[KVCache | RotatingKVCache] = []
        for g_idx in range(start_layer, local_source_end):
            if layer_types_global[g_idx] == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=sliding_window, keep=0))
        return caches

    model.make_cache = _make_cache

    inner._gemma4_pipeline_group = group
    inner._gemma4_device_rank = device_rank
    inner._gemma4_world_size = world_size
    inner._gemma4_is_prefill = False
    inner._gemma4_queue_sends = False
    # Fallback counter for tracking the source-kv sequence length on ranks
    # whose local cache list is empty (e.g. a rank that only owns shared
    # layers). Normal ranks read the offset from their local KVCache.
    inner._gemma4_prefix_counter = 0

    next_rank = (device_rank + 1) % world_size
    prev_rank = device_rank - 1

    def patched_call(
        self: Gemma4TextModel,
        inputs: mx.array | None = None,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        per_layer_inputs: mx.array | None = None,
    ) -> mx.array:
        if input_embeddings is None:
            assert inputs is not None
            input_embeddings = self.embed_tokens(inputs)
        h: mx.array = input_embeddings * self.embed_scale

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self._get_per_layer_inputs(inputs, input_embeddings)
            per_layer_inputs = self._project_per_layer_inputs(h, per_layer_inputs)
            # Both helpers above return tensors shaped for the GLOBAL layer
            # count (see patch_gemma4_pipeline). Slice to this shard's layers
            # so the local loop indexes into the right slots.
            per_layer_inputs = per_layer_inputs[:, :, start_layer:end_layer, :]
        per_layer_inputs_list: list[mx.array | None] = (
            [per_layer_inputs[:, :, i, :] for i in range(len(self.layers))]
            if per_layer_inputs is not None
            else [None] * len(self.layers)
        )

        local_cache: list[KVCache | RotatingKVCache | None] = (
            [None] * len(self.layers)
            if cache is None
            else cache + [None] * (len(self.layers) - len(cache))
        )
        masks: list[mx.array] = self._make_masks(h, local_cache)

        prior_offset: int = int(getattr(self, "_gemma4_prefix_counter", 0))

        current_seq_len = prior_offset + int(h.shape[1])

        if device_rank != 0:
            mx.eval(h)
            h = mx.distributed.recv_like(h, prev_rank, group=group)
            mx.eval(h)

        # Receive the source kvs. We always recv ALL consumed
        # sources (even ones we own locally) so the byte counts on each side
        # of the wire match — entries we own locally are simply ignored when
        # seeding intermediates below.
        received_source_kvs: dict[int, _SourceKvs] = {}
        if device_rank != 0:
            offset_template = _offset_template()
            len_template = mx.array(0, dtype=mx.int32)
            for g_idx in consumed_global_sources:
                rlen = mx.distributed.recv_like(len_template, prev_rank, group=group)
                mx.eval(rlen)
                actual_seq_len = int(rlen.item())
                kv_template = _kv_template(
                    args, layer_types_global, h, g_idx, actual_seq_len
                )
                rk = mx.distributed.recv_like(kv_template, prev_rank, group=group)
                mx.eval(rk)
                rv = mx.distributed.recv_like(kv_template, prev_rank, group=group)
                mx.eval(rv)
                ro = mx.distributed.recv_like(offset_template, prev_rank, group=group)
                mx.eval(ro)
                received_source_kvs[g_idx] = (rk, rv, ro)

        intermediates: list[_IntermediateEntry] = [(None, None)] * len(self.layers)
        for local_idx, source_g_idx in local_shared_to_global_source.items():
            if source_g_idx in local_owned_sources:
                continue
            entry = received_source_kvs.get(source_g_idx)
            if entry is not None:
                rk, rv, ro = entry
                intermediates[local_idx] = ((rk, rv), ro)

        for idx in range(len(self.layers)):
            layer = self.layers[idx]
            prev_idx = self.previous_kvs[idx]
            kvs, offset = intermediates[prev_idx]
            h, new_kvs, new_offset = layer(
                h,
                masks[idx],
                local_cache[idx],
                per_layer_input=per_layer_inputs_list[idx],
                shared_kv=kvs,
                offset=offset,
            )
            intermediates[idx] = (new_kvs, new_offset)
            mx.eval(h)

        # Build the outgoing source kvs. Start from whatever we received (so
        # any source we don't own locally gets forwarded along), then
        # overwrite with our own freshly-computed entries.
        outgoing_source_kvs: dict[int, _SourceKvs] = dict(received_source_kvs)
        for g_idx, local_idx in local_owned_sources.items():
            local_kvs, local_offset = intermediates[local_idx]
            if local_kvs is not None and local_offset is not None:
                outgoing_source_kvs[g_idx] = (local_kvs[0], local_kvs[1], local_offset)

        is_prefill: bool = bool(getattr(self, "_gemma4_is_prefill", False))
        queue_sends: bool = bool(getattr(self, "_gemma4_queue_sends", False))

        if device_rank != world_size - 1:
            mx.eval(h)
            if queue_sends:
                pending_prefill_sends.append((h, next_rank, group))
            else:
                h = mx.distributed.send(h, next_rank, group=group)
                mx.eval(h)
            offset_template = _offset_template()
            for g_idx in consumed_global_sources:
                entry = outgoing_source_kvs.get(g_idx)
                if entry is None:
                    kv_template = _kv_template(
                        args, layer_types_global, h, g_idx, current_seq_len
                    )
                    entry = (kv_template, kv_template, offset_template)
                kk, vv, oo = entry
                mx.eval(kk, vv, oo)
                if queue_sends:
                    pending_prefill_sends.append((kk, next_rank, group))
                    pending_prefill_sends.append((vv, next_rank, group))
                    pending_prefill_sends.append((oo, next_rank, group))
                else:
                    actual_len = mx.array(int(kk.shape[2]), dtype=mx.int32)
                    sent_len = mx.distributed.send(actual_len, next_rank, group=group)
                    mx.eval(sent_len)
                    kk = mx.distributed.send(kk, next_rank, group=group)
                    mx.eval(kk)
                    vv = mx.distributed.send(vv, next_rank, group=group)
                    mx.eval(vv)
                    oo = mx.distributed.send(oo, next_rank, group=group)
                    mx.eval(oo)

        self._gemma4_prefix_counter = prior_offset + int(h.shape[1])

        if not is_prefill:
            mx.eval(h)
            h = mx.distributed.all_gather(h, group=group)[-h.shape[0] :]
            mx.eval(h)

        return self.norm(h)

    type(inner).__call__ = patched_call
