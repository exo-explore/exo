# pyright: reportAny=false, reportPrivateUsage=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportArgumentType=false, reportCallIssue=false

"""Ring Attention implementation for sequence-parallel distributed inference.

Ring Attention splits the input sequence into blocks across devices in a ring
topology. Each device computes partial attention against its local Q and a
rotating KV block. Sends/receives of the next KV block overlap with attention
computation on the current block, hiding network latency behind compute.

During prefill, KV blocks rotate around the ring and every device accumulates
the full KV cache from the received blocks — so decode has the complete cache
available without any extra communication pass.

During decode, every device has the complete cache populated during prefill.
The normal synchronised generation path feeds the same next token to each
replicated model, so each device can update its own full cache locally.

References:
    Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)
    https://arxiv.org/abs/2310.01889
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.auto_parallel import (
    CustomMlxLayer,
    _LayerCallable,
    get_inner_model,
    get_layers,
    patch_tensor_model,
)

_SUPPORTED_ATTENTION_TYPES = frozenset(
    {
        ("mlx_lm.models.llama", "Attention"),
        ("mlx_lm.models.qwen3", "Attention"),
    }
)


class _AttentionLayer(Protocol):
    """Structural type for attention layers that expose q/k/v projections."""

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: object | None = None,
    ) -> mx.array: ...


def _make_block_causal_mask(
    local_seq_len: int,
    kv_seq_len: int,
    local_offset: int,
    kv_offset: int,
    *,
    n_heads: int,
) -> mx.array | None:
    """Build a causal mask for a specific (query-block, kv-block) pair.

    Returns ``None`` for full attention (no masking needed), or a boolean
    mask of shape ``(1, local_seq_len, kv_seq_len)`` where ``True``
    means "attend".
    """
    local_end = local_offset + local_seq_len
    kv_end = kv_offset + kv_seq_len

    if kv_end <= local_offset:
        return None

    if kv_offset >= local_end:
        return mx.zeros((n_heads, local_seq_len, kv_seq_len), dtype=mx.bool_)

    q_positions = mx.arange(local_offset, local_offset + local_seq_len)[:, None]
    k_positions = mx.arange(kv_offset, kv_offset + kv_seq_len)[None, :]
    causal = q_positions >= k_positions
    return causal[None, :, :]


class RingAttentionLayer(CustomMlxLayer):
    """Wraps an attention layer with ring-attention distributed communication.

    During prefill, the input sequence is split into blocks across devices.
    Each device computes Q from its local block, then rotates KV blocks
    around the ring. All received KV blocks are stored in the cache so
    decode has the full KV available.

    During decode, all ranks receive the same generated token and update their
    replicated full cache locally.
    """

    def __init__(
        self,
        original_layer: _LayerCallable,
        group: mx.distributed.Group,
        sequence_block_size: int = 512,
        send_stream: mx.Stream | None = None,
        receive_stream: mx.Stream | None = None,
    ):
        super().__init__(original_layer)
        self.group = group
        self.rank: int = group.rank()
        self.world_size: int = group.size()
        self.sequence_block_size = sequence_block_size
        # MLX distributed send/receive are CPU operations. Unified memory lets
        # them consume Metal-produced KV arrays without an explicit copy.
        self.send_stream = send_stream or mx.new_stream(mx.cpu)
        self.receive_stream = receive_stream or mx.new_stream(mx.cpu)
        self.is_prefill: bool = False

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: object | None = None,
    ) -> mx.array:
        if self.world_size == 1:
            return self.original_layer(x, mask=mask, cache=cache)

        if not self.is_prefill:
            return self._decode_step(x, mask=mask, cache=cache)

        return self._prefill_step(x, mask=mask, cache=cache)

    def _decode_step(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: object | None = None,
    ) -> mx.array:
        """Regular decode against the replicated full KV cache.

        Ring parallelism is useful for prefill, where each rank owns a
        different sequence block.  After prefill every rank has the complete
        cache, and generation is synchronised by the normal runner protocol,
        so every rank receives the *same* next token.  Gathering K/V here
        would append that token once per rank and corrupt the cache.

        Forwarding to the wrapped attention module keeps decode byte-for-byte
        identical to the non-ring path, including support for every cache
        type the model itself supports (BatchKVCache, quantized, rotating).
        """
        return self.original_layer(x, mask=mask, cache=cache)

    def _prefill_step(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: object | None = None,
    ) -> mx.array:
        """Ring-attention prefill: rotate KV blocks around the ring.

        Stores all received KV blocks in sequence order in the cache so
        decode has the full KV available without extra communication.
        """
        # The verified model implementations create a standard causal mask from
        # their local input. Ring prefill must rebuild that mask with global
        # sequence offsets, so the local mask is intentionally not forwarded.
        # `_validate_ring_attention_layer` rejects implementations whose mask
        # semantics have not been verified equivalent.
        del mask
        original = cast(_AttentionLayer, self.original_layer)

        _, local_seq_len, _ = x.shape
        if local_seq_len == 0:
            raise ValueError("Ring attention does not support empty sequence blocks")

        attn = self._get_attn_module(original)
        queries, keys, values, batch_dim, _, n_heads, n_kv_heads = self._project_qkv(
            attn, x
        )
        head_dim = keys.shape[-1]

        cache_obj = self._require_kv_cache(cache)
        cache_offset = self._cache_offset(cache_obj)
        block_lengths = self._block_lengths(local_seq_len)
        local_offset = cache_offset + sum(block_lengths[: self.rank])

        rope = getattr(attn, "rope", None)
        if rope is not None:
            queries = rope(queries, offset=local_offset)
            keys = rope(keys, offset=local_offset)

        # Communication ops must never depend on unfinished GPU work when they
        # are posted: with MLX_METAL_FAST_SYNCH the CPU send would then block
        # on a Metal shared event inside a bounded command-buffer queue, which
        # can form a cross-rank circular wait (observed as an intermittent
        # long-sequence prefill hang). Materialising the KV payload here keeps
        # the send and receive streams pure-CPU so transport always drains.
        mx.eval(keys, values)

        all_keys: dict[int, mx.array] = {self.rank: keys}
        all_values: dict[int, mx.array] = {self.rank: values}

        current_keys = keys
        current_values = values
        current_rank = self.rank

        running_max: mx.array | None = None
        running_sum: mx.array | None = None
        running_output: mx.array | None = None

        cached_keys, cached_values = self._existing_cache(cache_obj)
        if cached_keys is not None and cached_values is not None:
            running_max, running_sum, running_output = self._accumulate_attention(
                queries,
                cached_keys,
                cached_values,
                scale=self._get_scale(attn),
                mask=None,
                running_max=running_max,
                running_sum=running_sum,
                running_output=running_output,
            )

        pending_sends: list[mx.array] = []
        for step in range(self.world_size):
            next_keys: mx.array | None = None
            next_values: mx.array | None = None
            next_rank: int | None = None
            if step < self.world_size - 1:
                destination_rank = (self.rank + 1) % self.world_size
                source_peer_rank = (self.rank - 1) % self.world_size
                next_rank = (self.rank - step - 1) % self.world_size

                sent_keys = mx.distributed.send(
                    current_keys,
                    destination_rank,
                    group=self.group,
                    stream=self.send_stream,
                )
                sent_values = mx.distributed.send(
                    current_values,
                    destination_rank,
                    group=self.group,
                    stream=self.send_stream,
                )
                pending_sends.extend((sent_keys, sent_values))

                # Templates are allocated on the CPU receive stream so the
                # posted receive never waits on a GPU kernel (see the payload
                # materialisation note above).
                recv_keys = mx.distributed.recv_like(
                    mx.zeros(
                        (batch_dim, n_kv_heads, block_lengths[next_rank], head_dim),
                        dtype=current_keys.dtype,
                        stream=self.receive_stream,
                    ),
                    source_peer_rank,
                    group=self.group,
                    stream=self.receive_stream,
                )
                recv_values = mx.distributed.recv_like(
                    mx.zeros(
                        (batch_dim, n_kv_heads, block_lengths[next_rank], head_dim),
                        dtype=current_values.dtype,
                        stream=self.receive_stream,
                    ),
                    source_peer_rank,
                    group=self.group,
                    stream=self.receive_stream,
                )

                # Both directions are posted before attention is scheduled.
                # Communication uses dedicated streams while the attention graph
                # stays on the caller's compute stream.
                mx.async_eval(sent_keys, sent_values)
                mx.async_eval(recv_keys, recv_values)
                next_keys = recv_keys
                next_values = recv_values

            block_mask = _make_block_causal_mask(
                local_seq_len=local_seq_len,
                kv_seq_len=current_keys.shape[2],
                local_offset=local_offset,
                kv_offset=cache_offset + sum(block_lengths[:current_rank]),
                n_heads=n_heads,
            )

            running_max, running_sum, running_output = self._accumulate_attention(
                queries,
                current_keys,
                current_values,
                scale=self._get_scale(attn),
                mask=block_mask,
                running_max=running_max,
                running_sum=running_sum,
                running_output=running_output,
            )

            # MLX is lazy: explicitly scheduling the recurrence here is what
            # creates a computation window in which the posted receive can run.
            mx.async_eval(running_max, running_sum, running_output)

            if (
                next_keys is not None
                and next_values is not None
                and next_rank is not None
            ):
                mx.eval(next_keys, next_values)
                current_keys = next_keys
                current_values = next_values
                current_rank = next_rank
                all_keys[current_rank] = current_keys
                all_values[current_rank] = current_values

        # Ensure the communication streams no longer reference layer-local KV
        # buffers before cache assembly and the next transformer layer begin.
        if pending_sends:
            mx.eval(pending_sends)

        if cache_obj is not None:
            full_keys = mx.concatenate(
                [all_keys[i] for i in range(self.world_size)], axis=2
            )
            full_values = mx.concatenate(
                [all_values[i] for i in range(self.world_size)], axis=2
            )

            cache_obj.update_and_fetch(full_keys, full_values)

        assert running_output is not None and running_sum is not None
        output = (
            (running_output / running_sum)
            .transpose(0, 2, 1, 3)
            .reshape(batch_dim, local_seq_len, -1)
        )

        o_proj = getattr(attn, "o_proj", None)
        if o_proj is not None:
            output = o_proj(output)

        return output

    def _block_lengths(self, local_seq_len: int) -> list[int]:
        """Return every rank's local sequence length for this prefill call."""
        lengths = mx.distributed.all_gather(
            mx.array([local_seq_len], dtype=mx.int32), group=self.group
        )
        mx.eval(lengths)
        return cast(list[int], lengths.tolist())

    def _cache_offset(self, cache: object | None) -> int:
        offset = getattr(cache, "offset", 0)
        if not isinstance(offset, int):
            raise TypeError("Ring attention requires an integer cache offset")
        return offset

    def _require_kv_cache(self, cache: object | None) -> KVCache | None:
        if cache is None:
            return None
        if not isinstance(cache, KVCache):
            raise TypeError(
                "Ring attention supports only the standard unquantized KVCache; "
                f"received {type(cache).__name__}"
            )
        return cache

    def _existing_cache(
        self, cache: object | None
    ) -> tuple[mx.array | None, mx.array | None]:
        """Read the populated part of a standard, unquantized KV cache."""
        if cache is None or self._cache_offset(cache) == 0:
            return None, None

        keys = getattr(cache, "keys", None)
        values = getattr(cache, "values", None)
        if not isinstance(keys, mx.array) or not isinstance(values, mx.array):
            raise TypeError("Ring attention currently requires an unquantized KV cache")
        offset = self._cache_offset(cache)
        return keys[..., :offset, :], values[..., :offset, :]

    def _accumulate_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        scale: float,
        mask: mx.array | None,
        running_max: mx.array | None,
        running_sum: mx.array | None,
        running_output: mx.array | None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Merge one attention block using the numerically stable online softmax.

        Individual block attention outputs cannot be added: each is normalized
        by a different softmax denominator.  This keeps the running maximum,
        denominator and weighted-value numerator needed to exactly reproduce
        attention over the concatenation of all blocks.
        """
        if keys.shape[1] != queries.shape[1]:
            if queries.shape[1] % keys.shape[1] != 0:
                raise ValueError("Number of query heads must divide KV heads")
            repeats = queries.shape[1] // keys.shape[1]
            keys = mx.repeat(keys, repeats, axis=1)
            values = mx.repeat(values, repeats, axis=1)

        scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale
        if mask is None:
            valid = mx.ones(scores.shape, dtype=mx.bool_)
        else:
            valid = mask[None, :, :, :]
            scores = mx.where(valid, scores, -mx.inf)

        block_has_values = mx.any(valid, axis=-1, keepdims=True)
        block_max = mx.max(scores, axis=-1, keepdims=True)
        safe_block_max = mx.where(block_has_values, block_max, 0.0)
        weights = mx.where(valid, mx.exp(scores - safe_block_max), 0.0)
        block_sum = mx.sum(weights, axis=-1, keepdims=True)
        block_output = weights @ values

        if running_max is None or running_sum is None or running_output is None:
            return safe_block_max, block_sum, block_output

        has_values = (running_sum + block_sum) > 0
        new_max = mx.where(has_values, mx.maximum(running_max, safe_block_max), 0.0)
        running_scale = mx.where(running_sum > 0, mx.exp(running_max - new_max), 0.0)
        block_scale = mx.where(block_sum > 0, mx.exp(safe_block_max - new_max), 0.0)
        return (
            new_max,
            running_sum * running_scale + block_sum * block_scale,
            running_output * running_scale + block_output * block_scale,
        )

    @staticmethod
    def _get_attn_module(original: _AttentionLayer) -> nn.Module:
        if hasattr(original, "q_proj") or hasattr(original, "q_b_proj"):
            return cast(nn.Module, original)
        self_attn = getattr(original, "self_attn", None)
        if self_attn is not None:
            return cast(nn.Module, self_attn)
        attn = getattr(original, "attn", None)
        if attn is not None:
            return cast(nn.Module, attn)
        raise ValueError(
            f"Cannot find attention projections on {type(original).__name__}"
        )

    def _project_qkv(
        self, attn: nn.Module, x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, int, int, int, int]:
        """Project Q/K/V exactly as standard Llama and Qwen attention does."""
        batch_dim, sequence_length, _ = x.shape
        n_heads = self._get_n_heads(attn)
        n_kv_heads = self._get_n_kv_heads(attn)
        queries = self._project_q(attn, x)
        keys = self._project_k(attn, x)
        values = self._project_v(attn, x)
        head_dim = queries.shape[-1] // n_heads

        queries = queries.reshape(batch_dim, sequence_length, n_heads, head_dim)
        keys = keys.reshape(batch_dim, sequence_length, n_kv_heads, head_dim)
        values = values.reshape(batch_dim, sequence_length, n_kv_heads, head_dim)
        if queries.shape[-1] != keys.shape[-1] or keys.shape[-1] != values.shape[-1]:
            raise ValueError("Ring attention requires equal Q, K and V head dimensions")

        q_norm = getattr(attn, "q_norm", None)
        if q_norm is not None:
            queries = q_norm(queries)
        k_norm = getattr(attn, "k_norm", None)
        if k_norm is not None:
            keys = k_norm(keys)

        return (
            queries.transpose(0, 2, 1, 3),
            keys.transpose(0, 2, 1, 3),
            values.transpose(0, 2, 1, 3),
            batch_dim,
            sequence_length,
            n_heads,
            n_kv_heads,
        )

    def _project_q(self, attn: nn.Module, x: mx.array) -> mx.array:
        q_proj = getattr(attn, "q_proj", None)
        if q_proj is not None:
            return q_proj(x)
        raise ValueError(f"Cannot find Q projection on {type(attn).__name__}")

    def _project_k(self, attn: nn.Module, x: mx.array) -> mx.array:
        k_proj = getattr(attn, "k_proj", None)
        if k_proj is not None:
            return k_proj(x)
        raise ValueError(f"Cannot find K projection on {type(attn).__name__}")

    def _project_v(self, attn: nn.Module, x: mx.array) -> mx.array:
        v_proj = getattr(attn, "v_proj", None)
        if v_proj is not None:
            return v_proj(x)
        raise ValueError(f"Cannot find V projection on {type(attn).__name__}")

    def _get_n_heads(self, attn: nn.Module) -> int:
        for attr in ("n_heads", "num_heads", "num_attention_heads"):
            val = getattr(attn, attr, None)
            if isinstance(val, int):
                return val
        raise ValueError(f"Cannot find n_heads on {type(attn).__name__}")

    def _get_n_kv_heads(self, attn: nn.Module) -> int:
        for attr in ("n_kv_heads", "num_kv_heads", "num_key_value_heads"):
            val = getattr(attn, attr, None)
            if isinstance(val, int):
                return val
        return self._get_n_heads(attn)

    def _get_scale(self, attn: nn.Module) -> float:
        scale = getattr(attn, "scale", None)
        if isinstance(scale, float):
            return scale
        n_heads = self._get_n_heads(attn)
        head_dim = getattr(attn, "head_dim", None)
        if isinstance(head_dim, int):
            return head_dim**-0.5
        return (1.0 / n_heads) ** 0.5


def set_ring_prefill(model: nn.Module, is_prefill: bool) -> None:
    """Toggle prefill mode on all RingAttentionLayer wrappers in the model."""
    for layer in _ring_attention_layers(model):
        layer.is_prefill = is_prefill


def has_ring_attention_layer(model: nn.Module) -> bool:
    """Check if the model contains any RingAttentionLayer wrappers."""
    return any(_ring_attention_layers(model))


def uses_ring_sequence_parallel_prefill(
    model: nn.Module,
    prompt_token_count: int,
    group: mx.distributed.Group | None,
) -> bool:
    """Whether this prefill can be partitioned into non-empty rank blocks."""
    return (
        group is not None
        and group.size() > 1
        and prompt_token_count >= group.size()
        and has_ring_attention_layer(model)
    )


def ring_prefill_block_size(model: nn.Module) -> int:
    """Return the configured per-rank token block size for Ring prefill."""
    block_sizes = {layer.sequence_block_size for layer in _ring_attention_layers(model)}
    if len(block_sizes) != 1:
        raise ValueError("Ring attention layers must share one sequence block size")
    return block_sizes.pop()


def _model_layers(model: nn.Module) -> list[_LayerCallable]:
    """Find decoder layers on both flat and nested MLX model wrappers."""
    try:
        return get_layers(get_inner_model(model))
    except ValueError:
        layers = getattr(model, "layers", None)
        if not isinstance(layers, list):
            raise
        return cast(list[_LayerCallable], layers)


def _is_attention_layer(layer: object) -> bool:
    """Heuristic to detect if a layer is an attention layer.

    Checks for common attribute names used across MLX model implementations.
    """
    return any(hasattr(layer, attr) for attr in ("q_proj", "self_attn", "attn"))


def _ring_attention_layers(model: nn.Module) -> list[RingAttentionLayer]:
    """Find wrappers both as decoder layers and nested attention modules."""
    wrappers: list[RingAttentionLayer] = []
    for layer in _model_layers(model):
        if isinstance(layer, RingAttentionLayer):
            wrappers.append(layer)
            continue
        for attribute in ("self_attn", "attn"):
            attention = getattr(layer, attribute, None)
            if isinstance(attention, RingAttentionLayer):
                wrappers.append(attention)
    return wrappers


def _attention_target(layer: _LayerCallable) -> tuple[nn.Module, str | None]:
    """Return the attention module and its parent attribute, if nested."""
    if getattr(layer, "q_proj", None) is not None:
        return cast(nn.Module, layer), None
    for attribute in ("self_attn", "attn"):
        attention = getattr(layer, attribute, None)
        if isinstance(attention, nn.Module):
            return attention, attribute
    raise ValueError(f"Cannot find an attention module on {type(layer).__name__}")


def validate_ring_cache(cache_entries: Sequence[object]) -> None:
    """Reject cache implementations whose semantics Ring prefill cannot preserve."""
    unsupported_cache_types = {
        type(entry).__name__
        for entry in cache_entries
        if not isinstance(entry, KVCache)
    }
    if unsupported_cache_types:
        formatted_types = ", ".join(sorted(unsupported_cache_types))
        raise TypeError(
            "Ring attention supports only standard unquantized KVCache entries; "
            f"received {formatted_types}"
        )


def _validate_ring_attention_layer(layer: _LayerCallable) -> None:
    """Ensure the wrapper only replaces attention with known-equivalent semantics."""
    attention, _ = _attention_target(layer)
    missing_projections = [
        attribute
        for attribute in ("q_proj", "k_proj", "v_proj", "o_proj")
        if getattr(attention, attribute, None) is None
    ]
    if missing_projections:
        raise ValueError(
            "Ring attention supports direct Q/K/V projection attention only; "
            f"{type(attention).__name__} is missing {', '.join(missing_projections)}"
        )

    attention_type = (type(attention).__module__, type(attention).__name__)
    if attention_type not in _SUPPORTED_ATTENTION_TYPES:
        raise ValueError(
            "Ring attention has only been verified for standard MLX Llama and "
            f"Qwen3 attention; received {type(attention).__module__}."
            f"{type(attention).__name__}"
        )

    unsupported_attributes = (
        "use_sliding",
        "use_sliding_window",
        "is_sliding",
        "is_sliding_window",
        "sliding_window",
        "window_size",
        "attention_sinks",
        "sinks",
        "attn_logit_softcapping",
    )
    for owner in (layer, attention):
        for attribute in unsupported_attributes:
            value = getattr(owner, attribute, None)
            if (
                value is None
                or value is False
                or (isinstance(value, (int, float)) and value == 0)
            ):
                continue
            raise ValueError(
                f"Ring attention does not support {attribute} on {type(owner).__name__}"
            )


def _wrap_ring_attention_layer(
    layer: _LayerCallable,
    group: mx.distributed.Group,
    sequence_block_size: int,
    send_stream: mx.Stream,
    receive_stream: mx.Stream,
) -> _LayerCallable:
    """Wrap only attention, never an entire decoder block with an MLP/residual."""
    _validate_ring_attention_layer(layer)
    attention, attribute = _attention_target(layer)
    wrapped = RingAttentionLayer(
        cast(_LayerCallable, attention),
        group=group,
        sequence_block_size=sequence_block_size,
        send_stream=send_stream,
        receive_stream=receive_stream,
    )
    if attribute is None:
        return cast(_LayerCallable, cast(object, wrapped))
    setattr(layer, attribute, wrapped)
    return layer


def ring_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    sequence_block_size: int = 512,
) -> Generator[ModelLoadingResponse, None, nn.Module]:
    """Set up ring-attention parallelism on a model.

    Unlike pipeline (which splits layers) or tensor parallelism (which splits
    weights), ring attention replicates all layers on all devices. Only the
    attention layers are wrapped with :class:`RingAttentionLayer` to enable
    sequence-parallel KV rotation.

    MLP layers, normalisation, and embeddings run independently on each
    device with no communication overhead — they process the local sequence
    block without needing global context.
    """
    inner_model_instance = get_inner_model(model)
    layers = get_layers(inner_model_instance)

    total = len(layers)
    wrapped_layers = 0
    send_stream = mx.new_stream(mx.cpu)
    receive_stream = mx.new_stream(mx.cpu)
    for i, layer in enumerate(layers):
        mx.eval(layer)
        mx.clear_cache()

        if _is_attention_layer(layer):
            layers[i] = _wrap_ring_attention_layer(
                layer,
                group=group,
                sequence_block_size=sequence_block_size,
                send_stream=send_stream,
                receive_stream=receive_stream,
            )
            wrapped_layers += 1

        yield ModelLoadingResponse(layers_loaded=i, total=total)

    if wrapped_layers == 0:
        raise ValueError("Ring attention found no supported attention layers to wrap")

    return patch_tensor_model(model)
