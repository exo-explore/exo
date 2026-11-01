# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
import io

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import CacheList, QuantizedKVCache

from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    KVChunk,
    read_header,
    read_message,
)
from exo.worker.engines.mlx.disaggregated.adapter import (
    _window_arrays,  # pyright: ignore[reportPrivateUsage]
    write_cache_to_wire,
)
from exo.worker.engines.mlx.disaggregated.client import (
    PrefillResult,
    ingest_into_mlx_cache,
)
from exo.worker.engines.mlx.models.deepseek_v4 import model as deepseek_v4
from exo.worker.engines.mlx.models.deepseek_v4.model import DeepseekV4Cache
from exo.worker.engines.mlx.models.deepseek_v4.model import (
    _CompressorBranch as CompressorBranch,  # type: ignore
)

_BRANCH_KEYS = ("compressor", "indexer")


def _equal(a: mx.array | None, b: mx.array | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    if a.dtype == mx.bfloat16:
        return bool(
            np.array_equal(np.asarray(a.view(mx.uint16)), np.asarray(b.view(mx.uint16)))
        )
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _window_meta(cache: DeepseekV4Cache) -> tuple[int, int, int, int]:
    window = cache.local
    return window.keep, window.max_size, window.offset, window._idx


def _same_cache(a: DeepseekV4Cache, b: DeepseekV4Cache) -> bool:
    if _window_meta(a) != _window_meta(b) or a.empty() != b.empty():
        return False
    if not a.empty():
        a_keys, a_values = _window_arrays(a.local)
        b_keys, b_values = _window_arrays(b.local)
        if not (_equal(a_keys, b_keys) and _equal(a_values, b_values)):
            return False
    for key in _BRANCH_KEYS:
        x = a._branches[key]
        y = b._branches[key]
        arrays_x = [x.buffer_kv, x.buffer_gate, x.prev_kv, x.prev_gate, x.pool]
        arrays_y = [y.buffer_kv, y.buffer_gate, y.prev_kv, y.prev_gate, y.pool]
        if not all(_equal(p, q) for p, q in zip(arrays_x, arrays_y, strict=True)):
            return False
        if (x.buffer_lengths, x.pool_lengths, x.buffer_count) != (
            y.buffer_lengths,
            y.pool_lengths,
            y.buffer_count,
        ):
            return False
    return True


def _through_wire(caches: list[DeepseekV4Cache]) -> PrefillResult:
    stream = io.BytesIO()
    write_cache_to_wire(stream, caches, request_id="r", model_id="m")
    stream.seek(0)
    result = PrefillResult(header=read_header(stream))
    while True:
        message = read_message(stream)
        if message is None or isinstance(message, Done):
            break
        if isinstance(message, KVChunk):
            result.kv_chunks.setdefault(message.layer_idx, []).append(message)
        elif isinstance(message, ArraysState):
            result.arrays[message.layer_idx] = message.arrays
    return result


def _synthetic_cache(window: int, tokens: int) -> DeepseekV4Cache:
    mx.random.seed(1)
    cache = DeepseekV4Cache(window)
    # Past the window, so the rotating buffer has wrapped and _idx != offset.
    for _ in range(tokens):
        step = mx.random.normal((1, 1, 1, 8)).astype(mx.bfloat16)
        cache.update_and_fetch(step, step * 2)
    compressor = cache._branches["compressor"]
    compressor.buffer_kv = mx.random.normal((1, 4, 16)).astype(mx.float32)
    compressor.buffer_gate = mx.random.normal((1, 4, 16)).astype(mx.float32)
    compressor.pool = mx.random.normal((1, 3, 8)).astype(mx.bfloat16)
    compressor.buffer_count = 2
    compressor.buffer_lengths = [2]
    compressor.pool_lengths = [3]
    indexer = cache._branches["indexer"]
    indexer.prev_kv = mx.random.normal((1, 4, 16)).astype(mx.float32)
    indexer.prev_gate = mx.random.normal((1, 4, 16)).astype(mx.float32)
    indexer.pool = mx.random.normal((1, 3, 4)).astype(mx.bfloat16)
    return cache


def test_unsupported_cache_types_raise_not_implemented_error() -> None:
    cache_types = [
        QuantizedKVCache(),
        CacheList(),
    ]
    for cache in cache_types:
        with pytest.raises(NotImplementedError):
            write_cache_to_wire(io.BytesIO(), [cache], request_id="r", model_id="m")


def test_deepseek_v4_cache_round_trip_is_exact() -> None:
    source = [_synthetic_cache(window=8, tokens=21), DeepseekV4Cache(8)]
    result = _through_wire(source)
    assert result.kv_chunks == {}
    assert set(result.arrays) == {0, 1}

    target = [DeepseekV4Cache(8), DeepseekV4Cache(8)]
    final_offset = ingest_into_mlx_cache(result, list(target))

    assert final_offset == 21
    assert _same_cache(source[0], target[0])
    assert _same_cache(source[1], target[1])
    assert target[1].empty()


def _small_deepseek_v4(layers: int) -> deepseek_v4.Model:
    # Real compression ratios and sliding window, small widths so it runs on a CPU.
    args = deepseek_v4.ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 1024,
            "hidden_size": 256,
            "num_hidden_layers": layers,
            "num_attention_heads": 8,
            "head_dim": 128,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 64,
            "o_lora_rank": 64,
            "o_groups": 2,
            "index_n_heads": 4,
            "index_head_dim": 64,
            "index_topk": 64,
            "moe_intermediate_size": 64,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "sliding_window": 128,
            "compress_ratios": [0, 0, 4, 128, 4, 128][:layers],
        }
    )
    mx.random.seed(0)
    model = deepseek_v4.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def _make_caches(model: deepseek_v4.Model) -> list[DeepseekV4Cache]:
    caches = [c for c in model.make_cache() if isinstance(c, DeepseekV4Cache)]
    assert len(caches) == len(model.layers)
    return caches


def _prefill(
    model: deepseek_v4.Model, tokens: mx.array, cache: list[DeepseekV4Cache]
) -> None:
    for start in range(0, tokens.shape[0], 512):
        model(tokens[None, start : start + 512], cache=cache)
    mx.eval([c.local.keys for c in cache if c.local.keys is not None])


def _greedy(
    model: deepseek_v4.Model,
    last_token: mx.array,
    cache: list[DeepseekV4Cache],
    steps: int,
) -> tuple[mx.array, list[int]]:
    logits = model(last_token[None], cache=cache)[0, -1].astype(mx.float32)
    first_logits = logits
    generated: list[int] = []
    for _ in range(steps):
        next_token = mx.argmax(logits)
        generated.append(int(next_token.item()))
        logits = model(next_token.reshape(1, 1), cache=cache)[0, -1].astype(mx.float32)
    return first_logits, generated


@pytest.mark.slow
def test_deepseek_v4_decode_after_transport_matches_local_prefill() -> None:
    model = _small_deepseek_v4(layers=6)
    prompt = mx.random.randint(0, 1024, (3001,))

    reference = _make_caches(model)
    _prefill(model, prompt[:-1], reference)
    reference_logits, reference_tokens = _greedy(model, prompt[-1:], reference, 32)

    prefilled = _make_caches(model)
    _prefill(model, prompt[:-1], prefilled)
    result = _through_wire(prefilled)
    received = _make_caches(model)
    assert ingest_into_mlx_cache(result, list(received)) == 3000
    logits, tokens = _greedy(model, prompt[-1:], received, 32)

    assert bool(mx.array_equal(logits, reference_logits))
    assert tokens == reference_tokens

    # Dropping the indexer state must change the output, or this test proves nothing.
    damaged = _make_caches(model)
    _prefill(model, prompt[:-1], damaged)
    damaged_result = _through_wire(damaged)
    damaged_received = _make_caches(model)
    ingest_into_mlx_cache(damaged_result, list(damaged_received))
    for cache in damaged_received:
        cache._branches["indexer"] = CompressorBranch()
    damaged_logits, _ = _greedy(model, prompt[-1:], damaged_received, 0)
    assert not bool(mx.array_equal(damaged_logits, reference_logits))
