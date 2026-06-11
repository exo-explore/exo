# type: ignore
"""KV-cache disk persistence round-trips for exotic architectures.

GLM-5.1 / DeepSeek-V3.2 (DSA) caches hold zero-width arrays and DeepSeek-V4
caches hold Nones/ints/int-lists — none of which safetensors can store. These
tests cover the placeholder mechanism that preserves them across save/load,
plus backward compatibility with legacy (pre-placeholder) slots.

No model weights needed: caches are constructed directly.
"""

import hashlib
import json
import time

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    load_prompt_cache,
    save_prompt_cache,
)
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache

from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    _is_int_list_leaf,
)

MODEL_ID = "test/disk-model"
N_TOKENS = 1100  # disk load requires a >=1000-token prefix


@pytest.fixture
def kvp_factory(tmp_path, monkeypatch):
    """Build KVPrefixCache instances that all share tmp_path as the disk dir."""
    monkeypatch.setenv("EXO_KV_DISK_PATH", str(tmp_path))
    monkeypatch.setenv("EXO_KV_DISK_PERSISTENCE", "1")

    def factory():
        return KVPrefixCache(None, model_id=MODEL_ID)

    return factory


def _slot_dir(tmp_path):
    return tmp_path / hashlib.sha256(MODEL_ID.encode()).hexdigest()[:16]


def test_disk_persistence_is_opt_in(tmp_path, monkeypatch):
    """Without EXO_KV_DISK_PERSISTENCE=1 nothing touches the disk."""
    monkeypatch.setenv("EXO_KV_DISK_PATH", str(tmp_path))
    monkeypatch.delenv("EXO_KV_DISK_PERSISTENCE", raising=False)
    kv = KVPrefixCache(None, model_id=MODEL_ID)
    assert kv._disk_dir is None
    assert list(tmp_path.iterdir()) == []


def test_janitor_sweeps_other_models_stale_slots(tmp_path, kvp_factory):
    """Init must TTL-sweep OTHER models' dirs (their runners may never come
    back to clean up), while leaving fresh slots and our own dir alone."""
    other = tmp_path / "deadbeef00112233"
    other.mkdir()

    def write_slot(i, ts):
        (other / f"slot_{i}_meta.json").write_text(
            json.dumps({"model_id": "other/model", "token_count": 5, "timestamp": ts})
        )
        (other / f"slot_{i}_tokens.safetensors").write_text("stub")
        (other / f"slot_{i}_cache.safetensors").write_text("stub")

    write_slot(0, 1.0)  # ancient -> swept
    write_slot(1, time.time())  # fresh -> kept
    # A stale slot in OUR OWN dir is left to the hot-slot-aware eviction.
    own = _slot_dir(tmp_path)
    own.mkdir(parents=True)
    (own / "slot_7_meta.json").write_text(
        json.dumps({"model_id": MODEL_ID, "token_count": 5, "timestamp": 1.0})
    )

    kvp_factory()  # __init__ runs the janitor

    assert not (other / "slot_0_meta.json").exists()
    assert not (other / "slot_0_cache.safetensors").exists()
    assert not (other / "slot_0_tokens.safetensors").exists()
    assert (other / "slot_1_meta.json").exists()
    assert (own / "slot_7_meta.json").exists()


def test_should_update_entry_only_for_extensions(tmp_path, kvp_factory):
    """Update-in-place is for continuations only. A sibling conversation
    sharing a long prefix (two agent sessions with a common bootstrap) must
    get its own entry — updating steals the matched entry's disk slot."""
    kv = kvp_factory()
    kv.add_kv_cache(_tokens(2000), [_filled_kv(2000)])

    # Continuation: hit covers the stored prompt (modulo prefill rollback)
    assert kv.should_update_entry(0, 1998) is True
    assert kv.should_update_entry(0, 2000) is True
    # Sibling: long shared prefix but far short of the stored conversation
    assert kv.should_update_entry(0, 1500) is False
    # Below the minimum hit threshold
    assert kv.should_update_entry(0, 500) is False
    # No match / stale index
    assert kv.should_update_entry(None, 1998) is False
    assert kv.should_update_entry(5, 1998) is False


def test_sibling_conversation_gets_own_slot(tmp_path, kvp_factory):
    """Slot-theft regression: saving a sibling conversation must not
    overwrite the other conversation's disk slot (observed live as two
    sessions ping-ponging slot_10, each switch destroying the other's
    cache back to the shared prefix)."""
    kv = kvp_factory()
    conv_a = _tokens(2000)
    kv.add_kv_cache(conv_a, [_filled_kv(2000)])
    kv._flush_hot_slot()  # conv A -> slot_0

    # Sibling: shares the first 1500 tokens, then diverges.
    conv_b = mx.concatenate([conv_a[:1500], mx.array([9000 + i for i in range(600)])])
    assert kv.should_update_entry(0, 1500) is False  # decision: add, not update
    kv.add_kv_cache(conv_b, [_filled_kv(2100)])
    kv._flush_hot_slot()  # conv B -> its own slot_1

    d = _slot_dir(tmp_path)
    tok0 = mx.load(str(d / "slot_0_tokens.safetensors"))["tokens"]
    tok1 = mx.load(str(d / "slot_1_tokens.safetensors"))["tokens"]
    assert mx.array_equal(tok0, conv_a).item()  # A's slot intact
    assert mx.array_equal(tok1, conv_b).item()


def test_per_flush_eviction_also_ttl_sweeps_other_models(tmp_path, kvp_factory):
    """The TTL must reach other models' dirs from the per-flush path too —
    not only at init — so one model kept loaded for weeks still cleans up."""
    kv = kvp_factory()  # init janitor runs before the fixtures exist

    other = tmp_path / "cafebabe00112233"
    other.mkdir()
    (other / "slot_0_meta.json").write_text(
        json.dumps({"model_id": "other/model", "token_count": 5, "timestamp": 1.0})
    )
    (other / "slot_0_tokens.safetensors").write_text("stub")
    (other / "slot_0_cache.safetensors").write_text("stub")

    kv._evict_stale_disk_slots()  # what flush_to_disk calls after each flush

    assert not (other / "slot_0_meta.json").exists()
    assert not (other / "slot_0_cache.safetensors").exists()


def test_size_cap_is_global_and_evicts_oldest_first(tmp_path, kvp_factory, monkeypatch):
    """The size cap spans ALL model dirs: the globally oldest slot is evicted
    first (cross-model LRU), eviction stops once back under the cap, and the
    flushing model's own hot slot is never touched."""
    kv = kvp_factory()  # created first: the init janitor must not see fixtures
    kv.add_kv_cache(_tokens(), [_filled_kv()])
    kv._flush_hot_slot()  # own hot slot

    other = tmp_path / "feedface00112233"
    other.mkdir()

    def write_slot(i, ts, payload):
        (other / f"slot_{i}_meta.json").write_text(
            json.dumps({"model_id": "other/model", "token_count": 5, "timestamp": ts})
        )
        (other / f"slot_{i}_tokens.safetensors").write_text(payload)
        (other / f"slot_{i}_cache.safetensors").write_text(payload)

    write_slot(0, 2.0, "x" * 500)  # globally oldest, and big
    write_slot(1, time.time(), "y" * 10)  # fresh and small

    # Cap sits between (old + fresh) and (fresh alone); the hot slot is
    # excluded from the accounting, so only the foreign slots count.
    monkeypatch.setenv("EXO_KV_DISK_MAX_SIZE_GB", str(600 / 1024**3))
    kv._evict_stale_disk_slots()

    assert not (other / "slot_0_meta.json").exists()  # oldest evicted
    assert (other / "slot_1_meta.json").exists()  # under cap again -> kept
    assert (_slot_dir(tmp_path) / "slot_0_cache.safetensors").exists()  # hot slot safe


def _filled_kv(n=N_TOKENS, heads=2, dim=8):
    c = KVCache()
    c.update_and_fetch(
        mx.random.normal([1, heads, n, dim]), mx.random.normal([1, heads, n, dim])
    )
    return c


def _tokens(n=N_TOKENS):
    return mx.arange(n)


def _flush(kvp, cache, tokens):
    kvp.add_kv_cache(tokens, cache)
    kvp._flush_hot_slot()


def _states_equal(a_cache, b_cache):
    a = tree_flatten([c.state for c in a_cache], is_leaf=_is_int_list_leaf)
    b = tree_flatten([c.state for c in b_cache], is_leaf=_is_int_list_leaf)
    assert [p for p, _ in a] == [p for p, _ in b], "state tree paths differ"
    for (path, la), (_, lb) in zip(a, b, strict=True):
        if isinstance(la, mx.array):
            assert isinstance(lb, mx.array), f"{path}: array vs {type(lb)}"
            assert la.shape == lb.shape, f"{path}: shape {la.shape} != {lb.shape}"
            assert la.dtype == lb.dtype, f"{path}: dtype {la.dtype} != {lb.dtype}"
            assert mx.array_equal(la, lb).item(), f"{path}: values differ"
        else:
            assert la == lb, f"{path}: {la!r} != {lb!r}"


def test_standard_kvcache_roundtrip(tmp_path, kvp_factory):
    """Llama/Kimi regression: plain KVCache slots survive, exact and partial."""
    cache = [_filled_kv(), _filled_kv()]
    tokens = _tokens()
    _flush(kvp_factory(), cache, tokens)

    # Exact prefix + new suffix
    query = mx.concatenate([tokens, mx.array([7000, 7001])])
    result = kvp_factory()._try_load_from_disk(None, query)
    assert result is not None
    loaded, remaining, _, _ = result
    _states_equal(cache, loaded)
    assert all(c.offset == N_TOKENS for c in loaded)
    assert mx.array_equal(remaining, query[N_TOKENS:]).item()

    # Partial prefix: diverges at 1050 -> trim to 1050
    query2 = mx.concatenate([tokens[:1050], mx.array([9999] * 5)])
    result2 = kvp_factory()._try_load_from_disk(None, query2)
    assert result2 is not None
    loaded2, remaining2, _, _ = result2
    assert all(c.offset == 1050 for c in loaded2)
    assert mx.array_equal(remaining2, query2[1050:]).item()


def test_legacy_format_slot_loads(tmp_path, kvp_factory):
    """Slots in stock mlx-lm format (no placeholders key in meta.json) load
    through the same path — the placeholder mechanism is strictly additive."""
    cache = [_filled_kv(), _filled_kv()]
    tokens = _tokens()
    d = _slot_dir(tmp_path)
    d.mkdir(parents=True)
    save_prompt_cache(str(d / "slot_0_cache.safetensors"), cache)
    mx.save_safetensors(str(d / "slot_0_tokens.safetensors"), {"tokens": tokens})
    (d / "slot_0_meta.json").write_text(
        json.dumps({"model_id": MODEL_ID, "token_count": N_TOKENS, "timestamp": 1.0})
    )

    result = kvp_factory()._try_load_from_disk(
        None, mx.concatenate([tokens, mx.array([7000])])
    )
    assert result is not None
    _states_equal(cache, result[0])


def test_cache_shorter_than_tokens_loads(tmp_path, kvp_factory):
    """Real slots store the cache ~2 tokens shorter than the token file
    (prefill rollback). The reusable prefix must cap at the cache length —
    a strict equality check would reject every production slot."""
    cache = [_filled_kv(n=N_TOKENS - 2)]
    tokens = _tokens()  # 1100 tokens, cache holds 1098
    _flush(kvp_factory(), cache, tokens)

    query = mx.concatenate([tokens, mx.array([7000])])
    result = kvp_factory()._try_load_from_disk(None, query)
    assert result is not None
    loaded, remaining, _, _ = result
    assert all(c.offset == N_TOKENS - 2 for c in loaded)
    assert mx.array_equal(remaining, query[N_TOKENS - 2 :]).item()


def test_new_save_readable_by_stock_mlx_lm(tmp_path, kvp_factory):
    """For standard archs the new save stays byte-compatible with mlx-lm."""
    cache = [_filled_kv()]
    _flush(kvp_factory(), cache, _tokens())
    loaded = load_prompt_cache(str(_slot_dir(tmp_path) / "slot_0_cache.safetensors"))
    _states_equal(cache, loaded)


def _glm_style_cache(n_layers=2):
    """CacheList(main KVCache, indexer KVCache) per layer; the indexer stores a
    zero-width values array, mirroring deepseek_v32.py's Indexer."""
    layers = []
    for _ in range(n_layers):
        main = _filled_kv()
        indexer = KVCache()
        indexer.update_and_fetch(
            mx.random.normal([1, 1, N_TOKENS, 8]), mx.zeros([1, 1, N_TOKENS, 0])
        )
        layers.append(CacheList(main, indexer))
    return layers


def test_glm_cachelist_zero_width_roundtrip(tmp_path, kvp_factory):
    cache = _glm_style_cache()
    tokens = _tokens()
    _flush(kvp_factory(), cache, tokens)

    meta = json.loads((_slot_dir(tmp_path) / "slot_0_meta.json").read_text())
    assert meta["format"] == 2
    empties = [s for s in meta["placeholders"].values() if s["kind"] == "empty"]
    assert len(empties) == 2  # one zero-width values array per layer
    assert empties[0]["shape"] == [1, 1, N_TOKENS, 0]

    # Exact prefix: previously failed with "not enough values to unpack"
    result = kvp_factory()._try_load_from_disk(
        None, mx.concatenate([tokens, mx.array([7000])])
    )
    assert result is not None
    loaded = result[0]
    _states_equal(cache, loaded)
    assert loaded[0][1].values.shape == (1, 1, N_TOKENS, 0)
    assert all(cl[0].offset == N_TOKENS and cl[1].offset == N_TOKENS for cl in loaded)

    # Partial prefix: CacheList of KVCaches IS trimmable
    query = mx.concatenate([tokens[:1050], mx.array([9999] * 5)])
    result2 = kvp_factory()._try_load_from_disk(None, query)
    assert result2 is not None
    loaded2, remaining2, _, _ = result2
    assert all(cl[0].offset == 1050 and cl[1].offset == 1050 for cl in loaded2)
    assert mx.array_equal(remaining2, query[1050:]).item()


def _v4_cache(sliding_window=64):
    """DeepseekV4Cache with a rotated local window and mixed branch state
    (arrays, Nones, int-lists, ints) — previously crashed the flush."""
    c = DeepseekV4Cache(sliding_window)
    c.local.update_and_fetch(
        mx.random.normal([1, 1, N_TOKENS, 8]), mx.random.normal([1, 1, N_TOKENS, 8])
    )
    comp = c._branches["compressor"]
    comp.pool = mx.random.normal([1, 17, 8])
    comp.buffer_kv = mx.random.normal([1, 4, 16])
    comp.pool_lengths = [17]
    comp.buffer_count = 3
    idx = c._branches["indexer"]
    idx.pool = mx.random.normal([1, 17, 8])
    idx.buffer_lengths = []
    return c


def test_v4_cache_roundtrip(tmp_path, kvp_factory):
    cache = [_v4_cache(), _v4_cache()]
    tokens = _tokens()
    _flush(kvp_factory(), cache, tokens)

    # The flush must actually have produced a slot (old code always failed
    # with "'NoneType' object has no attribute 'size'").
    assert (_slot_dir(tmp_path) / "slot_0_cache.safetensors").exists()
    meta = json.loads((_slot_dir(tmp_path) / "slot_0_meta.json").read_text())
    kinds = {s["kind"] for s in meta["placeholders"].values()}
    assert {"none", "json"} <= kinds

    # Exact-prefix continuation
    result = kvp_factory()._try_load_from_disk(
        None, mx.concatenate([tokens, mx.array([7000, 7001])])
    )
    assert result is not None
    loaded, remaining, _, _ = result
    _states_equal(cache, loaded)
    for orig, new in zip(cache, loaded, strict=True):
        assert new.offset == orig.offset == N_TOKENS
        assert new.local.max_size == orig.local.max_size
        assert new.local._idx == orig.local._idx
        assert new._branches["compressor"].pool_lengths == [17]
        assert new._branches["compressor"].buffer_count == 3
        assert new._branches["compressor"].prev_kv is None
        assert new._branches["indexer"].buffer_lengths == []
    assert mx.array_equal(remaining, mx.array([7000, 7001])).item()

    from copy import deepcopy

    deepcopy(loaded)  # _try_load_from_disk deepcopies — must not raise


def test_v4_partial_prefix_guard(tmp_path, kvp_factory):
    """A partial-prefix hit on a non-trimmable cache must be skipped safely
    (previously: AttributeError on the read-only offset, outside the try)."""
    tokens = _tokens()
    _flush(kvp_factory(), [_v4_cache()], tokens)

    kv = kvp_factory()
    query = mx.concatenate([tokens[:1050], mx.array([9999] * 10)])
    assert kv._try_load_from_disk(None, query) is None
    assert len(kv.caches) == 0  # in-memory state untouched


def test_arrays_cache_none_roundtrip(tmp_path, kvp_factory):
    """SSM-style ArraysCache with None entries round-trips via placeholders
    (hybrid model: one KVCache layer + one ArraysCache layer)."""
    kv_layer = _filled_kv()
    ssm = ArraysCache(2)
    ssm[0] = mx.random.normal([1, 4])
    assert ssm.cache[1] is None
    cache = [kv_layer, ssm]
    tokens = _tokens()
    _flush(kvp_factory(), cache, tokens)

    result = kvp_factory()._try_load_from_disk(
        None, mx.concatenate([tokens, mx.array([7000])])
    )
    assert result is not None
    loaded = result[0]
    _states_equal(cache, loaded)
    assert loaded[1].cache[1] is None
