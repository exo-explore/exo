# type: ignore
"""Tests for the iPad remote KV cache tier.

The serialisation / RemoteCacheTier integration tests require MLX and are
automatically skipped when the library is unavailable. HTTP protocol tests
live in test_ipad_cache_server.py.
"""

import importlib.util
import io
import threading
import time
import types
from http.server import HTTPServer
from pathlib import Path

import numpy as np
import pytest

_MLX = pytest.importorskip("mlx.core", reason="mlx not available in this environment")

_SERVER_PATH = Path(__file__).parents[8] / "ipad_cache_server.py"


def _load_server_mod(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SERVER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_server(handler_cls, host="127.0.0.1", port=0):
    """Start an HTTPServer in a daemon thread; return (server, base_url)."""
    server = HTTPServer((host, port), handler_cls)
    assigned_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://{host}:{assigned_port}"


# ---------------------------------------------------------------------------
# 1. Serialisation round-trip (requires MLX)
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_array_round_trip(self):
        import mlx.core as mx

        from exo.worker.engines.mlx.remote_cache_tier import _read_array, _write_array

        original = mx.array(np.random.randn(3, 4).astype(np.float32))
        buf = io.BytesIO()
        _write_array(buf, original)
        buf.seek(0)
        restored = _read_array(buf)
        np.testing.assert_array_almost_equal(
            np.array(original), np.array(restored), decimal=5
        )

    def test_serialize_deserialize_round_trip(self):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        from exo.worker.engines.mlx.remote_cache_tier import _deserialize, _serialize

        layer = KVCache()
        layer.keys = mx.array(np.random.randn(1, 4, 8, 16).astype(np.float16))
        layer.values = mx.array(np.random.randn(1, 4, 8, 16).astype(np.float16))
        layer.offset = 8

        prompt = mx.array(np.array([1, 2, 3, 4, 5], dtype=np.int32))
        data = _serialize(prompt, [layer])
        assert data[:4] == b"EXOC"

        restored_prompt, restored_cache = _deserialize(data)
        np.testing.assert_array_equal(np.array(prompt), np.array(restored_prompt))
        assert len(restored_cache) == 1
        assert isinstance(restored_cache[0], KVCache)

        restored_keys = getattr(restored_cache[0], "keys", None)
        restored_offset = getattr(restored_cache[0], "offset", None)
        assert restored_keys is not None
        assert restored_offset == 8

    def test_serialize_skips_non_kvcache_layers(self):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        from exo.worker.engines.mlx.remote_cache_tier import _deserialize, _serialize

        layer_kv = KVCache()
        layer_kv.keys = mx.array(np.ones((1, 2, 4, 8), dtype=np.float32))
        layer_kv.values = mx.array(np.ones((1, 2, 4, 8), dtype=np.float32))
        layer_kv.offset = 4

        layer_rotating = RotatingKVCache(max_size=64, keep=4)
        prompt = mx.array(np.array([10, 20], dtype=np.int32))

        data = _serialize(prompt, [layer_kv, layer_rotating])
        _, restored_cache = _deserialize(data)

        assert len(restored_cache) == 2
        restored_keys = getattr(restored_cache[0], "keys", None)
        assert restored_keys is not None
        assert isinstance(restored_cache[1], KVCache)
        fresh_keys = getattr(restored_cache[1], "keys", None)
        assert fresh_keys is None


# ---------------------------------------------------------------------------
# 2. RemoteCacheTier integration (requires MLX + live mock server)
# ---------------------------------------------------------------------------


class TestRemoteCacheTier:
    @pytest.fixture
    def tier_and_url(self, tmp_path):
        if not _SERVER_PATH.exists():
            pytest.skip("ipad_cache_server.py not found")

        mod = _load_server_mod("_ipad_srv")
        store = mod._BlobStore(max_bytes=256 * 1024 * 1024)  # noqa: SLF001
        throttle = mod._Throttle(concurrent_puts=4, put_min_gap_s=0.0, max_put_bytes=256 * 1024 * 1024)  # noqa: SLF001
        handler_cls = mod._make_handler(store, "<html/>", throttle)  # noqa: SLF001
        srv, url = _run_server(handler_cls)

        from exo.worker.engines.mlx.remote_cache_tier import RemoteCacheTier

        tier = RemoteCacheTier(url, timeout_s=3.0, index_path=tmp_path / "index.json")
        yield tier, url
        srv.shutdown()

    def test_is_available(self, tier_and_url):
        tier, _ = tier_and_url
        assert tier.is_available() is True

    def test_store_and_fetch(self, tier_and_url):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        tier, _ = tier_and_url

        layer = KVCache()
        layer.keys = mx.array(np.ones((1, 2, 16, 32), dtype=np.float32))
        layer.values = mx.array(np.ones((1, 2, 16, 32), dtype=np.float32))
        layer.offset = 16

        prompt_tokens = mx.array(np.arange(50, dtype=np.int32))
        tier.store_async("test-entry-1", prompt_tokens, [layer])
        time.sleep(1.0)

        query_tokens = mx.array(np.arange(45, dtype=np.int32))
        result = tier.fetch(query_tokens, None)

        assert result is not None
        fetched_cache, matched_len = result
        assert matched_len == 45
        assert len(fetched_cache) == 1

    def test_fetch_returns_none_on_short_prefix(self, tier_and_url):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        tier, _ = tier_and_url

        layer = KVCache()
        layer.keys = mx.array(np.ones((1, 2, 10, 32), dtype=np.float32))
        layer.values = mx.array(np.ones((1, 2, 10, 32), dtype=np.float32))
        layer.offset = 10

        prompt_tokens = mx.array(np.arange(50, dtype=np.int32))
        tier.store_async("short-prefix-entry", prompt_tokens, [layer])
        time.sleep(1.0)

        query_tokens = mx.array(np.arange(10, dtype=np.int32))
        result = tier.fetch(query_tokens, None)
        assert result is None

    def test_remove(self, tier_and_url):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        tier, _ = tier_and_url

        layer = KVCache()
        layer.keys = mx.array(np.ones((1, 2, 10, 32), dtype=np.float32))
        layer.values = mx.array(np.ones((1, 2, 10, 32), dtype=np.float32))
        layer.offset = 10

        prompt_tokens = mx.array(np.arange(40, dtype=np.int32))
        tier.store_async("to-remove", prompt_tokens, [layer])
        time.sleep(1.0)

        query = mx.array(np.arange(35, dtype=np.int32))
        assert tier.fetch(query, None) is not None

        tier.remove("to-remove")
        assert all(e.id != "to-remove" for e in tier._index)
