# type: ignore
"""Tests for the iPad remote KV cache tier.

The HTTP blob-store protocol tests run without MLX (plain stdlib).
The serialisation / RemoteCacheTier integration tests require MLX and are
automatically skipped when the library is unavailable.
"""

import io
import json
import threading
import time
import urllib.request
from http.server import HTTPServer

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers shared across test groups
# ---------------------------------------------------------------------------

_MLX = pytest.importorskip("mlx.core", reason="mlx not available in this environment")


def _run_server(handler_cls, host="127.0.0.1", port=0):
    """Start an HTTPServer in a daemon thread; return (server, base_url)."""
    server = HTTPServer((host, port), handler_cls)
    assigned_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://{host}:{assigned_port}"


# ---------------------------------------------------------------------------
# 1. iPad cache server HTTP protocol (no MLX needed)
# ---------------------------------------------------------------------------


class TestIpadCacheServerProtocol:
    """Spin up the real ipad_cache_server handler and verify the HTTP protocol."""

    @pytest.fixture(scope="class")
    def server_url(self):
        # Import the handler factory directly from the server module
        import importlib.util
        import sys
        from pathlib import Path

        server_path = Path(__file__).parents[8] / "ipad_cache_server.py"
        if not server_path.exists():
            pytest.skip(f"ipad_cache_server.py not found at {server_path}")

        spec = importlib.util.spec_from_file_location("ipad_cache_server", server_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ipad_cache_server"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        store = mod._BlobStore(max_bytes=4 * 1024 * 1024)  # 4 MB
        html = "<html>test</html>"
        handler_cls = mod._make_handler(store, html)
        srv, url = _run_server(handler_cls)
        yield url
        srv.shutdown()

    def test_put_and_get(self, server_url):
        payload = b"hello ipad cache"
        req = urllib.request.Request(f"{server_url}/blob/abc123", data=payload, method="PUT")
        with urllib.request.urlopen(req) as r:
            body = json.loads(r.read())
        assert body["ok"] is True
        assert body["size"] == len(payload)

        with urllib.request.urlopen(f"{server_url}/blob/abc123") as r:
            data = r.read()
        assert data == payload

    def test_get_missing_returns_404(self, server_url):
        req = urllib.request.Request(f"{server_url}/blob/nonexistent", method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 404

    def test_stats(self, server_url):
        with urllib.request.urlopen(f"{server_url}/stats") as r:
            stats = json.loads(r.read())
        assert "used_bytes" in stats
        assert "entry_count" in stats
        assert "max_bytes" in stats

    def test_delete(self, server_url):
        payload = b"delete me"
        req = urllib.request.Request(f"{server_url}/blob/del_key", data=payload, method="PUT")
        with urllib.request.urlopen(req):
            pass

        req_del = urllib.request.Request(f"{server_url}/blob/del_key", method="DELETE")
        with urllib.request.urlopen(req_del) as r:
            body = json.loads(r.read())
        assert body["ok"] is True

        req_get = urllib.request.Request(f"{server_url}/blob/del_key", method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_get)
        assert exc_info.value.code == 404

    def test_blobs_list(self, server_url):
        payload = b"listed"
        req = urllib.request.Request(f"{server_url}/blob/listed_key", data=payload, method="PUT")
        with urllib.request.urlopen(req):
            pass

        with urllib.request.urlopen(f"{server_url}/blobs") as r:
            blobs = json.loads(r.read())
        assert isinstance(blobs, list)
        ids = [b["id"] for b in blobs]
        assert "listed_key" in ids

    def test_lru_eviction(self, server_url):
        """LRU eviction should remove oldest entry when over capacity."""
        import importlib.util
        from pathlib import Path

        server_path = Path(__file__).parents[8] / "ipad_cache_server.py"
        spec = importlib.util.spec_from_file_location("ipad_cache_server2", server_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        # 100-byte capacity
        store = mod._BlobStore(max_bytes=100)
        store.put("a", b"x" * 60)
        store.put("b", b"x" * 60)  # triggers eviction of "a"

        assert store.get("a") is None  # evicted
        assert store.get("b") is not None  # kept

    def test_portal_html_served(self, server_url):
        with urllib.request.urlopen(f"{server_url}/") as r:
            body = r.read().decode()
        assert "html" in body.lower()


# ---------------------------------------------------------------------------
# 2. Serialisation round-trip (requires MLX)
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

        # Build a fake KVCache with known keys/values
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
        # KVCache layer restored with data
        restored_keys = getattr(restored_cache[0], "keys", None)
        assert restored_keys is not None
        # RotatingKVCache slot gets a fresh KVCache (no data)
        assert isinstance(restored_cache[1], KVCache)
        fresh_keys = getattr(restored_cache[1], "keys", None)
        assert fresh_keys is None


# ---------------------------------------------------------------------------
# 3. RemoteCacheTier integration (requires MLX + live mock server)
# ---------------------------------------------------------------------------


class TestRemoteCacheTier:
    @pytest.fixture
    def tier_and_url(self, tmp_path):
        import importlib.util
        from pathlib import Path

        server_path = Path(__file__).parents[8] / "ipad_cache_server.py"
        if not server_path.exists():
            pytest.skip("ipad_cache_server.py not found")

        spec = importlib.util.spec_from_file_location("_ipad_srv", server_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        store = mod._BlobStore(max_bytes=256 * 1024 * 1024)
        handler_cls = mod._make_handler(store, "<html/>")
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

        # 50-token prompt
        prompt_tokens = mx.array(np.arange(50, dtype=np.int32))
        entry_id = "test-entry-1"

        tier.store_async(entry_id, prompt_tokens, [layer])
        # Wait for background thread
        time.sleep(1.0)

        # Query with a 45-token prefix of the same prompt (>= 32 token threshold)
        query_tokens = mx.array(np.arange(45, dtype=np.int32))
        result = tier.fetch(query_tokens, None)  # type: ignore[arg-type]

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

        # Only 10-token overlap — below min_useful_tokens (32)
        query_tokens = mx.array(np.arange(10, dtype=np.int32))
        result = tier.fetch(query_tokens, None)  # type: ignore[arg-type]
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

        # Confirm stored
        query = mx.array(np.arange(35, dtype=np.int32))
        assert tier.fetch(query, None) is not None  # type: ignore[arg-type]

        tier.remove("to-remove")

        # After removal, index no longer has the entry
        assert all(e.id != "to-remove" for e in tier._index)
