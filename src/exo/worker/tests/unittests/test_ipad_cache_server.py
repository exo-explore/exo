# type: ignore
"""Tests for the iPad cache server HTTP protocol.

These tests do NOT require MLX — they test the pure-stdlib blob store
and HTTP handler in ipad_cache_server.py using real HTTP requests.
"""

import importlib.util
import json
import sys
import threading
import types
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

_SERVER_PATH = Path(__file__).parents[5] / "ipad_cache_server.py"


def _load_server_module() -> types.ModuleType:
    if not _SERVER_PATH.exists():
        pytest.skip(f"ipad_cache_server.py not found at {_SERVER_PATH}")
    spec = importlib.util.spec_from_file_location("_ipad_cache_server_mod", _SERVER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ipad_cache_server_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _start(mod: types.ModuleType, max_bytes: int = 4 * 1024 * 1024) -> tuple[HTTPServer, str]:
    store = mod._BlobStore(max_bytes=max_bytes)  # noqa: SLF001
    throttle = mod._Throttle(concurrent_puts=4, put_min_gap_s=0.0, max_put_bytes=256 * 1024 * 1024)  # noqa: SLF001
    handler_cls = mod._make_handler(store, "<html>test</html>", throttle)  # noqa: SLF001
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def server():
    mod = _load_server_module()
    srv, url = _start(mod)
    yield url
    srv.shutdown()


@pytest.fixture(scope="module")
def tiny_server():
    """Server with 100-byte capacity for eviction tests."""
    mod = _load_server_module()
    srv, url = _start(mod, max_bytes=100)
    yield url
    srv.shutdown()


class TestBlobProtocol:
    def test_put_get(self, server):
        payload = b"hello ipad"
        req = urllib.request.Request(f"{server}/blob/k1", data=payload, method="PUT")
        with urllib.request.urlopen(req) as r:
            body = json.loads(r.read())
        assert body["ok"] is True
        assert body["size"] == len(payload)

        with urllib.request.urlopen(f"{server}/blob/k1") as r:
            assert r.read() == payload

    def test_get_missing_404(self, server):
        req = urllib.request.Request(f"{server}/blob/no_such_key", method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 404

    def test_delete(self, server):
        payload = b"will be deleted"
        req = urllib.request.Request(f"{server}/blob/del_me", data=payload, method="PUT")
        with urllib.request.urlopen(req):
            pass

        req_del = urllib.request.Request(f"{server}/blob/del_me", method="DELETE")
        with urllib.request.urlopen(req_del) as r:
            body = json.loads(r.read())
        assert body["ok"] is True

        req_get = urllib.request.Request(f"{server}/blob/del_me", method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req_get)
        assert exc.value.code == 404

    def test_stats(self, server):
        with urllib.request.urlopen(f"{server}/stats") as r:
            stats = json.loads(r.read())
        assert "used_bytes" in stats
        assert "entry_count" in stats
        assert "max_bytes" in stats
        assert stats["used_bytes"] >= 0
        assert stats["entry_count"] >= 0

    def test_blobs_list(self, server):
        payload = b"list_me"
        req = urllib.request.Request(f"{server}/blob/list_target", data=payload, method="PUT")
        with urllib.request.urlopen(req):
            pass

        with urllib.request.urlopen(f"{server}/blobs") as r:
            blobs = json.loads(r.read())
        assert isinstance(blobs, list)
        ids = [b["id"] for b in blobs]
        assert "list_target" in ids

    def test_head_stats(self, server):
        req = urllib.request.Request(f"{server}/stats", method="HEAD")
        with urllib.request.urlopen(req) as r:
            assert r.status == 200

    def test_portal_html(self, server):
        with urllib.request.urlopen(f"{server}/") as r:
            body = r.read().decode()
        assert "html" in body.lower()

    def test_overwrite(self, server):
        req1 = urllib.request.Request(f"{server}/blob/overwrite_me", data=b"v1", method="PUT")
        with urllib.request.urlopen(req1):
            pass
        req2 = urllib.request.Request(f"{server}/blob/overwrite_me", data=b"v2_longer", method="PUT")
        with urllib.request.urlopen(req2):
            pass
        with urllib.request.urlopen(f"{server}/blob/overwrite_me") as r:
            assert r.read() == b"v2_longer"


class TestLruEviction:
    def test_evicts_oldest_entry(self, tiny_server):
        # 100-byte cap: put 60 bytes → put 60 bytes → first should be evicted
        req_a = urllib.request.Request(f"{tiny_server}/blob/evict_a", data=b"x" * 60, method="PUT")
        with urllib.request.urlopen(req_a):
            pass
        req_b = urllib.request.Request(f"{tiny_server}/blob/evict_b", data=b"y" * 60, method="PUT")
        with urllib.request.urlopen(req_b):
            pass

        req_get_a = urllib.request.Request(f"{tiny_server}/blob/evict_a", method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req_get_a)
        assert exc.value.code == 404

        with urllib.request.urlopen(f"{tiny_server}/blob/evict_b") as r:
            assert r.read() == b"y" * 60

    def test_get_updates_lru_order(self):
        """Accessing an entry via GET promotes it to MRU in the BlobStore.

        Uses max=120 so x(30)+y(60)=90 fits below 85% proactive threshold (102).
        Putting z(30) raises used to 90 → target=72 < 90 → evicts y (LRU), keeps x (MRU).
        """
        mod = _load_server_module()
        store = mod._BlobStore(max_bytes=120)  # noqa: SLF001
        store.put("x", b"x" * 30)
        store.put("y", b"y" * 60)  # 90 total — both fit under 102-byte proactive target
        assert store.get("x") is not None  # access x → x becomes MRU, y is LRU
        store.put("z", b"z" * 30)  # triggers proactive eviction of y (LRU)
        assert store.get("y") is None       # y was LRU — evicted
        assert store.get("x") is not None   # x was MRU — survived
        assert store.get("z") is not None   # z just inserted — survived
