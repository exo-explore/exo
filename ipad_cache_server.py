#!/usr/bin/env python3
"""iPad / iPhone KV cache blob store + cluster portal.

Pure stdlib Python — no pip deps. Runs in a-Shell on iPad/iPhone.

Works on:
  - iPad (any) — up to 12 GB cache
  - iPhone SE 2020 (A13, 3 GB RAM) — use --max-mb 1500 --device iphone-se
    to stay cool: enforces request pacing + lightweight GETs + tiny footprint.

Usage:
  iPad a-Shell:
    python3 ipad_cache_server.py --max-mb 12000

  iPhone SE 2020 (throttled, cool):
    python3 ipad_cache_server.py --max-mb 1500 --device iphone-se

MacBook side:
    EXO_IPAD_CACHE_URL=http://<device-ip>:9876 uv run exo

Endpoints:
    PUT  /blob/{id}  — store bytes
    GET  /blob/{id}  — retrieve bytes (404 if missing)
    DELETE /blob/{id} — remove entry
    GET  /stats      — JSON usage summary (includes device + throttle state)
    GET  /blobs      — JSON list of all entries
    GET  /           — HTML cluster portal (links to MacBook services)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock, Semaphore
from typing import Any

_PORTAL_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
<title>exo Cache — {device_name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, sans-serif; }}
  header {{ background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 20px; display: flex; align-items: center; gap: 12px; }}
  header h1 {{ font-size: 18px; font-weight: 600; }}
  .badge {{ background: #238636; color: #fff; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600; }}
  .badge.orange {{ background: #9e6a03; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px; }}
  @media (max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  .card-header {{ background: #21262d; padding: 10px 16px; font-size: 13px; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; display: flex; justify-content: space-between; align-items: center; }}
  .card-body {{ padding: 0; }}
  iframe {{ width: 100%; border: none; height: 600px; }}
  .open-btn {{ display: block; text-align: center; padding: 16px; color: #58a6ff; text-decoration: none; font-size: 14px; }}
  .open-btn:hover {{ background: #21262d; }}
  .stats-row {{ padding: 12px 16px; border-bottom: 1px solid #21262d; display: flex; justify-content: space-between; font-size: 13px; }}
  .stats-row:last-child {{ border-bottom: none; }}
  .stats-val {{ color: #58a6ff; font-weight: 600; }}
  #stats-box {{ font-size: 13px; }}
</style>
</head>
<body>
<header>
  <h1>exo Cluster Portal</h1>
  <span class="badge">{device_name}</span>
  <span id="status-badge" class="badge orange">connecting...</span>
</header>
<div class="grid">
  <div class="card" style="grid-column: span 2;">
    <div class="card-header">exo Dashboard<a href="{macbook_url}" target="_blank" style="color:#58a6ff;font-size:11px;text-decoration:none;">open full ↗</a></div>
    <div class="card-body">
      <iframe src="{macbook_url}" id="dashboard-frame" loading="lazy"></iframe>
    </div>
  </div>
  <div class="card">
    <div class="card-header">iPad Cache Stats</div>
    <div class="card-body" id="stats-box">
      <div class="stats-row"><span>Used</span><span class="stats-val" id="s-used">—</span></div>
      <div class="stats-row"><span>Entries</span><span class="stats-val" id="s-count">—</span></div>
      <div class="stats-row"><span>Max</span><span class="stats-val" id="s-max">—</span></div>
    </div>
  </div>
  <div class="card">
    <div class="card-header">Quick Links</div>
    <div class="card-body">
      <a href="{macbook_url}" target="_blank" class="open-btn">exo Dashboard ↗</a>


      <a href="/blobs" target="_blank" class="open-btn">Cached Entries (JSON) ↗</a>
      <a href="/stats" target="_blank" class="open-btn">Cache Stats (JSON) ↗</a>
    </div>
  </div>
</div>
<script>
function fmtBytes(b) {{
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  if (b < 1073741824) return (b/1048576).toFixed(1) + ' MB';
  return (b/1073741824).toFixed(2) + ' GB';
}}
async function refreshStats() {{
  try {{
    const r = await fetch('/stats');
    const d = await r.json();
    document.getElementById('s-used').textContent = fmtBytes(d.used_bytes);
    document.getElementById('s-count').textContent = d.entry_count;
    document.getElementById('s-max').textContent = fmtBytes(d.max_bytes);
    document.getElementById('status-badge').textContent = 'online';
    document.getElementById('status-badge').style.background = '#238636';
  }} catch(e) {{
    document.getElementById('status-badge').textContent = 'error';
    document.getElementById('status-badge').style.background = '#da3633';
  }}
}}
refreshStats();
setInterval(refreshStats, 10000);
</script>
</body>
</html>
"""


_DEVICE_PROFILES: dict[str, dict[str, Any]] = {
    "ipad": {
        "concurrent_puts": 4,
        "put_min_gap_s": 0.0,   # no pacing
        "max_put_bytes": 256 * 1024 * 1024,  # 256 MB per PUT
        "description": "iPad (full speed)",
    },
    "iphone-se": {
        # A13 generates heat fast under sustained writes.
        # Limit: 1 concurrent PUT, ≥0.5 s between PUTs, max 32 MB per blob.
        "concurrent_puts": 1,
        "put_min_gap_s": 0.5,
        "max_put_bytes": 32 * 1024 * 1024,   # 32 MB per PUT
        "description": "iPhone SE 2020 (throttled — stays cool)",
    },
    "mac-mini": {
        "concurrent_puts": 8,
        "put_min_gap_s": 0.0,
        "max_put_bytes": 64 * 1024 * 1024,  # 64 MB/blob
        "description": "Mac Mini (full speed, no throttle)",
    },
}


class _Throttle:
    """Rate limiter for PUT requests to avoid overheating the device.

    Configured per device profile.  GETs are never throttled (read-only,
    low CPU).  PUTs are serialised through a semaphore and spaced apart by
    a minimum gap so the A13 thermal headroom is never exhausted.
    """

    def __init__(self, concurrent_puts: int, put_min_gap_s: float, max_put_bytes: int) -> None:
        self._sem = Semaphore(concurrent_puts)
        self._base_gap_s = put_min_gap_s
        self._effective_gap_s = put_min_gap_s
        self._max_put_bytes = max_put_bytes
        self._last_put_time: float = 0.0
        self._lock = Lock()
        self._put_durations: deque[float] = deque(maxlen=8)

    def acquire_put(self) -> None:
        """Block until a PUT slot is available + thermal gap has elapsed."""
        self._sem.acquire()
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_put_time
            gap = self._effective_gap_s
            if elapsed < gap:
                time.sleep(gap - elapsed)
            self._last_put_time = time.monotonic()

    def release_put(self, duration_s: float = 0.0) -> None:
        self._record_put_duration(duration_s)
        self._sem.release()

    def _record_put_duration(self, duration_s: float) -> None:
        """Adapt thermal gap based on rolling average of PUT durations."""
        with self._lock:
            self._put_durations.append(duration_s)
            if len(self._put_durations) < 3:
                return
            avg = sum(self._put_durations) / len(self._put_durations)
            if avg > 2.0:
                # Device is thermally throttling — widen gap up to 5s ceiling
                self._effective_gap_s = min(self._effective_gap_s + 0.5, 5.0)
            elif avg < 1.0:
                # Device is cool — decay gap back toward profile baseline
                self._effective_gap_s = max(self._effective_gap_s - 0.1, self._base_gap_s)

    @property
    def max_put_bytes(self) -> int:
        return self._max_put_bytes

    @property
    def effective_gap_s(self) -> float:
        return self._effective_gap_s


class _BlobStore:
    """Thread-safe in-memory LRU blob store."""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._data: OrderedDict[str, bytes] = OrderedDict()
        self._lock = Lock()

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            if key in self._data:
                del self._data[key]
            # Proactive early eviction: maintain 15% free headroom to avoid iOS OOM kills
            target = int(self._max_bytes * 0.85) - len(value)
            if target > 0:
                self._evict_to_target(target)
            self._data[key] = value
            self._data.move_to_end(key)
            self._evict()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def delete(self, key: str) -> bool:
        with self._lock:
            if key not in self._data:
                return False
            del self._data[key]
            return True

    def stats(self, device: str = "ipad", throttle: "_Throttle | None" = None) -> dict[str, Any]:
        with self._lock:
            used = sum(len(v) for v in self._data.values())
            result: dict[str, Any] = {
                "used_bytes": used,
                "entry_count": len(self._data),
                "max_bytes": self._max_bytes,
                "pressure_pct": round(used / self._max_bytes * 100, 1) if self._max_bytes > 0 else 0.0,
                "device": device,
            }
            if throttle is not None:
                result["throttle"] = {
                    "max_put_bytes": throttle.max_put_bytes,
                    "put_min_gap_s": throttle._base_gap_s,
                    "effective_gap_s": throttle.effective_gap_s,
                }
            return result

    def list_blobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"id": k, "size": len(v)}
                for k, v in self._data.items()
            ]

    def _evict_to_target(self, target_bytes: int) -> None:
        """Evict LRU entries until used <= target_bytes. Must hold lock."""
        while self._data:
            used = sum(len(v) for v in self._data.values())
            if used <= target_bytes:
                break
            oldest_key, _ = next(iter(self._data.items()))
            del self._data[oldest_key]

    def _evict(self) -> None:
        """Evict LRU entries until under max_bytes. Must hold lock."""
        self._evict_to_target(self._max_bytes)


_store: _BlobStore | None = None
_portal_html: str = ""


def _make_handler(
    store: _BlobStore,
    portal_html: str,
    throttle: _Throttle,
    device_name: str = "ipad",
) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Suppress default access log spam; print errors only
            if args and str(args[1]).startswith(("4", "5")):
                print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {format % args}")

        def _send_json(self, code: int, obj: Any) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, code: int, data: bytes, ct: str = "application/octet-stream") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _blob_id(self) -> str | None:
            path = self.path.split("?")[0]
            parts = path.split("/")
            if len(parts) == 3 and parts[1] == "blob" and parts[2]:
                return parts[2]
            return None

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/":
                body = portal_html.encode()
                self._send_bytes(200, body, "text/html; charset=utf-8")
            elif path == "/health":
                self._send_json(200, {"ok": True, "ts": time.time()})
            elif path == "/stats":
                self._send_json(200, store.stats(device=device_name, throttle=throttle))
            elif path == "/blobs":
                self._send_json(200, store.list_blobs())
            elif path.startswith("/blob/"):
                bid = self._blob_id()
                if not bid:
                    self._send_json(400, {"error": "missing id"})
                    return
                data = store.get(bid)
                if data is None:
                    self._send_json(404, {"error": "not found"})
                    return
                self._send_bytes(200, data)
            else:
                self._send_json(404, {"error": "not found"})

        def do_HEAD(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/stats":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_PUT(self) -> None:  # noqa: N802
            bid = self._blob_id()
            if not bid:
                self._send_json(400, {"error": "missing id"})
                return
            length = int(self.headers.get("Content-Length", 0))
            # Throttle: enforce max size per PUT and thermal pacing
            if throttle.max_put_bytes > 0 and length > throttle.max_put_bytes:
                self._send_json(413, {
                    "error": "payload too large",
                    "max_bytes": throttle.max_put_bytes,
                    "device": device_name,
                })
                return
            throttle.acquire_put()
            put_start = time.monotonic()
            try:
                body = self.rfile.read(length)
                store.put(bid, body)
                self._send_json(200, {"ok": True, "size": len(body)})
            finally:
                throttle.release_put(duration_s=time.monotonic() - put_start)

        def do_DELETE(self) -> None:  # noqa: N802
            bid = self._blob_id()
            if not bid:
                self._send_json(400, {"error": "missing id"})
                return
            found = store.delete(bid)
            self._send_json(200 if found else 404, {"ok": found})

    return _Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="iPad/iPhone exo KV cache blob store + portal")
    parser.add_argument("--port", type=int, default=9876, help="Port to listen on (default: 9876)")
    parser.add_argument("--max-mb", type=int, default=12000, help="Max cache size in MB (default: 12000 for iPad, use 1500 for iPhone SE)")
    parser.add_argument(
        "--device",
        choices=list(_DEVICE_PROFILES.keys()),
        default="ipad",
        help="Device profile: ipad (full speed) or iphone-se (thermally throttled). "
             "iphone-se enforces: 1 concurrent PUT, 0.5s gap, 32MB max per blob.",
    )
    parser.add_argument("--macbook-url", default="http://macbook.local:52415", help="MacBook exo dashboard URL")
    args = parser.parse_args()

    profile = _DEVICE_PROFILES[args.device]
    max_bytes = args.max_mb * 1024 * 1024
    store = _BlobStore(max_bytes)

    # Always construct throttle — adaptive gap only activates when PUTs actually slow down
    throttle = _Throttle(
        concurrent_puts=int(profile["concurrent_puts"]),
        put_min_gap_s=float(profile["put_min_gap_s"]),
        max_put_bytes=int(profile["max_put_bytes"]),
    )

    html = _PORTAL_HTML_TEMPLATE.format(
        macbook_url=args.macbook_url,
        device_name=args.device,
    )

    handler_cls = _make_handler(store, html, throttle=throttle, device_name=args.device)
    server = HTTPServer(("0.0.0.0", args.port), handler_cls)

    print(f"exo cache server — {profile['description']}")
    print(f"  Device:    {args.device}")
    print(f"  Max cache: {args.max_mb} MB")
    print(f"  Port:      {args.port}")
    if profile["put_min_gap_s"] > 0 or profile["concurrent_puts"] < 4:
        print(f"  Throttle:  {profile['concurrent_puts']} concurrent PUT(s), {profile['put_min_gap_s']}s gap, max {profile['max_put_bytes'] // (1024*1024)} MB/blob (adaptive)")
    print(f"  Portal:    http://0.0.0.0:{args.port}/")
    print(f"  MacBook:   {args.macbook_url}")
    print()
    print(f"Open Safari → http://<this-device-ip>:{args.port} for the cluster portal.")
    print(f"Set EXO_IPAD_CACHE_URL=http://<this-device-ip>:{args.port} on MacBook.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
