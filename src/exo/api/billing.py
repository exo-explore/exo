"""
Metered billing middleware for the exo inference API.

Every request to /v1/chat/completions and /v1/completions is:
  1. Authenticated via Bearer API key
  2. Quota-checked (tokens remaining on plan)
  3. Usage recorded (tokens in + tokens out + cost)
  4. Rate-limited (requests per minute per key)

Plans:
  - free:       100K tokens/mo, 10 req/min
  - starter:    5M tokens/mo,   60 req/min   — $199/mo
  - pro:        50M tokens/mo,  300 req/min  — $999/mo
  - enterprise: unlimited,      unlimited    — custom ACV

Storage: SQLite at ~/.exo/billing.db (local, zero COGS)

No Stripe integration yet — keys are created manually until first client.
Add Stripe webhook in billing_stripe.py once first paying customer signs.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import secrets
import sqlite3
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from fastapi import Request, Response

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_PATH = Path.home() / ".exo" / "billing.db"

_PlanConfig = dict[str, int]  # {"monthly_tokens": int, "rpm": int, "price_usd": int}

PLANS: dict[str, _PlanConfig] = {
    "internal": {"monthly_tokens": -1, "rpm": -1, "price_usd": 0},
    "free": {"monthly_tokens": 100_000, "rpm": 10, "price_usd": 0},
    "starter": {"monthly_tokens": 5_000_000, "rpm": 60, "price_usd": 199},
    "pro": {"monthly_tokens": 50_000_000, "rpm": 300, "price_usd": 999},
    "enterprise": {"monthly_tokens": -1, "rpm": -1, "price_usd": 0},
}

# Cost per 1M tokens in USD cents (Apple Silicon marginal cost, ~80% margin)
_COST_PER_1M_TOKENS_CENTS: int = 50


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key_hash    TEXT PRIMARY KEY,
            client_id   TEXT NOT NULL,
            plan        TEXT NOT NULL DEFAULT 'free',
            created_at  INTEGER NOT NULL,
            active      INTEGER NOT NULL DEFAULT 1,
            notes       TEXT
        );

        CREATE TABLE IF NOT EXISTS usage (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash        TEXT NOT NULL,
            ts              INTEGER NOT NULL,
            endpoint        TEXT NOT NULL,
            input_tokens    INTEGER NOT NULL DEFAULT 0,
            output_tokens   INTEGER NOT NULL DEFAULT 0,
            cost_cents      INTEGER NOT NULL DEFAULT 0,
            model           TEXT,
            request_id      TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON usage(key_hash, ts);
        CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts);
    """)
    conn.commit()


@contextmanager
def _db():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        _init_db(conn)
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------

def create_key(client_id: str, plan: str = "free", notes: str = "") -> str:
    """Create a new API key and return the raw key (shown once)."""
    raw_key = "exo-" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    with _db() as conn:
        conn.execute(
            "INSERT INTO api_keys (key_hash, client_id, plan, created_at, notes) VALUES (?,?,?,?,?)",
            (key_hash, client_id, plan, int(time.time()), notes),
        )
        conn.commit()
    return raw_key


def revoke_key(raw_key: str) -> bool:
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    with _db() as conn:
        cursor = conn.execute(
            "UPDATE api_keys SET active=0 WHERE key_hash=?", (key_hash,)
        )
        conn.commit()
        return bool(cursor.rowcount > 0)


def get_key_info(raw_key: str) -> dict[str, str] | None:
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    with _db() as conn:
        row: sqlite3.Row | None = cast(
            "sqlite3.Row | None",
            conn.execute(
                "SELECT key_hash, client_id, plan, created_at, active, notes "
                "FROM api_keys WHERE key_hash=? AND active=1",
                (key_hash,),
            ).fetchone(),
        )
        if row is None:
            return None
        return {
            "key_hash": cast(str, row["key_hash"]),
            "client_id": cast(str, row["client_id"]),
            "plan": cast(str, row["plan"]),
            "created_at": str(cast(int, row["created_at"])),
            "active": str(cast(int, row["active"])),
            "notes": cast(str, row["notes"] or ""),
        }


# ---------------------------------------------------------------------------
# Quota enforcement
# ---------------------------------------------------------------------------

def _month_start_ts() -> int:
    now = datetime.datetime.now(datetime.timezone.utc)
    return int(datetime.datetime(now.year, now.month, 1, tzinfo=datetime.timezone.utc).timestamp())


def tokens_used_this_month(key_hash: str) -> int:
    start = _month_start_ts()
    with _db() as conn:
        row: sqlite3.Row | None = cast(
            "sqlite3.Row | None",
            conn.execute(
                "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) as total "
                "FROM usage WHERE key_hash=? AND ts >= ?",
                (key_hash, start),
            ).fetchone(),
        )
        return cast(int, row["total"]) if row is not None else 0


def check_quota(key_hash: str, plan: str) -> tuple[bool, str]:
    """Returns (allowed, reason). monthly_tokens=-1 means unlimited."""
    plan_config = PLANS.get(plan, PLANS["free"])
    monthly_limit: int = plan_config["monthly_tokens"]
    if monthly_limit == -1:
        return True, "ok"
    used = tokens_used_this_month(key_hash)
    if used >= monthly_limit:
        return False, f"Monthly quota exceeded: {used:,}/{monthly_limit:,} tokens used"
    return True, "ok"


# ---------------------------------------------------------------------------
# Rate limiting (in-memory, per process)
# ---------------------------------------------------------------------------

_rate_buckets: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(key_hash: str, plan: str) -> tuple[bool, str]:
    plan_config = PLANS.get(plan, PLANS["free"])
    rpm_limit: int = plan_config["rpm"]
    if rpm_limit == -1:
        return True, "ok"
    now = time.monotonic()
    window = 60.0
    _rate_buckets[key_hash] = [t for t in _rate_buckets[key_hash] if now - t < window]
    if len(_rate_buckets[key_hash]) >= rpm_limit:
        return False, f"Rate limit exceeded: {rpm_limit} req/min on {plan} plan"
    _rate_buckets[key_hash].append(now)
    return True, "ok"


# ---------------------------------------------------------------------------
# Usage recording
# ---------------------------------------------------------------------------

def record_usage(
    *,
    key_hash: str,
    endpoint: str,
    input_tokens: int,
    output_tokens: int,
    model: str = "",
    request_id: str = "",
) -> None:
    total_tokens = input_tokens + output_tokens
    cost_cents = (total_tokens * _COST_PER_1M_TOKENS_CENTS) // 1_000_000
    with _db() as conn:
        conn.execute(
            "INSERT INTO usage "
            "(key_hash, ts, endpoint, input_tokens, output_tokens, cost_cents, model, request_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (key_hash, int(time.time()), endpoint, input_tokens, output_tokens, cost_cents, model, request_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------

_METERED_PREFIXES: tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
)

_INTERNAL_KEYS: set[str] = set()


def _load_internal_keys() -> None:
    """Load internal (bypass) keys from ~/.exo/internal_keys.json if present."""
    path = Path.home() / ".exo" / "internal_keys.json"
    if not path.exists():
        return
    try:
        parsed: object = cast(object, json.loads(path.read_text()))
        if not isinstance(parsed, dict):
            return
        typed_parsed: dict[str, object] = cast("dict[str, object]", parsed)
        keys_val: object = typed_parsed.get("keys", [])
        if not isinstance(keys_val, list):
            return
        for item in cast("list[object]", keys_val):
            if isinstance(item, str):
                _INTERNAL_KEYS.add(hashlib.sha256(item.encode()).hexdigest())
    except Exception:
        pass


_load_internal_keys()


async def billing_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """FastAPI middleware — authenticate + quota-check metered endpoints."""
    from fastapi.responses import JSONResponse

    req_path: str = request.url.path
    is_metered = any(req_path.startswith(p) for p in _METERED_PREFIXES)

    if not is_metered:
        return await call_next(request)

    # Extract Bearer token
    auth_header: str = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        # localhost / internal traffic is always allowed without a key
        client_host: str = request.client.host if request.client is not None else "127.0.0.1"
        if client_host in ("127.0.0.1", "::1", "localhost"):
            return await call_next(request)
        return JSONResponse(
            status_code=401,
            content={"error": "missing_api_key", "message": "Provide a Bearer API key."},
        )

    raw_key = auth_header[7:]
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    if key_hash in _INTERNAL_KEYS:
        return await call_next(request)

    key_info = get_key_info(raw_key)
    if key_info is None:
        return JSONResponse(
            status_code=401,
            content={"error": "invalid_api_key", "message": "API key not found or revoked."},
        )

    plan: str = key_info["plan"]

    rate_ok, rate_msg = check_rate_limit(key_hash, plan)
    if not rate_ok:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limit_exceeded", "message": rate_msg},
        )

    quota_ok, quota_msg = check_quota(key_hash, plan)
    if not quota_ok:
        return JSONResponse(
            status_code=429,
            content={"error": "quota_exceeded", "message": quota_msg},
        )

    request.state.billing_key_hash = key_hash
    request.state.billing_plan = plan

    return await call_next(request)


# ---------------------------------------------------------------------------
# CLI: key management
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="exo billing key management")
    sub = parser.add_subparsers(dest="cmd")

    c = sub.add_parser("create", help="Create a new API key")
    c.add_argument("client_id")
    c.add_argument("--plan", default="free", choices=list(PLANS))
    c.add_argument("--notes", default="")

    r = sub.add_parser("revoke", help="Revoke an API key")
    r.add_argument("raw_key")

    sub.add_parser("plans", help="Show available plans")
    sub.add_parser("usage", help="Show monthly usage summary")

    args = parser.parse_args()
    cmd: str = cast(str, args.cmd or "")

    if cmd == "create":
        arg_client_id: str = cast(str, args.client_id)
        arg_plan: str = cast(str, args.plan)
        arg_notes: str = cast(str, args.notes)
        key = create_key(arg_client_id, arg_plan, arg_notes)
        print(f"API key (save this — shown once):\n  {key}")
        print(f"Plan: {arg_plan}  Client: {arg_client_id}")

    elif cmd == "revoke":
        arg_raw_key: str = cast(str, args.raw_key)
        ok = revoke_key(arg_raw_key)
        print("Revoked." if ok else "Key not found.")

    elif cmd == "plans":
        for name, p in PLANS.items():
            tokens = f"{p['monthly_tokens']:,}" if p["monthly_tokens"] != -1 else "unlimited"
            rpm = str(p["rpm"]) if p["rpm"] != -1 else "unlimited"
            print(f"{name:12} ${p['price_usd']:>6}/mo  {tokens} tokens/mo  {rpm} req/min")

    elif cmd == "usage":
        start = _month_start_ts()
        with _db() as conn:
            rows: list[sqlite3.Row] = conn.execute(
                "SELECT k.client_id, k.plan, "
                "COALESCE(SUM(u.input_tokens+u.output_tokens),0) as tokens, "
                "COALESCE(SUM(u.cost_cents),0) as cost_cents "
                "FROM api_keys k LEFT JOIN usage u ON k.key_hash=u.key_hash AND u.ts>=? "
                "GROUP BY k.key_hash ORDER BY tokens DESC",
                (start,),
            ).fetchall()
            print(f"{'CLIENT':<20} {'PLAN':<12} {'TOKENS':>15} {'COST':>10}")
            print("-" * 62)
            for row in rows:
                client_id: str = cast(str, row["client_id"])
                plan_name: str = cast(str, row["plan"])
                token_count: int = cast(int, row["tokens"])
                cost_cents: int = cast(int, row["cost_cents"])
                print(
                    f"{client_id:<20} {plan_name:<12} "
                    f"{token_count:>15,} ${cost_cents/100:>9.2f}"
                )
