"""
exo billing CLI — manage API keys and view usage.

Usage:
    uv run exo-billing create-key --client foo@example.com --plan starter
    uv run exo-billing list-keys
    uv run exo-billing usage exo-abc123...
    uv run exo-billing revoke exo-abc123...
    uv run exo-billing dashboard
    uv run exo-billing plans
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, cast

from exo.api.billing import (
    PLANS,
    check_quota,
    create_key,
    revoke_key,
    tokens_used_this_month,
)

_DB_PATH = Path.home() / ".exo" / "billing.db"

# ─── ANSI colours ────────────────────────────────────────────────────────────
_CYAN   = "\033[38;2;0;229;255m"
_GREEN  = "\033[38;2;34;197;94m"
_AMBER  = "\033[38;2;245;158;11m"
_RED    = "\033[38;2;239;68;68m"
_DIM    = "\033[2m"
_BOLD   = "\033[1m"
_RST    = "\033[0m"


def _fmt_tokens(n: int) -> str:
    if n < 0:
        return "∞"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _month_start_ts() -> int:
    now = datetime.datetime.now(datetime.timezone.utc)
    return int(datetime.datetime(now.year, now.month, 1, tzinfo=datetime.timezone.utc).timestamp())


def _read_db_rows(query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    if not _DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        return list(cast("list[sqlite3.Row]", conn.execute(query, params).fetchall()))
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_create_key(args: argparse.Namespace) -> int:
    client: str = cast("str", getattr(args, "client", ""))
    plan: str = cast("str", getattr(args, "plan", "free"))
    notes: str = cast("str", getattr(args, "notes", ""))
    if plan not in PLANS:
        print(f"{_RED}Unknown plan '{plan}'. Choose from: {', '.join(PLANS)}{_RST}")
        return 1
    raw_key = create_key(client_id=client, plan=plan, notes=notes)
    price: int = PLANS[plan]["price_usd"]
    print(f"\n{_GREEN}{_BOLD}API key created successfully{_RST}")
    print(f"  {_BOLD}Key     :{_RST} {_CYAN}{raw_key}{_RST}")
    print(f"  {_BOLD}Client  :{_RST} {client}")
    print(f"  {_BOLD}Plan    :{_RST} {plan}")
    print(f"  {_BOLD}Price   :{_RST} {'Free' if price == 0 else f'${price}/mo'}")
    print(f"\n{_AMBER}⚠  Store this key securely — it is shown ONCE.{_RST}\n")
    return 0


def cmd_list_keys(_args: argparse.Namespace) -> int:
    rows = _read_db_rows(
        "SELECT key_hash, client_id, plan, created_at, active, notes FROM api_keys ORDER BY created_at DESC"
    )
    if not rows:
        print(f"{_DIM}No API keys found. Database: {_DB_PATH}{_RST}")
        return 0
    print(f"\n{_BOLD}{_CYAN}API KEYS  ({len(rows)} total){_RST}")
    print(f"{'STATUS':<8} {'PLAN':<12} {'CLIENT':<35} {'CREATED':<12} {'NOTES'}")
    print("─" * 90)
    for row in rows:
        active_val: int = cast("int", row["active"])
        status = f"{_GREEN}active{_RST}" if active_val else f"{_RED}revoked{_RST}"
        created_ts: int = cast("int", row["created_at"])
        created = datetime.datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d")
        key_hash: str = cast("str", row["key_hash"])
        prefix = key_hash[:12] + "…"
        client_id: str = cast("str", row["client_id"])
        plan_name: str = cast("str", row["plan"])
        notes_val: str = cast("str", row["notes"] or "")
        print(f"{status:<20} {plan_name:<12} {client_id[:35]:<35} {created:<12} {notes_val}")
        print(f"  {_DIM}hash: {prefix}{_RST}")
    print()
    return 0


def cmd_usage(args: argparse.Namespace) -> int:
    raw_key: str = cast("str", getattr(args, "key", ""))
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    month_start = _month_start_ts()

    info_rows = _read_db_rows(
        "SELECT client_id, plan, created_at, active FROM api_keys WHERE key_hash=?",
        (key_hash,),
    )
    if not info_rows:
        print(f"{_RED}Key not found.{_RST}")
        return 1

    info = info_rows[0]
    plan_name: str = cast("str", info["plan"])
    plan_config = PLANS.get(plan_name, PLANS["free"])
    monthly_limit: int = plan_config["monthly_tokens"]

    usage_rows = _read_db_rows(
        "SELECT SUM(input_tokens) as inp, SUM(output_tokens) as out, "
        "SUM(cost_cents) as cost, COUNT(*) as reqs "
        "FROM usage WHERE key_hash=? AND ts >= ?",
        (key_hash, month_start),
    )
    row = usage_rows[0] if usage_rows else None
    inp: int = cast("int", row["inp"] or 0) if row else 0
    out: int = cast("int", row["out"] or 0) if row else 0
    cost: int = cast("int", row["cost"] or 0) if row else 0
    reqs: int = cast("int", row["reqs"] or 0) if row else 0
    total_tokens = inp + out
    pct = f"{100 * total_tokens / monthly_limit:.1f}%" if monthly_limit > 0 else "∞"

    client_id: str = cast("str", info["client_id"])
    active_val: int = cast("int", info["active"])

    # also use check_quota / tokens_used_this_month to validate the data path
    _ = check_quota(key_hash, plan_name)
    _ = tokens_used_this_month(key_hash)

    print(f"\n{_BOLD}{_CYAN}USAGE REPORT{_RST}")
    print(f"  {_BOLD}Client :{_RST} {client_id}")
    print(f"  {_BOLD}Plan   :{_RST} {plan_name}")
    print(f"  {_BOLD}Active :{_RST} {'yes' if active_val else 'NO (revoked)'}")
    print(f"\n  {_BOLD}This month:{_RST}")
    print(f"    Requests      : {reqs:,}")
    print(f"    Input tokens  : {_fmt_tokens(inp)}")
    print(f"    Output tokens : {_fmt_tokens(out)}")
    print(f"    Total tokens  : {_fmt_tokens(total_tokens)} / {_fmt_tokens(monthly_limit)}  ({pct})")
    print(f"    Cost (COGS)   : ${cost / 100:.4f}")
    print()
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    raw_key: str = cast("str", getattr(args, "key", ""))
    ok = revoke_key(raw_key)
    if ok:
        print(f"{_GREEN}Key revoked.{_RST}")
    else:
        print(f"{_RED}Key not found or already revoked.{_RST}")
    return 0 if ok else 1


def cmd_dashboard(_args: argparse.Namespace) -> int:
    month_start = _month_start_ts()
    month_name = datetime.datetime.now().strftime("%B %Y")

    plan_rows = _read_db_rows(
        "SELECT plan, COUNT(*) as cnt FROM api_keys WHERE active=1 GROUP BY plan ORDER BY cnt DESC"
    )
    agg_rows = _read_db_rows(
        "SELECT SUM(input_tokens) as inp, SUM(output_tokens) as out, "
        "SUM(cost_cents) as cost, COUNT(*) as reqs, COUNT(DISTINCT key_hash) as uniq "
        "FROM usage WHERE ts >= ?",
        (month_start,),
    )
    agg = agg_rows[0] if agg_rows else None
    inp: int = cast("int", agg["inp"] or 0) if agg else 0
    out: int = cast("int", agg["out"] or 0) if agg else 0
    cost: int = cast("int", agg["cost"] or 0) if agg else 0
    reqs: int = cast("int", agg["reqs"] or 0) if agg else 0
    uniq: int = cast("int", agg["uniq"] or 0) if agg else 0
    total = inp + out

    seven_days_ago = int(time.time()) - 7 * 86400
    daily_rows = _read_db_rows(
        "SELECT date(ts, 'unixepoch') as day, SUM(input_tokens+output_tokens) as tokens, COUNT(*) as reqs "
        "FROM usage WHERE ts >= ? GROUP BY day ORDER BY day DESC",
        (seven_days_ago,),
    )

    print(f"\n{_BOLD}{_CYAN}── EXO BILLING DASHBOARD — {month_name} ──{_RST}")
    print(f"\n  {_BOLD}Active Keys by Plan:{_RST}")
    if plan_rows:
        for pr in plan_rows:
            cnt: int = cast("int", pr["cnt"])
            print(f"    {cast('str', pr['plan']):<12} {cnt} key(s)")
    else:
        print(f"    {_DIM}No active keys{_RST}")

    print(f"\n  {_BOLD}Month-to-date:{_RST}")
    print(f"    Requests    : {reqs:,}")
    print(f"    Active users: {uniq}")
    print(f"    Tokens      : {_fmt_tokens(total)}  (in:{_fmt_tokens(inp)} out:{_fmt_tokens(out)})")
    print(f"    COGS        : ${cost / 100:.2f}")
    print(f"    Revenue est.: ${_estimate_revenue(plan_rows):.0f}/mo")

    if daily_rows:
        print(f"\n  {_BOLD}Daily (last 7 days):{_RST}")
        for dr in daily_rows:
            tokens_val: int = cast("int", dr["tokens"] or 0)
            bar_len = min(40, int(tokens_val / max(1, total) * 40))
            bar = "█" * bar_len
            day_str: str = cast("str", dr["day"])
            dr_reqs: int = cast("int", dr["reqs"])
            print(f"    {day_str}  {bar:<40}  {_fmt_tokens(tokens_val)}  ({dr_reqs} req)")

    print()
    return 0


def _estimate_revenue(plan_rows: list[sqlite3.Row]) -> float:
    prices: dict[str, int] = {
        "internal": 0, "free": 0, "starter": 199, "pro": 999, "enterprise": 5000,
    }
    total = 0.0
    for pr in plan_rows:
        plan_name: str = cast("str", pr["plan"])
        cnt: int = cast("int", pr["cnt"])
        total += prices.get(plan_name, 0) * cnt
    return total


def cmd_plans(_args: argparse.Namespace) -> int:
    print(f"\n{_BOLD}{_CYAN}EXO API PLANS{_RST}\n")
    print(f"  {'PLAN':<12} {'TOKENS/MO':<14} {'REQ/MIN':<10} {'PRICE/MO'}")
    print("  " + "─" * 50)
    for name, cfg in PLANS.items():
        monthly_tokens: int = cfg["monthly_tokens"]
        rpm: int = cfg["rpm"]
        price_usd: int = cfg["price_usd"]
        tokens = _fmt_tokens(monthly_tokens)
        rpm_str = "∞" if rpm < 0 else str(rpm)
        price = "free" if price_usd == 0 else f"${price_usd}"
        print(f"  {name:<12} {tokens:<14} {rpm_str:<10} {price}")
    print()
    return 0


# ─── CLI plumbing ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="exo-billing",
        description="Manage exo API keys and view usage.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create-key", help="Create a new API key")
    p_create.add_argument("--client", required=True, help="Client identifier (email / name)")
    p_create.add_argument("--plan", default="free", help="Plan: free|starter|pro|enterprise|internal")
    p_create.add_argument("--notes", default="", help="Optional notes")

    sub.add_parser("list-keys", help="List all API keys")

    p_usage = sub.add_parser("usage", help="Show usage for a key")
    p_usage.add_argument("key", help="Raw API key (exo-...)")

    p_revoke = sub.add_parser("revoke", help="Revoke an API key")
    p_revoke.add_argument("key", help="Raw API key (exo-...)")

    sub.add_parser("dashboard", help="Monthly billing dashboard")
    sub.add_parser("plans", help="Show available plans")

    args = parser.parse_args()
    from collections.abc import Callable
    command: str = cast("str", getattr(args, "command", ""))
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "create-key": cmd_create_key,
        "list-keys": cmd_list_keys,
        "usage": cmd_usage,
        "revoke": cmd_revoke,
        "dashboard": cmd_dashboard,
        "plans": cmd_plans,
    }
    handler = handlers.get(command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
