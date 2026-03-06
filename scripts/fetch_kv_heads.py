#!/usr/bin/env python3
"""Fetch num_key_value_heads from HuggingFace config.json and update TOML model cards.

Usage:
    # Update only cards missing num_key_value_heads
    uv run python scripts/fetch_kv_heads.py --missing

    # Update all cards (overwrite existing values)
    uv run python scripts/fetch_kv_heads.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tomlkit

CARDS_DIR = Path(__file__).resolve().parent.parent / "resources" / "inference_model_cards"
MAX_WORKERS = 5


def fetch_kv_heads(model_id: str) -> int | None:
    """Fetch num_key_value_heads from HuggingFace config.json."""
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            config = json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}", file=sys.stderr)
        return None

    for source in [config, config.get("text_config", {})]:
        if "num_key_value_heads" in source:
            return int(source["num_key_value_heads"])

    return None


def update_toml(path: Path, kv_heads: int) -> bool:
    """Insert or update num_key_value_heads in a TOML file. Returns True if changed."""
    content = path.read_text()
    doc = tomlkit.parse(content)

    if doc.get("num_key_value_heads") == kv_heads:
        return False

    # Insert after hidden_size if adding for the first time
    if "num_key_value_heads" not in doc:
        new_doc = tomlkit.document()
        for key, value in doc.items():
            new_doc[key] = value
            if key == "hidden_size":
                new_doc["num_key_value_heads"] = kv_heads
        path.write_text(tomlkit.dumps(new_doc))
    else:
        doc["num_key_value_heads"] = kv_heads
        path.write_text(tomlkit.dumps(doc))

    return True


def process_card(path: Path) -> tuple[str, str]:
    """Fetch and update a single card. Returns (filename, status)."""
    content = path.read_text()
    doc = tomlkit.parse(content)
    model_id = doc.get("model_id")
    if not model_id:
        return path.name, "SKIP (no model_id)"

    kv_heads = fetch_kv_heads(str(model_id))
    if kv_heads is None:
        return path.name, "FAILED"

    changed = update_toml(path, kv_heads)
    return path.name, f"{kv_heads} ({'UPDATED' if changed else 'UNCHANGED'})"


def main():
    parser = argparse.ArgumentParser(description="Fetch num_key_value_heads from HuggingFace and update TOML cards.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Update all model cards (overwrite existing values)")
    group.add_argument("--missing", action="store_true", help="Only update cards missing num_key_value_heads")
    args = parser.parse_args()

    toml_files = sorted(CARDS_DIR.glob("*.toml"))
    if not toml_files:
        print(f"No TOML files found in {CARDS_DIR}", file=sys.stderr)
        sys.exit(1)

    to_process = []
    skipped = 0

    for path in toml_files:
        if args.missing and "num_key_value_heads" in path.read_text():
            skipped += 1
            continue
        to_process.append(path)

    updated = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_card, path): path for path in to_process}
        for future in as_completed(futures):
            name, status = future.result()
            print(f"  {name}: {status}")
            if "UPDATED" in status:
                updated += 1
            elif "FAILED" in status:
                failed += 1

    print(f"\nDone: {updated} updated, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
