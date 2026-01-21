"""
Test script to verify custom model registration works correctly.
"""

import argparse
import asyncio
import json
import os
import sys
import pytest
from typing import cast
from pathlib import Path
from typing import Any

from exo.shared.models.model_cards import get_model_cards, MODEL_CARDS
import exo.shared.models.model_cards as mc
from exo.worker.download.impl_shard_downloader import build_base_shard


@pytest.mark.asyncio
async def test_custom_model_registration():
    """Test that a custom model is registered and persisted."""
    # When running via pytest, we pull config from env vars
    model_id = os.environ.get("MODEL_ID", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    exo_home = os.environ.get("EXO_HOME")

    await run_registration_test(model_id, exo_home)


async def run_registration_test(model_id: str, exo_home: str | None = None) -> bool:
    print(f"\n{'=' * 70}")
    print(f"Testing custom model registration for: {model_id}")
    print(f"{'=' * 70}\n")

    # Get the persistent storage path
    env_exo_home = exo_home or os.environ.get("EXO_HOME")
    if env_exo_home:
        storage_path = Path(env_exo_home) / "custom_models.json"
    else:
        storage_path = Path.home() / ".exo" / "custom_models.json"

    print(f"📁 Persistent storage path: {storage_path}")

    # Clear any existing custom model registration for this test
    if storage_path.exists():
        print(f"⚠️  Found existing custom_models.json, backing up...")
        backup_path = storage_path.with_suffix(".json.backup")
        # Ensure backup path doesn't exist or handle overwrite if needed
        if backup_path.exists():
            backup_path.unlink()
        storage_path.rename(backup_path)
        print(f"✓  Backed up to: {backup_path}")

    # Build a shard for the custom model (this should trigger registration)
    print(f"\n🔧 Building shard for model: {model_id}")
    shard = await build_base_shard(model_id)
    print(f"✓  Shard created: {shard.model_meta.pretty_name}")
    print(f"   - Layers: {shard.n_layers}")
    print(f"   - Size: {shard.model_meta.storage_size.in_mb:.2f} MB")

    # Check if model is now in MODEL_CARDS
    short_id = model_id.split("/")[-1]

    # Use assertions for pytest compatibility if running in pytest context

    if short_id not in MODEL_CARDS:
        print(f"✗  Model NOT found in MODEL_CARDS")
        return False

    print(f"✓  Model found in MODEL_CARDS")
    card = MODEL_CARDS[short_id]
    print(f"   - Name: {card.name}")
    print(f"   - Model ID: {card.model_id}")

    # Check if the JSON file was created
    print(f"\n💾 Checking persistent storage...")
    if not storage_path.exists():
        print(f"✗  custom_models.json NOT created")
        return False

    print(f"✓  custom_models.json created")

    # Read and contents
    with open(storage_path, "r") as f:
        data: dict[str, Any] = cast(dict[str, Any], json.load(f))

    if short_id not in data:
        print(f"✗  Model '{short_id}' NOT found in JSON")
        return False

    print(f"✓  Model '{short_id}' found in JSON")

    # Test that get_model_cards() loads it correctly
    print(f"\n🔄 Testing reload from persistent storage...")

    # Clear the in-memory MODEL_CARDS for this specific custom model
    if short_id in MODEL_CARDS:
        del MODEL_CARDS[short_id]
        print(f"   - Cleared {short_id} from memory")

    # Reset the loaded flag to force a reload
    # reset the loaded flag (handle both older and newer names if needed, but assuming current codebase)
    if hasattr(mc, "_custom_models_loaded"):
        setattr(mc, "_custom_models_loaded", False)
    elif hasattr(mc, "custom_models_loaded"):
        setattr(mc, "custom_models_loaded", False)

    # Load from persistent storage
    model_cards = get_model_cards()

    if short_id not in model_cards:
        print(f"✗  Model NOT reloaded from persistent storage")
        return False

    print(f"✓  Model successfully reloaded from persistent storage")

    print(f"\n{'=' * 70}")
    print(f"✅ All tests passed! Custom model registration is working correctly.")
    print(f"{'=' * 70}\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register and persist custom model test"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    )
    parser.add_argument(
        "--exo-home",
        type=str,
        required=False,
        default=None,
        help="Override EXO_HOME for test persistence",
    )
    args = parser.parse_args()

    model_arg: str = cast(str, args.model)
    exo_home_arg: str | None = cast(str | None, args.exo_home)
    success = asyncio.run(run_registration_test(model_arg, exo_home_arg))
    if not success:
        sys.exit(1)
