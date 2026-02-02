#!/usr/bin/env python3
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""
LiveCodeBench runner wrapper for exo.

This wrapper allows running LiveCodeBench with custom OpenAI-compatible endpoints
by dynamically registering models and configuring the OpenAI client.

Usage:
    python -m bench.livecodebench_runner --model my-model --base-url http://localhost:52415/v1 [lcb args...]

The wrapper:
1. Registers the custom model in LiveCodeBench's model registry
2. Sets up environment variables for the OpenAI client
3. Runs the standard LiveCodeBench runner

Requires LiveCodeBench to be installed:
    git clone https://github.com/LiveCodeBench/LiveCodeBench
    cd LiveCodeBench && uv pip install -e .
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


def get_lcb_directory() -> Path | None:
    """Find the LiveCodeBench installation directory.

    LiveCodeBench uses relative paths like 'lcb_runner/prompts/few_shot_examples/...'
    which require running from the LiveCodeBench directory.
    """
    # Check environment variable first
    if env_path := os.environ.get("LIVECODEBENCH_DIR"):
        lcb_path = Path(env_path)
        if (lcb_path / "lcb_runner" / "prompts" / "few_shot_examples").exists():
            return lcb_path

    # Use importlib to find package location without executing module code
    # This avoids triggering the relative path imports that would fail
    try:
        import importlib.util

        spec = importlib.util.find_spec("lcb_runner")
        if spec and spec.origin:
            # spec.origin is the __init__.py path, go up two levels
            lcb_path = Path(spec.origin).parent.parent
            if (lcb_path / "lcb_runner" / "prompts" / "few_shot_examples").exists():
                return lcb_path
    except (ImportError, ModuleNotFoundError):
        pass

    # Check common locations relative to this script
    script_dir = Path(__file__).parent.parent  # exo/
    common_locations = [
        script_dir / "LiveCodeBench",  # exo/LiveCodeBench
        script_dir.parent / "LiveCodeBench",  # sibling to exo
    ]
    for loc in common_locations:
        if (loc / "lcb_runner" / "prompts" / "few_shot_examples").exists():
            return loc

    return None


def setup_custom_model(model_name: str, base_url: str) -> None:
    """Register a custom model in LiveCodeBench's registry."""
    try:
        from lcb_runner.lm_styles import (  # pyright: ignore[reportMissingImports]
            LanguageModel,
            LanguageModelList,
            LanguageModelStore,
            LMStyle,
        )
    except ImportError as e:
        print(
            "Error: LiveCodeBench not installed. Install with:\n"
            "  git clone https://github.com/LiveCodeBench/LiveCodeBench\n"
            "  cd LiveCodeBench && uv pip install -e .",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    # Check if model already exists
    if model_name in LanguageModelStore:
        return

    # Create a new model entry using OpenAIChat style
    # This will route through the oai_runner which respects OPENAI_BASE_URL
    custom_model = LanguageModel(
        model_name=model_name,
        model_repr=model_name,
        model_style=LMStyle.OpenAIChat,
        release_date=datetime.now(),
        link=base_url,
    )

    # Add to the model list and store
    LanguageModelList.append(custom_model)
    LanguageModelStore[model_name] = custom_model


def patch_openai_client(base_url: str) -> None:
    """Patch the OpenAI client to use a custom base URL.

    This patches the oai_runner module to use our custom base URL.
    """
    try:
        from lcb_runner.runner import oai_runner  # noqa: I001 # pyright: ignore[reportMissingImports]
    except ImportError as e:
        print(f"Error importing required modules: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    # Store original client creation
    original_init = oai_runner.OpenAI

    def patched_openai(*args: Any, **kwargs: Any) -> Any:
        """Create OpenAI client with custom base_url."""
        # Inject base_url if not already set
        if "base_url" not in kwargs:
            kwargs["base_url"] = base_url
        # Use dummy API key if not set (exo doesn't require auth)
        if "api_key" not in kwargs and not os.getenv("OPENAI_KEY"):
            kwargs["api_key"] = os.getenv("OPENAI_API_KEY", "exo-local")
        return original_init(*args, **kwargs)

    # Apply the patch
    oai_runner.OpenAI = patched_openai


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LiveCodeBench runner wrapper for exo",
        epilog="Additional arguments are passed to lcb_runner.runner.main",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:52415/v1"),
        help="OpenAI-compatible API base URL (default: OPENAI_BASE_URL or localhost:52415/v1)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (maps to LiveCodeBench's --custom_output_save_name)",
    )

    # Parse known args, pass rest to LiveCodeBench
    args, remaining = parser.parse_known_args()

    # Set up environment
    os.environ["OPENAI_BASE_URL"] = args.base_url
    if "OPENAI_API_KEY" not in os.environ and "OPENAI_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "exo-local"
        os.environ["OPENAI_KEY"] = "exo-local"

    # Change to LiveCodeBench directory before imports that use relative paths
    # LiveCodeBench uses paths like 'lcb_runner/prompts/few_shot_examples/...'
    lcb_dir = get_lcb_directory()
    if lcb_dir:
        os.chdir(lcb_dir)
    else:
        print(
            "Warning: Could not find LiveCodeBench directory. "
            "Relative path imports may fail.",
            file=sys.stderr,
        )

    # Setup custom model and patch client
    setup_custom_model(args.model, args.base_url)
    patch_openai_client(args.base_url)

    # Build arguments for LiveCodeBench runner
    lcb_args = ["--model", args.model]

    # Map our --output-dir to LiveCodeBench's --custom_output_save_name
    if args.output_dir:
        lcb_args.extend(["--custom_output_save_name", args.output_dir])

    lcb_args.extend(remaining)

    # Run LiveCodeBench
    try:
        from lcb_runner.runner.main import main as lcb_main  # noqa: I001 # pyright: ignore[reportMissingImports]

        # Patch sys.argv for argparse in lcb_main
        sys.argv = [sys.argv[0], *lcb_args]
        lcb_main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"Error running LiveCodeBench: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
