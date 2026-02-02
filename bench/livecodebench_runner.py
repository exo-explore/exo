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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


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

    # Parse known args, pass rest to LiveCodeBench
    args, remaining = parser.parse_known_args()

    # Set up environment
    os.environ["OPENAI_BASE_URL"] = args.base_url
    if "OPENAI_API_KEY" not in os.environ and "OPENAI_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "exo-local"
        os.environ["OPENAI_KEY"] = "exo-local"

    # Setup custom model and patch client
    setup_custom_model(args.model, args.base_url)
    patch_openai_client(args.base_url)

    # Build arguments for LiveCodeBench runner
    lcb_args = ["--model", args.model, *remaining]

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
