#!/usr/bin/env python3
"""
Batch RKLLM Model Converter

Converts multiple HuggingFace models to .rkllm format.

Usage:
  python batch_convert.py --config models.yaml
  python batch_convert.py --help
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


DEFAULT_CONFIG = """
# Batch conversion configuration
# Each entry specifies a model to convert

models:
  - name: qwen2.5-1.5b
    model: Qwen/Qwen2.5-1.5B-Instruct
    output: qwen2.5-1.5b-instruct.rkllm
    platform: rk3588
    quant: w8a8

  - name: qwen2.5-3b
    model: Qwen/Qwen2.5-3B-Instruct
    output: qwen2.5-3b-instruct.rkllm
    platform: rk3588
    quant: w8a8

  # - name: deepseek-r1-1.5b
  #   model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  #   output: deepseek-r1-1.5b.rkllm
  #   platform: rk3588
  #   quant: w8a8
  #   max_new_tokens: 8192  # DeepSeek needs more for chain-of-thought
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch convert HuggingFace models to RKLLM format",
    )
    parser.add_argument(
        "--config", "-c",
        default="/workspace/models.yaml",
        help="YAML configuration file (default: /workspace/models.yaml)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="/workspace/output",
        help="Output directory for converted models (default: /workspace/output)"
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate example configuration file and exit"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be converted without actually converting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        print("Use --generate-config to create an example configuration")
        sys.exit(1)

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def convert_model(model_config: dict, output_dir: str, dry_run: bool, verbose: bool):
    """Convert a single model using convert.py."""
    name = model_config.get("name", "unknown")
    model = model_config.get("model")
    output = model_config.get("output")

    if not model or not output:
        print(f"SKIP: Missing 'model' or 'output' for {name}")
        return False

    # Build output path
    output_path = Path(output_dir) / output

    # Build command
    cmd = [
        "python", "/workspace/scripts/convert.py",
        "--model", model,
        "--output", str(output_path),
    ]

    # Add optional parameters
    if "platform" in model_config:
        cmd.extend(["--platform", model_config["platform"]])
    if "quant" in model_config:
        cmd.extend(["--quant", model_config["quant"]])
    if "opt_level" in model_config:
        cmd.extend(["--opt-level", str(model_config["opt_level"])])
    if "npu_cores" in model_config:
        cmd.extend(["--npu-cores", str(model_config["npu_cores"])])
    if "max_context" in model_config:
        cmd.extend(["--max-context", str(model_config["max_context"])])
    if "max_new_tokens" in model_config:
        cmd.extend(["--max-new-tokens", str(model_config["max_new_tokens"])])
    if "dataset" in model_config:
        cmd.extend(["--dataset", model_config["dataset"]])
    if verbose:
        cmd.append("--verbose")

    print(f"\n{'=' * 60}")
    print(f"Converting: {name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True

    # Execute conversion
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Conversion failed with code {e.returncode}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    args = parse_args()

    if args.generate_config:
        print(DEFAULT_CONFIG)
        return

    # Load configuration
    config = load_config(args.config)

    models = config.get("models", [])
    if not models:
        print("ERROR: No models defined in configuration")
        sys.exit(1)

    print(f"Found {len(models)} model(s) to convert")
    print(f"Output directory: {args.output_dir}")

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Convert each model
    results = []
    for model_config in models:
        success = convert_model(
            model_config,
            args.output_dir,
            args.dry_run,
            args.verbose,
        )
        results.append((model_config.get("name", "unknown"), success))

    # Summary
    print("\n" + "=" * 60)
    print("Batch Conversion Summary")
    print("=" * 60)
    successful = sum(1 for _, s in results if s)
    failed = len(results) - successful

    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    print(f"\nTotal: {successful} succeeded, {failed} failed")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
