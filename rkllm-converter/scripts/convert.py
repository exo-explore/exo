#!/usr/bin/env python3
"""
RKLLM Model Converter

Converts HuggingFace models to .rkllm format for Rockchip RK3588/RK3576 NPU.

Usage:
  python convert.py --model Qwen/Qwen2.5-1.5B-Instruct --output qwen2.5-1.5b.rkllm
  python convert.py --model /workspace/models/my-model --output my-model.rkllm
  python convert.py --help

Environment variables:
  HF_TOKEN: HuggingFace token for gated models
  HF_HOME: HuggingFace cache directory (default: /workspace/cache)
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from rkllm.api import RKLLM
except ImportError:
    print("ERROR: RKLLM Toolkit not installed.")
    print("This script requires the RKLLM Toolkit package.")
    print("Install from: https://github.com/airockchip/rknn-llm/releases")
    sys.exit(1)


# Default configuration
DEFAULT_TARGET_PLATFORM = "rk3588"
DEFAULT_QUANTIZED_DTYPE = "w8a8"
DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_NUM_NPU_CORE = 3
DEFAULT_MAX_CONTEXT_LEN = 4096
DEFAULT_MAX_NEW_TOKENS = 2048

# Supported quantization types
QUANT_TYPES = [
    "w4a16",        # 4-bit weights, 16-bit activations
    "w4a16_g128",   # Group-wise 4-bit (group size 128)
    "w8a8",         # 8-bit weights and activations (recommended)
    "w8a8_g128",    # Group-wise 8-bit (group size 128)
]

# Supported target platforms
TARGET_PLATFORMS = ["rk3588", "rk3576"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to RKLLM format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Qwen2.5-1.5B with default settings
  python convert.py --model Qwen/Qwen2.5-1.5B-Instruct --output qwen2.5-1.5b.rkllm

  # Convert with specific quantization
  python convert.py --model Qwen/Qwen2.5-1.5B-Instruct --output model.rkllm --quant w4a16

  # Convert local model directory
  python convert.py --model /workspace/models/my-model --output my-model.rkllm

  # Convert for RK3576 platform
  python convert.py --model Qwen/Qwen2.5-1.5B-Instruct --output model.rkllm --platform rk3576
        """,
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="HuggingFace model ID (e.g., 'Qwen/Qwen2.5-1.5B-Instruct') or local path"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output .rkllm file path"
    )
    parser.add_argument(
        "--platform", "-p",
        default=DEFAULT_TARGET_PLATFORM,
        choices=TARGET_PLATFORMS,
        help=f"Target platform (default: {DEFAULT_TARGET_PLATFORM})"
    )
    parser.add_argument(
        "--quant", "-q",
        default=DEFAULT_QUANTIZED_DTYPE,
        choices=QUANT_TYPES,
        help=f"Quantization type (default: {DEFAULT_QUANTIZED_DTYPE})"
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=DEFAULT_OPTIMIZATION_LEVEL,
        choices=[0, 1, 2],
        help=f"Optimization level 0-2 (default: {DEFAULT_OPTIMIZATION_LEVEL})"
    )
    parser.add_argument(
        "--npu-cores",
        type=int,
        default=DEFAULT_NUM_NPU_CORE,
        choices=[1, 2, 3],
        help=f"Number of NPU cores to use (default: {DEFAULT_NUM_NPU_CORE})"
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=DEFAULT_MAX_CONTEXT_LEN,
        help=f"Maximum context length (default: {DEFAULT_MAX_CONTEXT_LEN})"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum new tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})"
    )
    parser.add_argument(
        "--dataset",
        help="Custom calibration dataset path (optional)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def resolve_model_path(model_id: str) -> str:
    """Resolve model ID to local path, downloading if necessary."""
    # Check if it's already a local path
    if os.path.isdir(model_id):
        print(f"Using local model directory: {model_id}")
        return model_id

    # Otherwise, download from HuggingFace
    print(f"Downloading model from HuggingFace: {model_id}")
    try:
        from huggingface_hub import snapshot_download

        # Check for HF token for gated models
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            print("Using HF_TOKEN for authentication")

        cache_dir = os.environ.get("HF_HOME", "/workspace/cache")
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=hf_token,
        )
        print(f"Downloaded to: {local_path}")
        return local_path
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        sys.exit(1)


def convert_model(args):
    """Convert model to RKLLM format."""
    print("=" * 60)
    print("RKLLM Model Converter")
    print("=" * 60)
    print(f"Model:           {args.model}")
    print(f"Output:          {args.output}")
    print(f"Platform:        {args.platform}")
    print(f"Quantization:    {args.quant}")
    print(f"Optimization:    Level {args.opt_level}")
    print(f"NPU Cores:       {args.npu_cores}")
    print(f"Max Context:     {args.max_context}")
    print(f"Max New Tokens:  {args.max_new_tokens}")
    print("=" * 60)

    # Resolve model path (download if needed)
    model_path = resolve_model_path(args.model)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create RKLLM instance
    print("\n[1/4] Loading model...")
    llm = RKLLM()

    # Load the HuggingFace model
    ret = llm.load_huggingface(model=model_path)
    if ret != 0:
        print(f"ERROR: Failed to load model (code: {ret})")
        sys.exit(1)
    print("Model loaded successfully")

    # Build/quantize the model
    print(f"\n[2/4] Building model (quantization: {args.quant})...")

    # Prepare build config
    build_config = {
        "do_quantization": True,
        "quantized_dtype": args.quant,
        "optimization_level": args.opt_level,
        "num_npu_core": args.npu_cores,
        "max_context_len": args.max_context,
        "max_new_tokens": args.max_new_tokens,
    }

    # Add custom dataset if provided
    if args.dataset:
        if os.path.isfile(args.dataset):
            build_config["dataset"] = args.dataset
            print(f"Using calibration dataset: {args.dataset}")
        else:
            print(f"WARNING: Dataset file not found: {args.dataset}")

    ret = llm.build(**build_config)
    if ret != 0:
        print(f"ERROR: Failed to build model (code: {ret})")
        sys.exit(1)
    print("Model built successfully")

    # Export to .rkllm format
    print(f"\n[3/4] Exporting to {args.output}...")
    ret = llm.export_rkllm(str(output_path))
    if ret != 0:
        print(f"ERROR: Failed to export model (code: {ret})")
        sys.exit(1)

    # Verify output
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Export successful: {output_path} ({size_mb:.1f} MB)")
    else:
        print("ERROR: Output file not created")
        sys.exit(1)

    print(f"\n[4/4] Cleanup...")
    llm.release()

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output: {output_path.absolute()}")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  1. Copy {args.output} to your RK3588/RK3576 device")
    print("  2. Place in ~/RKLLAMA/models/")
    print("  3. Start RKLLAMA server and use with exo")


def main():
    args = parse_args()

    if args.verbose:
        os.environ["DEBUG"] = "1"

    try:
        convert_model(args)
    except KeyboardInterrupt:
        print("\nConversion cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
