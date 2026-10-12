#!/usr/bin/env python3
"""Convert the DeepSeek V4 MTP block for speculative decoding in exo.

mlx-community's DeepSeek V4 conversions drop the checkpoint's multi-token
prediction block (``mtp.0.*``). This script takes those tensors from the
original DeepSeek checkpoint, converts them like the rest of the MLX model
(8-bit affine linears, mxfp4 experts) and writes them where exo looks for
them: ``~/.exo/mtp/<model dir name>/mtp.safetensors``. exo then loads the
block next to the model and uses it to draft tokens.

Usage:
    # The mtp.0.* tensors are in the last shard of the original checkpoint.
    hf download deepseek-ai/DeepSeek-V4-Flash \\
        model-00046-of-00046.safetensors --local-dir ~/dsv4-mtp

    uv run python scripts/convert_deepseek_v4_mtp.py \\
        ~/.exo/models/mlx-community--DeepSeek-V4-Flash \\
        ~/dsv4-mtp/model-00046-of-00046.safetensors

The output goes to every node that runs the model (each node loads its own
copy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.deepseek_v4 import ModelArgs

from exo.worker.engines.mlx.deepseek_v4_mtp import (
    convert_mtp_weights,
    mtp_weights_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "model", type=Path, help="exo model directory (holds config.json)"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="original DeepSeek V4 safetensors shard with the mtp.0.* tensors",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="override the output path"
    )
    args = parser.parse_args()

    model_dir = args.model.expanduser()
    with open(model_dir / "config.json") as f:
        model_args = ModelArgs.from_dict(json.load(f))
    params = convert_mtp_weights(args.checkpoint.expanduser(), model_args)
    mx.eval(params)

    output = args.output or mtp_weights_path(model_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output), params)
    size_gb = output.stat().st_size / 1e9
    print(f"Wrote {len(params)} tensors ({size_gb:.2f} GB) to {output}")


if __name__ == "__main__":
    main()
