"""Convert GLM-5 to MXFP4-Q8: experts use MXFP4, dense layers use 8-bit affine.

Usage:
    python convert_glm5_mxfp4_q8.py \
        --hf-path ~/.exo/models/zai-org--GLM-5 \
        --mlx-path ~/.exo/models/mlx-community--GLM-5-MXFP4-Q8 \
        --upload-repo mlx-community/GLM-5-MXFP4-Q8
"""

import argparse
import copy
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from mlx_lm.utils import compute_bits_per_weight, load, save, upload_to_hub

MXFP4_PARAMS = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
AFFINE_Q8_PARAMS = {"group_size": 64, "bits": 8, "mode": "affine"}


def mxfp4_q8_predicate(path: str, module: nn.Module) -> dict | bool:
    """MXFP4 for expert (switch_mlp) weights, 8-bit affine for everything else."""
    if not hasattr(module, "to_quantized"):
        return False
    if not hasattr(module, "weight"):
        return False

    # Expert layers get MXFP4
    if "switch_mlp" in path:
        if module.weight.shape[-1] % MXFP4_PARAMS["group_size"] != 0:
            return False
        return MXFP4_PARAMS

    # Dense layers get 8-bit affine
    if module.weight.shape[-1] % AFFINE_Q8_PARAMS["group_size"] != 0:
        return False
    return AFFINE_Q8_PARAMS


def main():
    parser = argparse.ArgumentParser(description="Convert GLM-5 to MXFP4-Q8")
    parser.add_argument("--hf-path", required=True, help="Path to HF model")
    parser.add_argument("--mlx-path", required=True, default="mlx_model", help="Output path")
    parser.add_argument("--upload-repo", default=None, help="HF repo to upload to")
    args = parser.parse_args()

    mlx_path = Path(args.mlx_path)
    if mlx_path.exists():
        raise ValueError(f"Output path {mlx_path} already exists. Delete it first.")

    print("[INFO] Loading")
    model, tokenizer, config = load(
        args.hf_path,
        return_config=True,
        lazy=True,
    )

    # Apply dtype from config
    dtype = config.get("torch_dtype", None)
    if dtype in ("float16", "bfloat16", "float32"):
        print(f"[INFO] Using dtype: {dtype}")
        dt = getattr(mx, dtype)
        cast_predicate = getattr(model, "cast_predicate", lambda _: True)

        def set_dtype(k, v):
            if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
                return v.astype(dt)
            return v

        model.update(tree_map_with_path(set_dtype, model.parameters()))

    # Build per-layer quantization config
    quantized_config = copy.deepcopy(config)
    quantized_config["quantization"] = {}

    def tracked_predicate(path: str, module: nn.Module) -> dict | bool:
        result = mxfp4_q8_predicate(path, module)
        if isinstance(result, dict):
            quantized_config["quantization"][path] = result
        return result

    print("[INFO] Quantizing (MXFP4 experts + Q8 dense)")
    nn.quantize(
        model,
        class_predicate=tracked_predicate,
    )
    quantized_config["quantization_config"] = quantized_config["quantization"]

    bpw = compute_bits_per_weight(model)
    print(f"[INFO] Quantized model with {bpw:.3f} bits per weight.")

    save(mlx_path, args.hf_path, model, tokenizer, quantized_config)

    if args.upload_repo:
        upload_to_hub(mlx_path, args.upload_repo)


if __name__ == "__main__":
    main()
