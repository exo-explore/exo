#!/usr/bin/env python3
"""Convert Qwen3.8-Flash-Next's MTP head for speculative decoding in exo.

mlx-community's Qwen3.8-Flash-Next conversions drop the checkpoint's multi-token
prediction layer (``mtp.*``). This script takes those tensors from Qwen's BF16 release,
quantizes them like the rest of the MLX model and writes them where exo looks for them:
``~/.exo/mtp/<model dir name>/mtp.safetensors``. exo then attaches the head on the node
that runs the model's last pipeline stage and drafts tokens with it.

The 31 ``mtp.*`` tensors (~5.2 GB) are spread over 28 of the release's 131 files, so by
default only their byte ranges are fetched (HF_TOKEN is sent if set). ``--source`` takes a
local safetensors file with them instead.

Usage:
    uv run python scripts/convert_qwen4_exp_mtp.py \\
        ~/.exo/models/mlx-community--Qwen3.8-Flash-Next-4bit

    # optional: draft only among the tokens your own outputs use most (a smaller output
    # layer for the head); any number of UTF-8 text files
    uv run python scripts/convert_qwen4_exp_mtp.py \\
        ~/.exo/models/mlx-community--Qwen3.8-Flash-Next-4bit --draft-vocab outputs/*.txt

Copy the output folder to every node that may run the last pipeline stage.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import cast

import mlx.core as mx
import numpy as np

from exo.worker.engines.mlx.models.qwen3.qwen4_exp_mtp import (
    DRAFT_VOCAB_FILE,
    mtp_weights_path,
)

_SIZE = {"BF16": 2, "F16": 2, "F32": 4}
_CHUNK = 16 * 2**20
# The routers are 8-bit (group 64) in the mlx-community conversion, the other linears 4-bit
_ROUTERS = ("mlp.gate", "mlp.shared_expert_gate")


def _get(url: str, start: int, end: int) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120
            ) as response:
                data = cast(bytes, response.read())
            if len(data) != end - start + 1:
                raise OSError(f"short read: {len(data)} of {end - start + 1} bytes")
            return data
        except OSError:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError


def fetch_mtp_tensors(repo: str, out: Path) -> None:
    """Write the repo's ``mtp.*`` tensors (names, dtypes and shapes as stored) to ``out``,
    reading only their byte ranges."""
    base = f"https://huggingface.co/{repo}/resolve/main/"
    with urllib.request.urlopen(base + "model.safetensors.index.json") as response:
        index = cast(dict[str, dict[str, str]], json.loads(response.read()))
    files = {k: v for k, v in index["weight_map"].items() if k.startswith("mtp.")}
    plan: list[tuple[str, str, str, list[int], int, int]] = []
    for name in sorted(set(files.values())):
        (length,) = struct.unpack("<Q", _get(base + name, 0, 7))
        header = cast(
            dict[str, dict[str, object]],
            json.loads(_get(base + name, 8, 8 + length - 1)),
        )
        for key, info in header.items():
            if files.get(key) == name:
                start, end = cast(list[int], info["data_offsets"])
                dtype, shape = cast(str, info["dtype"]), cast(list[int], info["shape"])
                plan.append((key, name, dtype, shape, 8 + length + start, end - start))
    if sorted(p[0] for p in plan) != sorted(files):
        raise ValueError(f"{repo}: some mtp.* tensors are not in their files")
    plan.sort()
    offsets: dict[str, object] = {}
    position = 0
    for key, _, dtype, shape, _, size in plan:
        offsets[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [position, position + size],
        }
        position += size
    header_bytes = json.dumps({"__metadata__": {"format": "pt"}, **offsets}).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    print(f"Fetching {len(plan)} tensors ({position / 1e9:.2f} GB) from {repo}")
    with out.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for key, name, _, _, start, size in plan:
            for offset in range(start, start + size, _CHUNK):
                fh.write(
                    _get(base + name, offset, min(start + size, offset + _CHUNK) - 1)
                )
            print(f"  {key}")


def convert(source: Path, model_dir: Path) -> dict[str, mx.array]:
    """The head's tensors in the layout and quantization of the MLX model at
    ``model_dir``, checked against the trunk's own full-attention layer 3 and final mixer.
    """
    config = cast(
        dict[str, object], json.loads((model_dir / "config.json").read_text())
    )
    quant = cast(dict[str, object], config["quantization"])
    group, bits = cast(int, quant["group_size"]), cast(int, quant["bits"])
    tensors: dict[str, mx.array] = {}
    for name, value in cast(dict[str, mx.array], mx.load(str(source))).items():
        if not name.startswith("mtp."):
            continue
        # The fused experts as exo's qwen4_exp sanitize splits the trunk's
        if name.endswith("mlp.experts.gate_up_proj"):
            gate, up = mx.split(value, 2, axis=-2)
            base = name.removesuffix("experts.gate_up_proj")
            tensors[base + "switch_mlp.gate_proj.weight"] = gate
            tensors[base + "switch_mlp.up_proj.weight"] = up
        elif name.endswith("mlp.experts.down_proj"):
            base = name.removesuffix("experts.down_proj")
            tensors[base + "switch_mlp.down_proj.weight"] = value
        else:
            tensors[name] = value
    out: dict[str, mx.array] = {}
    for name, weight in sorted(tensors.items()):
        if not (name.endswith(".weight") and weight.ndim >= 2):
            out[name] = weight
            continue
        module = name.removesuffix(".weight")
        if module.endswith(_ROUTERS):
            override = quant.get(
                "language_model.model.layers.3." + module.split("layers.0.", 1)[-1]
            )
            g, b = (
                (
                    cast(int, cast(dict[str, int], override)["group_size"]),
                    cast(int, cast(dict[str, int], override)["bits"]),
                )
                if isinstance(override, dict)
                else (64, 8)
            )
        else:
            g, b = group, bits
        packed, scales, biases = mx.quantize(
            weight.astype(mx.bfloat16), group_size=g, bits=b
        )
        mx.eval(packed, scales, biases)
        out[name], out[module + ".scales"], out[module + ".biases"] = (
            packed,
            scales,
            biases,
        )

    weight_map = cast(
        dict[str, str],
        json.loads((model_dir / "model.safetensors.index.json").read_text())[
            "weight_map"
        ],
    )
    shards: dict[str, dict[str, mx.array]] = {}
    problems: list[str] = []
    for name, array in out.items():
        if name.startswith("mtp.layers.0."):
            trunk = "language_model.model.layers.3." + name.removeprefix(
                "mtp.layers.0."
            )
        elif name.startswith("mtp.hyper_connection_mixer."):
            trunk = "language_model.model." + name.removeprefix("mtp.")
        else:
            continue
        if trunk not in weight_map:
            problems.append(f"{name}: the trunk has no {trunk}")
            continue
        shard = weight_map[trunk]
        if shard not in shards:
            shards[shard] = cast(dict[str, mx.array], mx.load(str(model_dir / shard)))
        ref = shards[shard][trunk]
        if tuple(ref.shape) != tuple(array.shape) or ref.dtype != array.dtype:
            problems.append(
                f"{name}: {tuple(array.shape)} {array.dtype}, "
                f"trunk {tuple(ref.shape)} {ref.dtype}"
            )
    if problems:
        raise ValueError(
            "the converted head doesn't match the trunk:\n" + "\n".join(problems)
        )
    return out


def draft_vocabulary(model_dir: Path, texts: list[Path], size: int) -> np.ndarray:
    """The ``size`` tokens the texts use most, plus every special and EOS token."""
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    counts: collections.Counter[int] = collections.Counter()
    for path in texts:
        counts.update(tokenizer.encode(path.read_text(errors="replace")))
    ids = {token for token, _ in counts.most_common(size)}
    inner = tokenizer._tokenizer  # pyright: ignore[reportPrivateUsage]
    ids |= {int(i) for i in getattr(inner, "all_special_ids", []) or []}
    ids |= {int(i) for i in getattr(inner, "added_tokens_decoder", {}) or {}}
    ids |= {int(i) for i in tokenizer.eos_token_ids or []}
    return np.array(sorted(ids), dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("model", type=Path, help="exo model directory (MLX, quantized)")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="a local safetensors file with the BF16 mtp.* tensors (default: fetch them)",
    )
    parser.add_argument("--repo", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument(
        "--output", type=Path, default=None, help="override the output path"
    )
    parser.add_argument(
        "--draft-vocab",
        type=Path,
        nargs="*",
        default=[],
        help="text files whose most used tokens the head drafts from",
    )
    parser.add_argument("--draft-vocab-size", type=int, default=32768)
    args = parser.parse_args()

    model_dir = cast(Path, args.model).expanduser()
    output = cast(Path | None, args.output) or mtp_weights_path(model_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        source = cast(Path | None, args.source)
        if source is None:
            source = Path(scratch) / "mtp-bf16.safetensors"
            fetch_mtp_tensors(cast(str, args.repo), source)
        mx.set_default_device(mx.cpu)  # leaves the GPU to a running exo
        params = convert(source.expanduser(), model_dir)
    mx.save_safetensors(str(output), params, metadata={"format": "mlx"})
    print(
        f"Wrote {len(params)} tensors ({output.stat().st_size / 1e9:.2f} GB) to {output}"
    )
    texts = cast(list[Path], args.draft_vocab)
    if texts:
        vocab = draft_vocabulary(model_dir, texts, cast(int, args.draft_vocab_size))
        np.save(output.with_name(DRAFT_VOCAB_FILE), vocab)
        print(
            f"Wrote a {len(vocab)}-token draft vocabulary to {output.with_name(DRAFT_VOCAB_FILE)}"
        )


if __name__ == "__main__":
    main()
