"""
Model transfer via MLX distributed all_sum.

Provides two transfer modes:
1. Metadata file transfer: broadcast small files (config.json, tokenizer, etc.) to disk
2. Weight tensor broadcast: stream weight tensors directly into memory via all_sum
3. Full file transfer: broadcast all files (including safetensors) to disk

All functions are collective operations — every rank in the group must call them.

Protocol relies on all_sum: source has real data, receivers have zeros.
all_sum(source + zeros) = source data on all ranks.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Final, cast

import mlx.core as mx

from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.models.model_cards import ModelId
from exo.worker.runner.bootstrap import logger

Group = mx.distributed.Group

CHUNK_SIZE: Final[int] = 100 * 1024 * 1024  # 100 MB

# File extensions that are metadata (not weight data)
_METADATA_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".json",
        ".txt",
        ".md",
        ".model",  # sentencepiece tokenizer.model
        ".py",
    }
)


def _all_sum_cpu(x: mx.array, group: Group) -> mx.array:
    """all_sum on CPU stream to avoid GPU memory pressure."""
    return mx.distributed.all_sum(
        x, stream=mx.default_stream(mx.Device(mx.cpu)), group=group
    )


def _is_metadata_file(filename: str) -> bool:
    """Check if a file is a metadata file (not a weight file).

    Excludes safetensors index files (e.g. model.safetensors.index.json) since
    they reference .safetensors shard files that won't exist on the receiver.
    """
    if filename.endswith(".safetensors.index.json"):
        return False
    _, ext = os.path.splitext(filename)
    return ext.lower() in _METADATA_EXTENSIONS


def has_weight_files(model_path: Path) -> bool:
    """Check if model directory has both config and weight files."""
    if not model_path.exists():
        return False
    has_config = (model_path / "config.json").exists()
    has_weights = any(model_path.glob("*.safetensors")) or any(
        model_path.glob("**/*.safetensors")
    )
    return has_config and has_weights


def has_metadata_files(model_path: Path) -> bool:
    """Check if model directory has at least config.json."""
    return model_path.exists() and (model_path / "config.json").exists()


def model_path_for_id(model_id: ModelId) -> Path:
    """Get model path without requiring directory to exist (unlike build_model_path)."""
    return EXO_MODELS_DIR / model_id.normalize()


# ---------------------------------------------------------------------------
# Coordination
# ---------------------------------------------------------------------------


def coordinate_transfer(group: Group, has_local_model: bool) -> tuple[bool, int]:
    """
    Determine if a transfer is needed and which rank is the source.

    All ranks must call this function (uses collective all_sum).

    Returns:
        (needs_transfer, source_rank) — source_rank is the lowest rank
        that has the model. needs_transfer is True if any rank is missing it.
    """
    all_sum = partial(_all_sum_cpu, group=group)
    world_size = group.size()

    # Each rank broadcasts a one-hot vector at its position if it has the model
    bitmask = mx.zeros(world_size, dtype=mx.int32)
    if has_local_model:
        bitmask = bitmask.at[group.rank()].add(1)
    summed = all_sum(bitmask)
    mx.eval(summed)

    # Find who has it
    has_model_flags: list[int] = summed.tolist()  # type: ignore[assignment]
    total_have = sum(has_model_flags)

    if total_have == 0:
        raise RuntimeError(
            "No rank has the model files — cannot transfer. "
            "At least one node must have downloaded the model."
        )

    if total_have == world_size:
        logger.info("All ranks have model files, no transfer needed")
        return False, 0

    # Source is the lowest rank that has the model
    source_rank = next(i for i, flag in enumerate(has_model_flags) if flag > 0)
    logger.info(
        f"Transfer needed: source_rank={source_rank}, "
        f"{total_have}/{world_size} ranks have model"
    )
    return True, source_rank


# ---------------------------------------------------------------------------
# Low-level: broadcast bytes via all_sum
# ---------------------------------------------------------------------------


def _broadcast_int(value: int, group: Group, is_source: bool) -> int:
    """Broadcast a single int from source to all ranks."""
    all_sum = partial(_all_sum_cpu, group=group)
    arr = mx.array([value if is_source else 0], dtype=mx.int64)
    result = all_sum(arr)
    mx.eval(result)
    return int(result.item())


def _broadcast_bytes(data: bytes, group: Group, is_source: bool) -> bytes:
    """Broadcast a byte string from source to all ranks."""
    all_sum = partial(_all_sum_cpu, group=group)

    # Broadcast length
    length = _broadcast_int(len(data) if is_source else 0, group, is_source)
    if length == 0:
        return b""

    # Broadcast data
    if is_source:
        arr = mx.array(list(data), dtype=mx.uint8)
    else:
        arr = mx.zeros(length, dtype=mx.uint8)
    result = all_sum(arr)
    mx.eval(result)
    return bytes(cast(list[int], result.tolist()))


def _broadcast_json(obj: object, group: Group, is_source: bool) -> object:
    """Broadcast a JSON-serializable object from source to all ranks."""
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8") if is_source else b""
    received = _broadcast_bytes(data, group, is_source)
    result: object = json.loads(received)  # pyright: ignore[reportAny]
    return result


# ---------------------------------------------------------------------------
# File transfer to disk (used by metadata transfer and full file transfer)
# ---------------------------------------------------------------------------


def _build_manifest(
    model_path: Path, metadata_only: bool = False
) -> list[dict[str, str | int]]:
    """Build a list of files in the model directory with their relative paths and sizes."""
    manifest: list[dict[str, str | int]] = []
    for root, _dirs, files in os.walk(model_path):
        for fname in sorted(files):
            if metadata_only and not _is_metadata_file(fname):
                continue
            full_path = Path(root) / fname
            rel_path = str(full_path.relative_to(model_path))
            manifest.append(
                {
                    "path": rel_path,
                    "size": full_path.stat().st_size,
                }
            )
    return manifest


def _transfer_file_to_disk(
    source_path: Path,
    rel_path: str,
    file_size: int,
    group: Group,
    is_source: bool,
    dest_path: Path,
) -> None:
    """Transfer a single file chunk-by-chunk via all_sum. Source reads from disk, receivers write to dest_path."""
    all_sum = partial(_all_sum_cpu, group=group)

    if is_source:
        src_file = source_path / rel_path
        with open(src_file, "rb") as f:
            offset = 0
            while offset < file_size:
                chunk_bytes = min(CHUNK_SIZE, file_size - offset)
                data = f.read(chunk_bytes)
                if not data:
                    break
                # Broadcast chunk size then data
                size_arr = mx.array([len(data)], dtype=mx.int64)
                mx.eval(all_sum(size_arr))
                chunk_arr = mx.array(list(data), dtype=mx.uint8)
                result = all_sum(chunk_arr)
                mx.eval(result)
                offset += len(data)
            # Signal end of file
            mx.eval(all_sum(mx.array([0], dtype=mx.int64)))
    else:
        dst_file = dest_path / rel_path
        os.makedirs(dst_file.parent, exist_ok=True)
        with open(dst_file, "wb") as f:
            while True:
                size_arr = all_sum(mx.zeros(1, dtype=mx.int64))
                mx.eval(size_arr)
                chunk_size = int(size_arr.item())
                if chunk_size == 0:
                    break
                chunk_data = all_sum(mx.zeros(chunk_size, dtype=mx.uint8))
                mx.eval(chunk_data)
                f.write(bytes(cast(list[int], chunk_data.tolist())))


def _transfer_files_to_disk(
    model_path: Path,
    group: Group,
    is_source: bool,
    metadata_only: bool = False,
) -> None:
    """
    Transfer files from source to all receivers' disk.

    Source broadcasts a manifest then each file. Receivers write to a temp dir
    then atomically move files to model_path.
    """
    # Broadcast manifest
    if is_source:
        source_manifest = _build_manifest(model_path, metadata_only=metadata_only)
    else:
        source_manifest = []
    manifest = cast(
        list[dict[str, str | int]],
        _broadcast_json(source_manifest if is_source else None, group, is_source),
    )

    if not manifest:
        logger.info("No files to transfer")
        return

    logger.info(
        f"Transferring {len(manifest)} files ({'metadata only' if metadata_only else 'all'})"
    )

    # Receivers write to temp dir for atomic move
    temp_dir: Path | None = None
    if not is_source:
        os.makedirs(model_path.parent, exist_ok=True)
        temp_dir = Path(
            tempfile.mkdtemp(
                dir=model_path.parent,
                prefix=f".transfer_{model_path.name}_",
            )
        )

    try:
        for entry in manifest:
            rel_path = str(entry["path"])
            file_size = int(entry["size"])
            logger.info(f"  {rel_path} ({file_size} bytes)")
            _transfer_file_to_disk(
                source_path=model_path,
                rel_path=rel_path,
                file_size=file_size,
                group=group,
                is_source=is_source,
                dest_path=temp_dir if temp_dir is not None else model_path,
            )

        # Atomic move from temp to final
        if not is_source and temp_dir is not None:
            os.makedirs(model_path, exist_ok=True)
            for entry in manifest:
                rel_path = str(entry["path"])
                src = temp_dir / rel_path
                dst = model_path / rel_path
                os.makedirs(dst.parent, exist_ok=True)
                os.replace(src, dst)
            logger.info(
                f"Transfer complete: {len(manifest)} files moved to {model_path}"
            )
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public: metadata file transfer (config.json, tokenizer, etc.)
# ---------------------------------------------------------------------------


def transfer_metadata_files(
    model_path: Path, group: Group, is_source: bool
) -> None:
    """
    Transfer metadata files (config.json, tokenizer files, etc.) to receivers' disk.

    All ranks must call this function (collective operation).
    Only the designated source (is_source=True) should send; all others receive.
    """
    _transfer_files_to_disk(
        model_path, group, is_source=is_source, metadata_only=True
    )


# ---------------------------------------------------------------------------
# Public: full file transfer to disk (Feature 1)
# ---------------------------------------------------------------------------


def transfer_all_files(model_path: Path, group: Group, is_source: bool) -> None:
    """
    Transfer ALL model files (including safetensors) to receivers' disk.

    All ranks must call this function (collective operation).
    Only the designated source (is_source=True) should send; all others receive.
    """
    _transfer_files_to_disk(
        model_path, group, is_source=is_source, metadata_only=False
    )


# ---------------------------------------------------------------------------
# Public: weight tensor broadcast to memory (Feature 2)
# ---------------------------------------------------------------------------


def _parse_mx_dtype(dtype_str: str) -> mx.Dtype:
    """Convert a dtype string like 'float16' or 'mlx.core.float16' to mx.Dtype."""
    # Strip 'mlx.core.' prefix if present
    name = dtype_str.split(".")[-1]
    dtype = getattr(mx, name, None)
    if dtype is None:
        raise ValueError(f"Unknown MLX dtype: {dtype_str}")
    return dtype  # type: ignore[return-value]


def broadcast_model_weights(
    model_path: Path,
    group: Group,
    is_source: bool,
) -> dict[str, mx.array]:
    """
    Broadcast model weight tensors from source rank to all receivers' memory.

    Source loads weights from .safetensors files on disk and broadcasts each
    tensor via all_sum. Receivers receive tensors directly as mx.arrays in
    memory — no disk write for weight data.

    All ranks must call this function (collective operation).
    Only the designated source (is_source=True) should send; all others receive.

    Returns:
        dict mapping weight names to mx.arrays (on all ranks).
    """
    all_sum = partial(_all_sum_cpu, group=group)

    # Source lazily loads weights (data stays on disk until mx.eval per tensor)
    weights: dict[str, mx.array] = {}
    if is_source:
        weight_files = sorted(model_path.glob("*.safetensors"))
        if not weight_files:
            weight_files = sorted(model_path.glob("**/*.safetensors"))
        for wf in weight_files:
            loaded = cast(dict[str, mx.array], mx.load(str(wf), lazy=True))  # pyright: ignore[reportCallIssue]
            weights.update(loaded)
        logger.info(
            f"Source mapped {len(weights)} weight tensors from {len(weight_files)} files"
        )

    # Broadcast weight metadata: {name: {shape, dtype}}
    if is_source:
        source_meta: dict[str, dict[str, Any]] = {
            name: {"s": list(tensor.shape), "d": str(tensor.dtype)}
            for name, tensor in weights.items()
        }
    else:
        source_meta = {}
    meta = cast(
        dict[str, dict[str, Any]],
        _broadcast_json(source_meta if is_source else None, group, is_source),
    )

    logger.info(f"Broadcasting {len(meta)} weight tensors")

    # Broadcast each tensor in sorted order (deterministic across ranks).
    # Source loads one tensor at a time from disk (lazy), broadcasts it,
    # then drops the reference so only one tensor is in flight at a time.
    result: dict[str, mx.array] = {}
    for i, name in enumerate(sorted(meta.keys())):
        info = meta[name]
        shape = cast(list[int], info["s"])
        dtype_str = cast(str, info["d"])
        dtype = _parse_mx_dtype(dtype_str)

        if is_source:
            tensor = weights.pop(name)  # pop to free lazy ref after broadcast
            mx.eval(tensor)  # loads from disk
        else:
            tensor = mx.zeros(shape, dtype=dtype)

        broadcasted = all_sum(tensor)
        mx.eval(broadcasted)
        result[name] = broadcasted

        if (i + 1) % 100 == 0:
            logger.info(f"  Broadcast {i + 1}/{len(meta)} tensors")

    logger.info(f"Weight broadcast complete: {len(result)} tensors")
    return result
