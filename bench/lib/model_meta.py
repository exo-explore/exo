"""Fetch HuggingFace model metadata for benchmark planning.

Two pieces of metadata drive every benchmark we run:

  1. **Total weight size** — used to derive ``min-memory`` and ``min-disk``
     constraints when picking a host. We sum the sizes of all
     ``.safetensors`` (or ``.bin``) shards from the repo's file listing.
  2. **Max position embeddings** — the model's training context length.
     Used to bound a context-scaling sweep at the model's max context, and
     to derive a sensible Δ given a target step count.

The fetcher uses the ``huggingface_hub`` python API, which talks to the
public HF Hub HTTPS endpoints — no exo cluster required, no download
of weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

# Files that count toward the on-disk weight footprint.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".npz")


@dataclass(frozen=True)
class ModelMeta:
    """Subset of HF metadata that a benchmark needs."""

    model_id: str
    total_weight_bytes: int
    max_position_embeddings: int
    num_hidden_layers: int
    raw_config: dict[str, Any] = field(default_factory=dict)

    @property
    def total_weight_gb(self) -> float:
        return self.total_weight_bytes / (1024**3)

    @property
    def memory_constraint_gb(self) -> float:
        """Estimated minimum host memory to hold weights + overhead.

        Picks the model size + 30 % headroom (KV cache, activations,
        framework bookkeeping). Rounded up to the next whole GiB.
        """
        return float(int(self.total_weight_gb * 1.30) + 1)

    @property
    def disk_constraint_gb(self) -> float:
        """Disk space the host must have free for the download."""
        return float(int(self.total_weight_gb * 1.10) + 1)


def _read_config_json(model_id: str) -> dict[str, Any]:
    from huggingface_hub import (
        hf_hub_download,  # type: ignore[reportUnknownVariableType]
    )

    raw_path = hf_hub_download(repo_id=model_id, filename="config.json", dry_run=False)
    with open(raw_path) as f:
        loaded: Any = json.load(f)  # type: ignore[reportAny]
    return cast("dict[str, Any]", loaded) if isinstance(loaded, dict) else {}


def _sum_weight_sizes(model_id: str) -> int:
    """Sum sizes of all weight-shard files in the repo's file listing."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo_id=model_id, files_metadata=True)
    siblings = info.siblings or []
    total = 0
    for sib in siblings:
        rfilename = getattr(sib, "rfilename", None)
        size = getattr(sib, "size", None)
        if not isinstance(rfilename, str) or not isinstance(size, int):
            continue
        if any(rfilename.endswith(suf) for suf in _WEIGHT_SUFFIXES):
            total += size
    return total


def _first_int(config: dict[str, Any], *keys: str) -> int:
    """Return the first key from ``config`` that holds a usable positive int."""
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                continue
            if parsed > 0:
                return parsed
    return 0


def fetch_model_meta(model_id: str) -> ModelMeta:
    """Fetch the metadata our benchmarks care about for ``model_id``.

    Args:
        model_id: HuggingFace repo id, e.g. ``mlx-community/Qwen3-30B-A3B-4bit``.

    Returns:
        Populated :class:`ModelMeta`.

    Raises:
        Exception: any HTTP / parse error from ``huggingface_hub`` propagates.
    """
    config = _read_config_json(model_id)
    return ModelMeta(
        model_id=model_id,
        total_weight_bytes=_sum_weight_sizes(model_id),
        max_position_embeddings=_first_int(
            config,
            "max_position_embeddings",
            "max_seq_len",
            "model_max_length",
            "n_positions",
        ),
        num_hidden_layers=_first_int(
            config,
            "num_hidden_layers",
            "num_layers",
            "n_layer",
            "n_layers",
            "num_decoder_layers",
        ),
        raw_config=config,
    )


def derive_context_ramp(
    meta: ModelMeta,
    *,
    num_steps: int,
    fraction_of_max: float = 1.0,
    min_pp_step: int = 256,
    round_to: int = 256,
) -> tuple[int, int]:
    """Pick ``(pp_step, num_steps)`` covering ``fraction_of_max`` of the context.

    Δ is rounded down to the nearest ``round_to`` so the per-step prompt is a
    clean number, and clamped to ``min_pp_step`` for tiny-context models.
    """
    if meta.max_position_embeddings <= 0:
        raise ValueError(
            f"{meta.model_id} reports max_position_embeddings=0 in config.json"
        )
    if not (0.0 < fraction_of_max <= 1.0):
        raise ValueError(f"fraction_of_max must be in (0, 1], got {fraction_of_max}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be >0, got {num_steps}")

    target_max = int(meta.max_position_embeddings * fraction_of_max)
    raw_step = max(min_pp_step, target_max // num_steps)
    pp_step = (raw_step // round_to) * round_to or round_to
    return pp_step, num_steps


def derive_cold_controls(
    meta: ModelMeta,
    *,
    pp_step: int,
    num_steps: int,
    count: int = 4,
) -> tuple[int, ...]:
    """Pick ``count`` evenly-spaced cold-control points across the ramp.

    Always includes the largest ramp point (``pp_step * num_steps``).
    Returns control pp values in ascending order, deduped.
    """
    if count <= 0:
        return ()
    max_pp = pp_step * num_steps
    if count == 1:
        return (max_pp,)
    spaced = sorted({(max_pp * (i + 1)) // count for i in range(count)})
    # Filter out anything below pp_step (a control at <Δ is meaningless).
    return tuple(p for p in spaced if p >= pp_step)
