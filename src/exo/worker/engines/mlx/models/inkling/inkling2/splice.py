# type: ignore
# Vendored from ml-explore/mlx-lm @ 0c4c99192230f1943dd468d297f0b8a64ae7b7c2 (PR #2086, Inkling 2), mlx_lm/models/inkling2/splice.py
# (splice.py: mlx_lm/layers/common.py). MIT licence, Copyright 2026 Apple Inc. Local change: imports only (see __init__.py).
# Copyright © 2026 Apple Inc.
# From mlx-lm 0c4c9919 (ml-explore/mlx-lm#2086), mlx_lm/layers/common.py, unchanged; exo's mlx-lm pin predates it.

import mlx.core as mx


def splice(
    embeddings: mx.array,
    inputs: mx.array,
    *,
    rows: mx.array,
    token_id: int,
) -> mx.array:
    """The embeddings with the rows, in order, at the token_id positions of inputs."""
    # A shape check only, so no host sync.
    if rows.shape[0] == 0:
        return embeddings
    mask = inputs == token_id
    # The row of each masked token: the number of masked tokens before it.
    idx = mx.cumsum(mask.flatten()).reshape(mask.shape) - 1
    idx = mx.clip(idx, 0, rows.shape[0] - 1)
    selected = rows[idx].astype(embeddings.dtype)
    return mx.where(mask[..., None], selected, embeddings)
