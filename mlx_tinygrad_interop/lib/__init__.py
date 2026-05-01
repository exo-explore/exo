"""Reusable MLX <-> tinygrad interop helpers.

Benchmarks, stress harnesses, and tests stay in the top-level
`mlx_tinygrad_interop/` package. Reusable bridge and lease-pool code lives
under `mlx_tinygrad_interop/lib/`.
"""

from mlx_tinygrad_interop.lib.lease_pool import (
  MlxToTinygradCopyKey,
  MlxToTinygradCopyLeasePool,
  MlxToTinygradCopyLeasePools,
  MlxToTinygradLease,
  MlxToTinygradLeaseKey,
  MlxToTinygradLeasePool,
  MlxToTinygradLeasePools,
)
from mlx_tinygrad_interop.lib.tensor_bridge import mlx_to_tinygrad, sync_mlx, sync_tinygrad, tinygrad_to_mlx

__all__ = [
  "MlxToTinygradCopyKey",
  "MlxToTinygradCopyLeasePool",
  "MlxToTinygradCopyLeasePools",
  "MlxToTinygradLease",
  "MlxToTinygradLeaseKey",
  "MlxToTinygradLeasePool",
  "MlxToTinygradLeasePools",
  "mlx_to_tinygrad",
  "sync_mlx",
  "sync_tinygrad",
  "tinygrad_to_mlx",
]
