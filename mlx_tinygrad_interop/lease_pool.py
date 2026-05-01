"""Compatibility shim for the moved lease-pool implementation.

Reusable interop code now lives under `mlx_tinygrad_interop.lib`.
Benchmarks and tests remain at the package top level.
"""

from mlx_tinygrad_interop.lib.lease_pool import *  # noqa: F401,F403
