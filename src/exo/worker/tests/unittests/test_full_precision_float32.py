"""Runner processes keep MLX's float32 matmuls in full float32 unless the user opts out."""

import os
import subprocess
import sys

import mlx.core as mx
import pytest

from exo.worker.runner.bootstrap import use_full_precision_float32


def test_unset_defaults_to_full_precision() -> None:
    environ: dict[str, str] = {}
    use_full_precision_float32(environ)
    assert environ == {"MLX_ENABLE_TF32": "0"}


@pytest.mark.parametrize("value", ["0", "1"])
def test_a_value_the_user_set_is_kept(value: str) -> None:
    environ = {"MLX_ENABLE_TF32": value}
    use_full_precision_float32(environ)
    assert environ == {"MLX_ENABLE_TF32": value}


# Run in a fresh process: MLX reads MLX_ENABLE_TF32 once, at its first matmul.
_PROBE = """
import os
from exo.worker.runner.bootstrap import use_full_precision_float32
use_full_precision_float32(os.environ)
import mlx.core as mx
import numpy as np
rng = np.random.default_rng(0)
a = rng.standard_normal((1024, 1024)).astype(np.float32)
b = rng.standard_normal((1024, 1024)).astype(np.float32)
got = np.array(mx.array(a) @ mx.array(b))
want = a.astype(np.float64) @ b.astype(np.float64)
print(float(np.abs(got - want).max() / np.abs(want).max()))
"""


@pytest.mark.skipif(
    sys.platform != "darwin" or not mx.metal.is_available(), reason="needs Metal"
)
def test_float32_matmul_in_a_runner_process_is_full_precision() -> None:
    environ = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=environ,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    error = float(result.stdout.strip().splitlines()[-1])
    # float32 sums over 1024 terms: ~1e-6 relative; TF32-like products: ~1e-3
    assert error < 1e-5, error
