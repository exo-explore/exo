from exo.worker.engines.mlx.patches.deepseek_v4_cache_extend import (
    patch_deepseek_v4_cache_extend,
)
from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch
from exo.worker.engines.mlx.patches.standard_yarn_rope import patch_yarn_rope

_applied = False


def apply_mlx_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    patch_yarn_rope()
    apply_batch_gen_patch()
    patch_deepseek_v4_cache_extend()
