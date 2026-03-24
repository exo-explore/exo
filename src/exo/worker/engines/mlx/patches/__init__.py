from exo.worker.engines.mlx.patches.high_precision_gdn_softplus import (
    patch_gdn_softplus,
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
    patch_gdn_softplus()
    apply_batch_gen_patch()
