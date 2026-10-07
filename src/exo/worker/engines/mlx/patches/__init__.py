from exo.worker.engines.mlx.patches.deepseek_v4_decode_kernels import (
    patch_deepseek_v4_decode_kernels,
)
from exo.worker.engines.mlx.patches.deepseek_v4_indexer import (
    patch_deepseek_v4_indexer,
)
from exo.worker.engines.mlx.patches.deepseek_v4_moe_gate import (
    patch_deepseek_v4_moe_gate,
)
from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch
from exo.worker.engines.mlx.patches.standard_yarn_rope import patch_yarn_rope
from exo.worker.engines.mlx.patches.switch_lhs_indices import patch_switch_lhs_indices

_applied = False


def apply_mlx_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    patch_yarn_rope()
    apply_batch_gen_patch()
    patch_deepseek_v4_moe_gate()
    patch_deepseek_v4_indexer()
    patch_deepseek_v4_decode_kernels()
    patch_switch_lhs_indices()
