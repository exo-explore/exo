try:
    from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch
except ImportError:
    def apply_batch_gen_patch():
        pass

try:
    from exo.worker.engines.mlx.patches.standard_yarn_rope import patch_yarn_rope
except ImportError:
    def patch_yarn_rope():
        pass

_applied = False


def apply_mlx_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    patch_yarn_rope()
    apply_batch_gen_patch()
