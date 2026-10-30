"""Which model ids are Hy3 checkpoints. Pure Python (no MLX)."""


def is_hy3_model_id(model_id: str) -> bool:
    """Repositories named ``Hy3`` or ``Hy3-...`` (tencent/Hy3, tencent/Hy3-FP8,
    XavierLocalAI/Hy3-4bit, mlx-community/Hy3-oQ2e, ...), not Hunyuan3D's ``hy3d``."""
    name = model_id.rsplit("/", 1)[-1].lower()
    return name == "hy3" or name.startswith(("hy3-", "hy3_"))
