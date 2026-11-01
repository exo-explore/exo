from exo.worker.engines.mlx.models.spec import FamilySpec

SPEC = FamilySpec(
    name="minimax",
    # minimax_m3_vl: MiniMax M3 and M3.1, whose checkpoints carry a vision tower that exo
    # doesn't load (text only).
    model_types=frozenset({"minimax", "minimax_m2", "minimax_m3_vl"}),
    tensor_parallel_architectures=frozenset(
        {"MiniMaxM2ForCausalLM", "MiniMaxM3SparseForConditionalGeneration"}
    ),
)
