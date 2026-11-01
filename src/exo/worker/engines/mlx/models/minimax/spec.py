from exo.worker.engines.mlx.models.spec import FamilySpec

SPEC = FamilySpec(
    name="minimax",
    model_types=frozenset({"minimax", "minimax_m2"}),
    tensor_parallel_architectures=frozenset({"MiniMaxM2ForCausalLM"}),
)
