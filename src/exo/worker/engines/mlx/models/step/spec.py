from exo.worker.engines.mlx.models.spec import FamilySpec

SPEC = FamilySpec(
    name="step",
    model_types=frozenset({"step3p5"}),
    tensor_parallel_architectures=frozenset({"Step3p5ForCausalLM"}),
)
