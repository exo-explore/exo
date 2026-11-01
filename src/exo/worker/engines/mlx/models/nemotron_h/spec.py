from exo.worker.engines.mlx.models.spec import FamilySpec

SPEC = FamilySpec(
    name="nemotron_h",
    model_types=frozenset({"nemotron_h", "nemotron_h_omni"}),
    tensor_parallel_architectures=frozenset({"NemotronHForCausalLM"}),
)
