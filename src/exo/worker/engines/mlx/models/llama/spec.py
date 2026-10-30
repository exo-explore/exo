from exo.worker.engines.mlx.models.spec import FamilySpec

SPEC = FamilySpec(
    name="llama",
    model_types=frozenset({"llama", "ministral3"}),
    tensor_parallel_architectures=frozenset({"LlamaForCausalLM"}),
)
