from exo.worker.engines.mlx.models.spec import EosRule, FamilySpec

SPEC = FamilySpec(
    name="gpt_oss",
    model_types=frozenset({"gpt_oss"}),
    tensor_parallel_architectures=frozenset({"GptOssForCausalLM"}),
    eos_rules=(EosRule(("gpt-oss",), (200002, 200012)),),
)
