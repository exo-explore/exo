from exo.worker.engines.mlx.models.spec import EosRule, FamilySpec

SPEC = FamilySpec(
    name="glm",
    model_types=frozenset({"glm4_moe", "glm4_moe_lite", "glm_moe_dsa"}),
    tensor_parallel_architectures=frozenset(
        {"Glm4MoeLiteForCausalLM", "GlmMoeDsaForCausalLM"}
    ),
    eos_rules=(
        # 154820: <|endoftext|>, 154827: <|user|>, 154829: <|observation|>
        EosRule(("glm-5",), (154820, 154827, 154829)),
        # For GLM-4.7 and older
        EosRule(("glm",), (151336, 151329, 151338)),
    ),
)
