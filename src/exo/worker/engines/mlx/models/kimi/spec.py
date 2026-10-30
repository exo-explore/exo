from exo.worker.engines.mlx.models.spec import EosRule, FamilySpec

SPEC = FamilySpec(
    name="kimi",
    # Kimi K2's checkpoints say DeepseekV3ForCausalLM, which the deepseek_v3 family lists.
    model_types=frozenset({"kimi_k2", "kimi_k25"}),
    eos_rules=(EosRule(("kimi-k2",), (163586,)),),
    mlx_vlm_model_type_aliases=(("kimi_k25", "kimi_vl"), ("kimi_k26", "kimi_vl")),
)
