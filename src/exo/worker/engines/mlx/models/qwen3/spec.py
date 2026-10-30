from exo.worker.engines.mlx.models.spec import EosRule, FamilySpec

SPEC = FamilySpec(
    name="qwen3",
    model_types=frozenset(
        {
            "qwen3",
            "qwen3_moe",
            "qwen3_next",
            "qwen3_vl",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen4_exp",
        }
    ),
    tensor_parallel_architectures=frozenset(
        {
            "Qwen3NextForCausalLM",
            "Qwen3MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
            "Qwen4ExpForConditionalGeneration",
        }
    ),
    eos_rules=(
        # For Qwen3.5 / Qwen3.6 / Qwen3.8: 248046 (<|im_end|>), 248044 (<|endoftext|>)
        EosRule(
            ("qwen3.5", "qwen-3.5", "qwen3.6", "qwen-3.6", "qwen3.8", "qwen-3.8"),
            (248046, 248044),
        ),
    ),
)
