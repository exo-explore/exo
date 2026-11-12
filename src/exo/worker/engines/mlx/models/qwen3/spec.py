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
    # Qwen3.8-Flash-Next's vision tower is Qwen3.5's (same config fields and weight names).
    # mlx-vlm has no qwen4_exp module, so without the alias the vision encoder doesn't load
    # and images are dropped from the prompt.
    mlx_vlm_model_type_aliases=(("qwen4_exp", "qwen3_5"),),
    eos_rules=(
        # For Qwen3.5 / Qwen3.6 / Qwen3.8: 248046 (<|im_end|>), 248044 (<|endoftext|>)
        EosRule(
            ("qwen3.5", "qwen-3.5", "qwen3.6", "qwen-3.6", "qwen3.8", "qwen-3.8"),
            (248046, 248044),
        ),
    ),
)
