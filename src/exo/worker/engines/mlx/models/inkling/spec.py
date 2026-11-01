from exo.worker.engines.mlx.models.spec import EosRule, FamilySpec

SPEC = FamilySpec(
    name="inkling",
    model_types=frozenset({"inkling_mm_model"}),
    tensor_parallel_architectures=frozenset({"InklingForConditionalGeneration"}),
    eos_rules=(
        # 200006 <|content_model_end_sampling|>: config.json's eos_token_id, which the
        # chat template writes after every assistant turn. The tokenizer config names no
        # EOS token at all.
        EosRule(("inkling",), (200006,)),
    ),
)
