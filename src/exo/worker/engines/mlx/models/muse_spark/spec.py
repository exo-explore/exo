from exo.worker.engines.mlx.models.spec import EosRule, FamilySpec

SPEC = FamilySpec(
    name="muse_spark",
    model_types=frozenset({"muse_spark"}),
    tensor_parallel_architectures=frozenset({"MuseSparkForConditionalGeneration"}),
    eos_rules=(
        # 200001 <|end_of_text|>, 200008 <|eot|>: generation_config.json's eos_token_id.
        # The tokenizer config names only <|end_of_text|>, and a turn ends with <|eot|>
        # (<|eom|> ends a message within the turn: reasoning, a preamble, one tool call).
        EosRule(("muse-spark",), (200001, 200008)),
    ),
)
