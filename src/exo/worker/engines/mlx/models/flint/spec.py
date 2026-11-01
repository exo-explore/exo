from exo.worker.engines.mlx.models.spec import EosRule, FamilySpec

SPEC = FamilySpec(
    name="flint",
    model_types=frozenset({"flint"}),
    tensor_parallel_architectures=frozenset({"FlintForCausalLM"}),
    eos_rules=(
        # 150001 <|end_of_text|>, 150003 <|user|>, 150005 <|observation|>,
        # 150018 <|developer|>: config.json's eos_token_id. Flint ends a turn by starting
        # the next one, and its tokenizer config names only <|end_of_text|>.
        EosRule(("recursivesi/flint", "/flint-"), (150001, 150003, 150005, 150018)),
    ),
)
