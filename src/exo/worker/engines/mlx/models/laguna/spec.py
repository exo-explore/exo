from exo.worker.engines.mlx.models.spec import EosRule, FamilySpec

SPEC = FamilySpec(
    name="laguna",
    model_types=frozenset({"laguna"}),
    tensor_parallel_architectures=frozenset({"LagunaForCausalLM"}),
    eos_rules=(
        # 2 〈|EOS|〉, 24 </assistant>: config.json's eos_token_id. Laguna ends a turn with
        # </assistant>, and its tokenizer config names only 〈|EOS|〉.
        EosRule(("/laguna-",), (2, 24)),
    ),
)
