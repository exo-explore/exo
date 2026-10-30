from exo.worker.engines.mlx.models.spec import FamilySpec

# The tokenizer's eos_token (<｜hy_eos:opensource｜>, 120025) is config.json's
# eos_token_id and ends every assistant turn, tool calls included: no EOS rule needed.
SPEC = FamilySpec(
    name="hy_v3",
    model_types=frozenset({"hy_v3"}),
    tensor_parallel_architectures=frozenset({"HYV3ForCausalLM"}),
)
