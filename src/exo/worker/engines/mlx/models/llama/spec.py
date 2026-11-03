from exo.worker.engines.mlx.models.spec import EosRule, FamilySpec

# Llama 3.x checkpoints by (lowercased) model id: Meta's 3.0-3.3 and conversions or
# fine-tunes that keep the name, e.g. mlx-community/Llama-3.2-3B-Instruct-4bit and
# mlx-community/Meta-Llama-3.1-8B-Instruct-4bit. ("llama-3." rather than "llama-3", which
# would also match a "...llama-3b" id.)
LLAMA_3_MODEL_ID_SUBSTRINGS = ("llama-3.", "llama-3-")

# 128001 <|end_of_text|>, 128008 <|eom_id|>, 128009 <|eot_id|>: the eos_token_id of the
# checkpoints' config.json and generation_config.json. tokenizer_config.json names only
# <|eot_id|>. <|eom_id|> ends a message that hands over to a tool (an ipython call, or a
# JSON tool call in a prompt with tools); without it the model went on past its tool call
# into a new assistant turn of its own. In the Llama 3.0 vocabulary 128008 is an unused
# reserved token.
LLAMA_3_EOS_TOKEN_IDS = (128001, 128008, 128009)

SPEC = FamilySpec(
    name="llama",
    model_types=frozenset({"llama", "ministral3"}),
    tensor_parallel_architectures=frozenset({"LlamaForCausalLM"}),
    eos_rules=(EosRule(LLAMA_3_MODEL_ID_SUBSTRINGS, LLAMA_3_EOS_TOKEN_IDS),),
)
