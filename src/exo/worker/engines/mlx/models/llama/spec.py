from exo.worker.engines.mlx.models.spec import AddedEosTokens, FamilySpec

# Llama 3.x checkpoints by (lowercased) model id: Meta's 3.0-3.3 and conversions or
# fine-tunes that keep the name, e.g. mlx-community/Llama-3.2-3B-Instruct-4bit and
# mlx-community/Meta-Llama-3.1-8B-Instruct-4bit. ("llama-3." rather than "llama-3", which
# would also match a "...llama-3b" id.)
LLAMA_3_MODEL_ID_SUBSTRINGS = ("llama-3.", "llama-3-")

# The eos_token_id of the checkpoints' config.json and generation_config.json.
# tokenizer_config.json names only <|eot_id|>. <|eom_id|> ends a message that hands over to
# a tool (an ipython call, or a JSON tool call in a prompt with tools); without it the model
# went on past its tool call into a new assistant turn of its own. Each is added to the
# tokenizer's own EOS token only where the tokenizer has it at that id: fine-tunes named
# after Llama 3 can end their turns with another token (ChatML ones with <|im_end|>) or
# give the id to another token (128008 is <tools> in one, and an unused reserved token in
# Llama 3.0).
LLAMA_3_EOS_TOKENS = (
    (128001, "<|end_of_text|>"),
    (128008, "<|eom_id|>"),
    (128009, "<|eot_id|>"),
)

SPEC = FamilySpec(
    name="llama",
    model_types=frozenset({"llama", "ministral3"}),
    tensor_parallel_architectures=frozenset({"LlamaForCausalLM"}),
    added_eos_tokens=(AddedEosTokens(LLAMA_3_MODEL_ID_SUBSTRINGS, LLAMA_3_EOS_TOKENS),),
)
