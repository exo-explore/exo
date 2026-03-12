from vllm.tokenizers import TokenizerLike as TokenizerLike

INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET: int

def convert_prompt_ids_to_tokens(
    tokenizer: TokenizerLike, prompt_ids: list[int], skip_special_tokens: bool = False
) -> tuple[list[str], int, int]: ...
def convert_ids_list_to_tokens(
    tokenizer: TokenizerLike, token_ids: list[int]
) -> list[str]: ...
def detokenize_incrementally(
    tokenizer: TokenizerLike,
    all_input_ids: list[int],
    prev_tokens: list[str] | None,
    prefix_offset: int,
    read_offset: int,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> tuple[list[str], str, int, int]: ...
