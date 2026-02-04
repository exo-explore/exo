from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from loguru import logger

GLM_EOS_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
)
GEMMA_END_OF_TURN_TOKEN = "<end_of_turn>"


def _get_underlying_tokenizer(tokenizer_wrapper: Any) -> Any:
    # mlx_lm.TokenizerWrapper commonly exposes the HF tokenizer via `.tokenizer`
    # or `._tokenizer`. Fall back to the wrapper itself.
    for attr in ("tokenizer", "_tokenizer"):
        tok = getattr(tokenizer_wrapper, attr, None)
        if tok is not None:
            return tok
    return tokenizer_wrapper


def try_resolve_token_id(tokenizer_wrapper: Any, token: str) -> int | None:
    """Best-effort mapping from token string to token id.

    Returns None when the token cannot be resolved.
    """

    tok = _get_underlying_tokenizer(tokenizer_wrapper)

    # Prefer vocab lookups where available.
    try:
        vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else None
        if isinstance(vocab, dict):
            tid = vocab.get(token)
            if isinstance(tid, int):
                return tid
    except Exception:
        pass

    # Fall back to tokenizer method.
    try:
        if hasattr(tok, "convert_tokens_to_ids"):
            tid = tok.convert_tokens_to_ids(token)
            if isinstance(tid, int):
                # Some tokenizers return unk_token_id for unknown tokens.
                unk_id = getattr(tok, "unk_token_id", None)
                if isinstance(unk_id, int) and tid == unk_id:
                    return None
                return tid
    except Exception:
        pass

    return None


def extend_eos_token_ids(tokenizer_wrapper: Any, extra_ids: Iterable[int]) -> None:
    extras = [i for i in extra_ids if isinstance(i, int)]
    if not extras:
        return

    current = getattr(tokenizer_wrapper, "eos_token_ids", None)
    current_list: list[int] = []
    if isinstance(current, (list, tuple)):
        current_list = [i for i in current if isinstance(i, int)]

    seen: set[int] = set()
    merged: list[int] = []
    for i in current_list + extras:
        if i in seen:
            continue
        seen.add(i)
        merged.append(i)

    setattr(tokenizer_wrapper, "eos_token_ids", merged)
    logger.debug(f"Patched eos_token_ids={merged}")


def extend_eos_token_ids_by_token_strings(
    tokenizer_wrapper: Any, token_strings: Iterable[str]
) -> None:
    token_ids: list[int] = []
    for token in token_strings:
        tid = try_resolve_token_id(tokenizer_wrapper, token)
        if tid is not None:
            token_ids.append(tid)

    extend_eos_token_ids(tokenizer_wrapper, token_ids)

