"""Encode and decode prompts with the tokenizer stored in a GGUF header.

Byte-level byte-pair encoding covers GPT-2, Qwen2, and Llama 3. Llama
sentencepiece vocabs use the U+2581 space marker. Chat text is wrapped with
the special tokens present in the vocabulary. Jinja templates are not run.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Literal, final

from exo.backends.tinygrad_checkpoint import GgufCheckpoint
from exo.backends.tinygrad_weights import TinygradWeightError

type TokenizerModelName = Literal["llama", "gpt2"]
type PretokenizerName = Literal["gpt-2", "llama-bpe", "qwen2", "sentencepiece"]

_TOKEN_TYPE_CONTROL = 3
_TOKEN_TYPE_USER_DEFINED = 4
_TOKEN_TYPE_BYTE = 6
_SPACE_MARKER = "\u2581"
_CHAT_SPECIAL_TOKENS = frozenset(
    {
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|im_start|>",
        "<|im_end|>",
    }
)
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


@final
@dataclass(frozen=True)
class GgufTokenizer:
    """Vocabulary and merge ranks read from one GGUF file."""

    model: TokenizerModelName
    pretokenizer: PretokenizerName
    tokens: tuple[str, ...]
    merges: tuple[str, ...]
    token_types: tuple[int, ...]
    bos_token_id: int | None
    eos_token_id: int | None
    add_bos_token: bool
    ids_by_token: Mapping[str, int]
    merge_ranks: Mapping[tuple[str, str], int]

    def contains(self, token: str) -> bool:
        return token in self.ids_by_token


def gguf_tokenizer_from_checkpoint(checkpoint: GgufCheckpoint) -> GgufTokenizer | None:
    """Build a tokenizer from header arrays, or ``None`` when they are absent.

    Raises:
        TinygradWeightError: The runner entrypoint handles this when a merge
            or token-type array cannot be paired with the token list.
    """
    if not checkpoint.tokenizer_tokens:
        return None
    model_value = checkpoint.scalar("tokenizer.ggml.model")
    pre_value = checkpoint.scalar("tokenizer.ggml.pre")
    model = _tokenizer_model(model_value)
    pretokenizer = _pretokenizer_name(model, pre_value)
    ids_by_token: dict[str, int] = {}
    for index, token in enumerate(checkpoint.tokenizer_tokens):
        if token not in ids_by_token:
            ids_by_token[token] = index
    return GgufTokenizer(
        model=model,
        pretokenizer=pretokenizer,
        tokens=checkpoint.tokenizer_tokens,
        merges=checkpoint.tokenizer_merges,
        token_types=checkpoint.tokenizer_token_types,
        bos_token_id=_optional_id(checkpoint.scalar("tokenizer.ggml.bos_token_id")),
        eos_token_id=_optional_id(checkpoint.scalar("tokenizer.ggml.eos_token_id")),
        add_bos_token=_optional_bool(checkpoint.scalar("tokenizer.ggml.add_bos_token")),
        ids_by_token=MappingProxyType(ids_by_token),
        merge_ranks=MappingProxyType(_merge_ranks(checkpoint.tokenizer_merges)),
    )


def gpt2_byte_symbols_for_text(text: str) -> tuple[str, ...]:
    """Return the GPT-2 unicode symbols for each distinct byte in ``text``."""
    encoder = _byte_encoder()
    symbols: list[str] = []
    seen: set[str] = set()
    for byte in text.encode("utf-8"):
        symbol = encoder[byte]
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return tuple(symbols)


def format_chat_messages(
    tokenizer: GgufTokenizer,
    messages: Sequence[tuple[str, str]],
    instructions: str | None,
) -> str:
    """Wrap chat turns with Qwen or Llama 3 markers when those tokens exist."""
    rows: list[tuple[str, str]] = []
    if instructions:
        rows.append(("system", instructions))
    rows.extend(messages)
    if tokenizer.contains("<|im_start|>") and tokenizer.contains("<|im_end|>"):
        parts = [f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in rows]
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
    if (
        tokenizer.contains("<|start_header_id|>")
        and tokenizer.contains("<|end_header_id|>")
        and tokenizer.contains("<|eot_id|>")
    ):
        parts: list[str] = []
        if tokenizer.contains("<|begin_of_text|>"):
            parts.append("<|begin_of_text|>")
        for role, content in rows:
            parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)
    return "\n".join(content for _role, content in rows)


def encode_chat(
    tokenizer: GgufTokenizer,
    messages: Sequence[tuple[str, str]],
    instructions: str | None,
) -> tuple[int, ...]:
    """Encode chat turns, prepending the BOS id only for the plain wrapper.

    Raises:
        TinygradWeightError: The runner entrypoint handles a piece that is
            not in the vocabulary.
    """
    text = format_chat_messages(tokenizer, messages, instructions)
    token_ids = list(encode(tokenizer, text))
    bos_token_id = tokenizer.bos_token_id
    if (
        _chat_wrapper(tokenizer) == "plain"
        and tokenizer.add_bos_token
        and bos_token_id is not None
        and (not token_ids or token_ids[0] != bos_token_id)
    ):
        token_ids.insert(0, bos_token_id)
    return tuple(token_ids)


def encode(tokenizer: GgufTokenizer, text: str) -> tuple[int, ...]:
    """Encode ``text``, matching special tokens before byte-pair encoding.

    Raises:
        TinygradWeightError: The runner entrypoint handles a piece that is
            not in the vocabulary.
    """
    encoded: list[int] = []
    for piece in _split_special_tokens(tokenizer, text):
        if isinstance(piece, int):
            encoded.append(piece)
            continue
        if tokenizer.pretokenizer == "sentencepiece":
            encoded.extend(_encode_sentencepiece(tokenizer, piece))
            continue
        for pretoken in _pretokenize(piece, tokenizer.pretokenizer):
            symbols = _byte_pair_encode(_byte_encode(pretoken), tokenizer.merge_ranks)
            encoded.extend(_ids_for_symbols(tokenizer, symbols))
    return tuple(encoded)


def decode_token(tokenizer: GgufTokenizer, token_id: int) -> str:
    """Decode one token. The end-of-sequence id decodes to an empty string."""
    return decode_tokens(tokenizer, (token_id,))


def decode_tokens(tokenizer: GgufTokenizer, token_ids: Sequence[int]) -> str:
    """Decode a token-id sequence, flushing bytes before control tokens."""
    pieces: list[str] = []
    pending_bytes = bytearray()

    def flush_bytes() -> None:
        if not pending_bytes:
            return
        pieces.append(pending_bytes.decode("utf-8", errors="replace"))
        pending_bytes.clear()

    byte_decoder = _byte_decoder()
    for token_id in token_ids:
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            continue
        if token_id < 0 or token_id >= len(tokenizer.tokens):
            continue
        token = tokenizer.tokens[token_id]
        token_type = _token_type(tokenizer, token_id)
        if token_type == _TOKEN_TYPE_BYTE or _is_explicit_byte_token(token):
            flush_bytes()
            pieces.append(_decode_explicit_byte_token(token))
            continue
        if _is_control_token(token, token_type):
            flush_bytes()
            pieces.append(token)
            continue
        if tokenizer.pretokenizer == "sentencepiece":
            flush_bytes()
            pieces.append(token.replace(_SPACE_MARKER, " "))
            continue
        for character in token:
            byte = byte_decoder.get(character)
            if byte is None:
                flush_bytes()
                pieces.append(character)
                continue
            pending_bytes.append(byte)
    flush_bytes()
    return "".join(pieces)


def incremental_token_text(
    tokenizer: GgufTokenizer,
    previous_token_ids: Sequence[int],
    token_id: int,
) -> str:
    """Return the new text contributed by ``token_id`` after ``previous_token_ids``."""
    if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
        return ""
    previous = decode_tokens(tokenizer, previous_token_ids)
    current = decode_tokens(tokenizer, (*previous_token_ids, token_id))
    if current.startswith(previous):
        return current[len(previous) :]
    return current


def _tokenizer_model(value: object) -> TokenizerModelName:
    if value == "llama":
        return "llama"
    return "gpt2"


def _pretokenizer_name(model: TokenizerModelName, value: object) -> PretokenizerName:
    if value == "qwen2":
        return "qwen2"
    if value in ("llama-bpe", "llama3"):
        return "llama-bpe"
    if value in ("gpt-2", "gpt2"):
        return "gpt-2"
    if model == "llama":
        return "sentencepiece"
    return "gpt-2"


def _optional_id(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_bool(value: object) -> bool:
    return value is True


def _merge_ranks(merges: tuple[str, ...]) -> dict[tuple[str, str], int]:
    ranks: dict[tuple[str, str], int] = {}
    for rank, merge in enumerate(merges):
        left, separator, right = merge.partition(" ")
        if separator != " " or left == "" or right == "":
            continue
        ranks.setdefault((left, right), rank)
    return ranks


def _chat_wrapper(tokenizer: GgufTokenizer) -> Literal["qwen", "llama3", "plain"]:
    if tokenizer.contains("<|im_start|>") and tokenizer.contains("<|im_end|>"):
        return "qwen"
    if (
        tokenizer.contains("<|start_header_id|>")
        and tokenizer.contains("<|end_header_id|>")
        and tokenizer.contains("<|eot_id|>")
    ):
        return "llama3"
    return "plain"


def _token_type(tokenizer: GgufTokenizer, token_id: int) -> int:
    if token_id < 0 or token_id >= len(tokenizer.token_types):
        return 1
    return tokenizer.token_types[token_id]


def _is_control_token(token: str, token_type: int) -> bool:
    return (
        token_type in (_TOKEN_TYPE_CONTROL, _TOKEN_TYPE_USER_DEFINED)
        or token in _CHAT_SPECIAL_TOKENS
    )


def _is_explicit_byte_token(token: str) -> bool:
    if len(token) != 6 or not token.startswith("<0x") or not token.endswith(">"):
        return False
    hex_digits = token[3:5]
    return all(character in "0123456789abcdefABCDEF" for character in hex_digits)


def _decode_explicit_byte_token(token: str) -> str:
    byte = int(token[3:5], 16)
    return bytes((byte,)).decode("latin-1")


def _specials_longest_first(tokenizer: GgufTokenizer) -> tuple[tuple[str, int], ...]:
    specials: list[tuple[str, int]] = []
    for index, token in enumerate(tokenizer.tokens):
        if token == "":
            continue
        if _is_control_token(token, _token_type(tokenizer, index)):
            specials.append((token, tokenizer.ids_by_token.get(token, index)))
    specials.sort(key=_special_length, reverse=True)
    return tuple(specials)


def _special_length(item: tuple[str, int]) -> int:
    return len(item[0])


def _split_special_tokens(tokenizer: GgufTokenizer, text: str) -> tuple[str | int, ...]:
    specials = _specials_longest_first(tokenizer)
    pieces: list[str | int] = []
    buffer: list[str] = []
    index = 0
    while index < len(text):
        matched_id: int | None = None
        matched_length = 0
        for token, token_id in specials:
            if text.startswith(token, index):
                matched_id = token_id
                matched_length = len(token)
                break
        if matched_id is None:
            buffer.append(text[index])
            index += 1
            continue
        if buffer:
            pieces.append("".join(buffer))
            buffer = []
        pieces.append(matched_id)
        index += matched_length
    if buffer:
        pieces.append("".join(buffer))
    return tuple(pieces)


def _encode_sentencepiece(tokenizer: GgufTokenizer, text: str) -> list[int]:
    if text == "":
        return []
    prepared = text.replace(" ", _SPACE_MARKER)
    if not prepared.startswith(_SPACE_MARKER):
        prepared = _SPACE_MARKER + prepared
    words: list[str] = []
    if prepared == _SPACE_MARKER:
        words.append(_SPACE_MARKER)
    else:
        for chunk_index, chunk in enumerate(prepared.split(_SPACE_MARKER)):
            if chunk_index == 0:
                continue
            words.append(_SPACE_MARKER + chunk)
    encoded: list[int] = []
    for word in words:
        symbols = _byte_pair_encode(word, tokenizer.merge_ranks)
        encoded.extend(_ids_for_symbols(tokenizer, symbols))
    return encoded


def _ids_for_symbols(tokenizer: GgufTokenizer, symbols: tuple[str, ...]) -> list[int]:
    encoded: list[int] = []
    for symbol in symbols:
        token_id = tokenizer.ids_by_token.get(symbol)
        if token_id is None:
            raise TinygradWeightError(f"GGUF tokenizer has no token for {symbol!r}")
        encoded.append(token_id)
    return encoded


def _byte_encode(text: str) -> str:
    encoder = _byte_encoder()
    return "".join(encoder[byte] for byte in text.encode("utf-8"))


def _byte_pair_encode(
    piece: str, merge_ranks: Mapping[tuple[str, str], int]
) -> tuple[str, ...]:
    if piece == "":
        return ()
    symbols = list(piece)
    while len(symbols) >= 2:
        best_rank: int | None = None
        best_index = 0
        for index in range(len(symbols) - 1):
            rank = merge_ranks.get((symbols[index], symbols[index + 1]))
            if rank is None:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_index = index
        if best_rank is None:
            break
        merged = symbols[best_index] + symbols[best_index + 1]
        symbols = [*symbols[:best_index], merged, *symbols[best_index + 2 :]]
    return tuple(symbols)


def _pretokenize(text: str, pretokenizer: PretokenizerName) -> tuple[str, ...]:
    if text == "":
        return ()
    pieces: list[str] = []
    index = 0
    while index < len(text):
        if pretokenizer == "qwen2":
            length = _match_qwen2(text, index)
        elif pretokenizer == "llama-bpe":
            length = _match_llama_bpe(text, index)
        else:
            length = _match_gpt2(text, index)
        if length is None or length <= 0:
            pieces.append(text[index])
            index += 1
            continue
        pieces.append(text[index : index + length])
        index += length
    return tuple(pieces)


type _Matcher = Callable[[str, int], int | None]
type _CharacterPredicate = Callable[[str], bool]


def _match_gpt2(text: str, index: int) -> int | None:
    return _first_match(
        text,
        index,
        (
            _match_contraction_sensitive,
            _match_optional_space_letters,
            _match_optional_space_numbers,
            _match_optional_space_other,
            _match_trailing_whitespace,
            _match_whitespace,
        ),
    )


def _match_llama_bpe(text: str, index: int) -> int | None:
    return _first_match(
        text,
        index,
        (
            _match_contraction_insensitive,
            _match_prefixed_letters,
            _match_llama_numbers,
            _match_punctuation_run,
            _match_newline_run,
            _match_trailing_whitespace,
            _match_whitespace,
        ),
    )


def _match_qwen2(text: str, index: int) -> int | None:
    return _first_match(
        text,
        index,
        (
            _match_contraction_insensitive,
            _match_prefixed_letters,
            _match_qwen_number,
            _match_punctuation_run,
            _match_newline_run,
            _match_trailing_whitespace,
            _match_whitespace,
        ),
    )


def _first_match(text: str, index: int, matchers: Sequence[_Matcher]) -> int | None:
    for matcher in matchers:
        length = matcher(text, index)
        if length is not None and length > 0:
            return length
    return None


def _match_contraction_sensitive(text: str, index: int) -> int | None:
    return _match_contraction(text, index, ignore_case=False)


def _match_contraction_insensitive(text: str, index: int) -> int | None:
    return _match_contraction(text, index, ignore_case=True)


def _match_llama_numbers(text: str, index: int) -> int | None:
    return _match_numbers(text, index, limit=3)


def _match_qwen_number(text: str, index: int) -> int | None:
    return _match_numbers(text, index, limit=1)


def _match_contraction(text: str, index: int, *, ignore_case: bool) -> int | None:
    remaining = text[index:]
    for contraction in _CONTRACTIONS:
        prefix = remaining[: len(contraction)]
        if ignore_case:
            prefix = prefix.lower()
        if prefix == contraction:
            return len(contraction)
    return None


def _match_optional_space_letters(text: str, index: int) -> int | None:
    return _match_optional_space(text, index, _is_letter)


def _match_optional_space_numbers(text: str, index: int) -> int | None:
    return _match_optional_space(text, index, _is_number)


def _match_optional_space_other(text: str, index: int) -> int | None:
    return _match_optional_space(text, index, _is_other)


def _match_optional_space(
    text: str,
    index: int,
    predicate: _CharacterPredicate,
) -> int | None:
    cursor = index
    if cursor < len(text) and text[cursor] == " ":
        cursor += 1
    start = cursor
    while cursor < len(text) and predicate(text[cursor]):
        cursor += 1
    if cursor == start:
        return None
    return cursor - index


def _match_prefixed_letters(text: str, index: int) -> int | None:
    cursor = index
    if cursor < len(text) and _is_letter_prefix(text[cursor]):
        cursor += 1
    start = cursor
    while cursor < len(text) and _is_letter(text[cursor]):
        cursor += 1
    if cursor == start:
        return None
    return cursor - index


def _match_numbers(text: str, index: int, *, limit: int) -> int | None:
    cursor = index
    while cursor < len(text) and cursor - index < limit and _is_number(text[cursor]):
        cursor += 1
    if cursor == index:
        return None
    return cursor - index


def _match_punctuation_run(text: str, index: int) -> int | None:
    cursor = index
    if cursor < len(text) and text[cursor] == " ":
        cursor += 1
    start = cursor
    while cursor < len(text) and _is_other(text[cursor]):
        cursor += 1
    if cursor == start:
        return None
    while cursor < len(text) and text[cursor] in "\r\n":
        cursor += 1
    return cursor - index


def _match_newline_run(text: str, index: int) -> int | None:
    end = index
    while end < len(text) and _is_whitespace(text[end]):
        end += 1
    while end > index and text[end - 1] not in "\r\n":
        end -= 1
    if end == index:
        return None
    return end - index


def _match_trailing_whitespace(text: str, index: int) -> int | None:
    end = index
    while end < len(text) and _is_whitespace(text[end]):
        end += 1
    while end > index:
        if end == len(text) or _is_whitespace(text[end]):
            return end - index
        end -= 1
    return None


def _match_whitespace(text: str, index: int) -> int | None:
    end = index
    while end < len(text) and _is_whitespace(text[end]):
        end += 1
    if end == index:
        return None
    return end - index


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _is_whitespace(character: str) -> bool:
    return character.isspace()


def _is_other(character: str) -> bool:
    return not (
        _is_whitespace(character) or _is_letter(character) or _is_number(character)
    )


def _is_letter_prefix(character: str) -> bool:
    return (
        character not in "\r\n"
        and not _is_letter(character)
        and not _is_number(character)
    )


@cache
def _byte_encoder() -> Mapping[int, str]:
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    bytes_to_unicode = dict(zip(printable, printable, strict=True))
    next_code = 0
    for byte in range(256):
        if byte in bytes_to_unicode:
            continue
        bytes_to_unicode[byte] = 256 + next_code
        next_code += 1
    return MappingProxyType(
        {byte: chr(code_point) for byte, code_point in bytes_to_unicode.items()}
    )


@cache
def _byte_decoder() -> Mapping[str, int]:
    return MappingProxyType({symbol: byte for byte, symbol in _byte_encoder().items()})
