from collections.abc import Mapping
from typing import Literal

import numpy as np
from numpy.typing import NDArray

JsonParserState = Literal[
    "INITIAL",
    "IN_OBJECT_EXPECT_KEY_OR_CLOSE",
    "IN_OBJECT_EXPECT_COLON",
    "IN_OBJECT_EXPECT_VALUE",
    "IN_OBJECT_EXPECT_COMMA_OR_CLOSE",
    "IN_STRING",
    "COMPLETED",
]

NEGATIVE_INFINITY_MASK = -float("inf")


class SimpleJsonGrammarMasker:
    """Lightweight pushdown-automaton grammar masker for JSON structured generation.

    Executed on peripheral edge nodes to offload grammar verification overhead from GPUs.
    """

    def __init__(self, vocab_id_to_text: Mapping[int, str]) -> None:
        self._vocab_id_to_text = vocab_id_to_text
        self._state: JsonParserState = "INITIAL"
        self._brace_depth: int = 0
        self._in_quote: bool = False

    @property
    def current_state(self) -> JsonParserState:
        return self._state

    @property
    def brace_depth(self) -> int:
        return self._brace_depth

    def update_state(self, token_text: str) -> None:
        """Transitions internal state given newly sampled token string."""
        for char in token_text:
            if char == '"' and not self._in_quote:
                self._in_quote = True
                continue
            elif char == '"' and self._in_quote:
                self._in_quote = False
                continue

            if self._in_quote:
                continue

            # Outside strings
            if char == "{":
                self._brace_depth += 1
                self._state = "IN_OBJECT_EXPECT_KEY_OR_CLOSE"
            elif char == "}":
                self._brace_depth = max(0, self._brace_depth - 1)
                if self._brace_depth == 0:
                    self._state = "COMPLETED"
                else:
                    self._state = "IN_OBJECT_EXPECT_COMMA_OR_CLOSE"
            elif char == ":":
                self._state = "IN_OBJECT_EXPECT_VALUE"
            elif char == ",":
                self._state = "IN_OBJECT_EXPECT_KEY_OR_CLOSE"

    def compute_mask(self, logits: NDArray[np.float32]) -> NDArray[np.float32]:
        """Applies -inf mask to logits for tokens that would violate the JSON grammar."""
        masked_logits = logits.copy()

        # In COMPLETED state, only allow empty string or whitespace
        if self._state == "COMPLETED":
            for token_id, text in self._vocab_id_to_text.items():
                if 0 <= token_id < len(masked_logits) and text.strip() != "":
                    masked_logits[token_id] = NEGATIVE_INFINITY_MASK
            return masked_logits

        # In INITIAL state, only allow whitespace or '{'
        if self._state == "INITIAL":
            for token_id, text in self._vocab_id_to_text.items():
                if 0 <= token_id < len(masked_logits):
                    stripped = text.strip()
                    if stripped != "" and not stripped.startswith("{"):
                        masked_logits[token_id] = NEGATIVE_INFINITY_MASK
            return masked_logits

        # In IN_OBJECT_EXPECT_COLON state, forbid '{', '}', ','
        if self._state == "IN_OBJECT_EXPECT_COLON":
            for token_id, text in self._vocab_id_to_text.items():
                if 0 <= token_id < len(masked_logits):
                    stripped = text.strip()
                    if stripped in ("{", "}", ","):
                        masked_logits[token_id] = NEGATIVE_INFINITY_MASK
            return masked_logits

        return masked_logits
