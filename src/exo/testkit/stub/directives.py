"""What the stub engine replies to a prompt, and the faults a prompt can ask it for.

The reply is a pure function of the prompt, so a test can check every token it receives:
token i of the reply to prompt P is `token_text(request_key(P), i)`. Give every request its
own prompt (the harness numbers them), or two concurrent replies can't be told apart.

A prompt can carry directives in one `[stub ...]` tag of space-separated key=value pairs:

    "story 17 [stub tokens=200 rate=20 fault=crash at=50 rank=1]"

- tokens: how many tokens the reply has (default 16), ending with finish_reason "stop". A
  request's max_tokens below that cuts the reply short with finish_reason "length".
- rate: tokens per second, per request (default 200).
- prefill: seconds before the first token (default 0).
- fault, at, rank: what the runner of rank `rank` (default 0) does just before it generates
  token `at` (default 0):
  - crash: the runner process dies at once (SIGKILL to itself, like a segfault or the OS
    killing it);
  - raise: the engine raises an exception (exo's runner reports it and exits);
  - hang: the runner stops forever (exo's supervisor notices after its stall timeout);
  - stall: the runner stops for `stall` seconds (default 10), then carries on.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Literal, final

FaultKind = Literal["none", "crash", "raise", "hang", "stall"]
FAULT_KINDS: tuple[FaultKind, ...] = ("none", "crash", "raise", "hang", "stall")

DEFAULT_TOKENS = 16
DEFAULT_RATE = 200.0
DEFAULT_STALL_SECONDS = 10.0

_TAG = re.compile(r"\[stub(?P<body>[^\]]*)\]")


@final
@dataclass(frozen=True)
class Directives:
    tokens: int = DEFAULT_TOKENS
    rate: float = DEFAULT_RATE
    prefill_seconds: float = 0.0
    fault: FaultKind = "none"
    fault_at: int = 0
    fault_rank: int = 0
    stall_seconds: float = DEFAULT_STALL_SECONDS

    def tag(self) -> str:
        """The `[stub ...]` tag that parses back to these directives."""
        parts = [f"tokens={self.tokens}", f"rate={self.rate:g}"]
        if self.prefill_seconds:
            parts.append(f"prefill={self.prefill_seconds:g}")
        if self.fault != "none":
            parts += [f"fault={self.fault}", f"at={self.fault_at}"]
            parts.append(f"rank={self.fault_rank}")
            if self.fault == "stall":
                parts.append(f"stall={self.stall_seconds:g}")
        return "[stub " + " ".join(parts) + "]"


def _fault_kind(value: str) -> FaultKind:
    for kind in FAULT_KINDS:
        if kind == value:
            return kind
    raise ValueError(f"unknown stub fault {value!r}; one of {', '.join(FAULT_KINDS)}")


def parse_directives(prompt: str) -> Directives:
    """The directives in the prompt's `[stub ...]` tag; the defaults if it has none.

    Raises ValueError for a tag the stub doesn't understand, so a typo fails the test that
    wrote it instead of being ignored."""
    match = _TAG.search(prompt)
    if match is None:
        return Directives()
    values: dict[str, str] = {}
    for part in match.group("body").split():
        key, separator, value = part.partition("=")
        if not separator:
            raise ValueError(f"stub directive {part!r} is not key=value")
        values[key] = value
    known = {"tokens", "rate", "prefill", "fault", "at", "rank", "stall"}
    if unknown := set(values) - known:
        raise ValueError(f"unknown stub directives {sorted(unknown)}")
    directives = Directives(
        tokens=int(values.get("tokens", DEFAULT_TOKENS)),
        rate=float(values.get("rate", DEFAULT_RATE)),
        prefill_seconds=float(values.get("prefill", 0.0)),
        fault=_fault_kind(values.get("fault", "none")),
        fault_at=int(values.get("at", 0)),
        fault_rank=int(values.get("rank", 0)),
        stall_seconds=float(values.get("stall", DEFAULT_STALL_SECONDS)),
    )
    if directives.tokens < 1 or directives.rate <= 0:
        raise ValueError(f"stub needs tokens >= 1 and rate > 0, got {directives}")
    return directives


def request_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def token_text(key: str, index: int) -> str:
    return f"{key}.{index} "


@final
@dataclass(frozen=True)
class ExpectedReply:
    tokens: tuple[str, ...]
    finish_reason: Literal["stop", "length"]

    @property
    def text(self) -> str:
        return "".join(self.tokens)


def reply_length(directives: Directives, max_tokens: int | None) -> int:
    if max_tokens is None:
        return directives.tokens
    return max(1, min(directives.tokens, max_tokens))


def expected_reply(prompt: str, max_tokens: int | None = None) -> ExpectedReply:
    """Exactly what the stub replies to this prompt when nothing goes wrong."""
    directives = parse_directives(prompt)
    key = request_key(prompt)
    length = reply_length(directives, max_tokens)
    return ExpectedReply(
        tokens=tuple(token_text(key, i) for i in range(length)),
        finish_reason="length" if length < directives.tokens else "stop",
    )
