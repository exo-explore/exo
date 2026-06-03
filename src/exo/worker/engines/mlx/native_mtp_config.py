import os

EXO_NATIVE_MTP_ENABLED_ENV = "EXO_NATIVE_MTP_ENABLED"


def native_mtp_enabled_from_env() -> bool:
    """Return whether native-MTP dispatch is enabled for supported cards."""
    raw = os.environ.get(EXO_NATIVE_MTP_ENABLED_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def clamp_native_mtp_num_draft_tokens(
    num_draft_tokens: int, *, max_k: int
) -> tuple[int, bool]:
    """Clamp native-MTP K to the card-declared runtime budget."""
    bounded_max = max(1, max_k)
    if num_draft_tokens < 1:
        return 1, True
    if num_draft_tokens > bounded_max:
        return bounded_max, True
    return num_draft_tokens, False


def resolve_native_mtp_num_draft_tokens(
    *,
    request_num_draft_tokens: int | None,
    configured_num_draft_tokens: int | None,
    card_default_k: int,
    card_max_k: int,
) -> tuple[int, bool]:
    """Resolve native-MTP K with request > startup config > card default."""
    chosen = (
        request_num_draft_tokens
        if request_num_draft_tokens is not None
        else configured_num_draft_tokens
    )
    if chosen is None:
        chosen = card_default_k
    return clamp_native_mtp_num_draft_tokens(chosen, max_k=card_max_k)
