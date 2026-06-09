"""Streaming-safe stop-sequence scanning.

Stop sequences can span multiple decoded tokens, so a naive substring check on
each token's text leaks the leading bytes of a sequence before the whole
sequence has been generated (e.g. ``"END"`` arriving as ``"E"`` then ``"ND"``
would stream the ``"E"`` before the match is recognised). The scanner below
holds back any trailing text that might still grow into a stop sequence until it
is known to be safe to emit.
"""


def scan_stop_sequences(
    buffer: str,
    stop_sequences: list[str],
) -> tuple[str, str | None, str]:
    """Split a running text buffer around stop sequences.

    The caller appends each newly decoded chunk of text to the ``pending`` value
    returned by the previous call and passes the result back in as ``buffer``.

    Returns ``(emit, matched, pending)``:

    - ``emit``: text that is safe to stream now. It contains everything before
      the first fully matched stop sequence and excludes any trailing partial
      match that could still grow into a stop sequence on the next token.
    - ``matched``: the stop sequence that was fully matched, or ``None``.
    - ``pending``: a trailing partial match to carry into the next call. Empty
      when a sequence matched or when nothing needs to be held back. Bounded by
      the length of the longest stop sequence minus one.
    """
    # Ignore empty stop sequences — they would match everywhere and stop
    # generation immediately with no output.
    active = [stop_sequence for stop_sequence in stop_sequences if stop_sequence]
    if not active:
        return buffer, None, ""

    # Earliest fully matched stop sequence wins: generation stops there.
    earliest_index = len(buffer)
    matched: str | None = None
    for stop_sequence in active:
        index = buffer.find(stop_sequence)
        if index != -1 and index < earliest_index:
            earliest_index = index
            matched = stop_sequence
    if matched is not None:
        return buffer[:earliest_index], matched, ""

    # No full match: hold back the longest suffix of the buffer that is a proper
    # prefix of any stop sequence, since the next token might complete it.
    hold = 0
    for stop_sequence in active:
        max_partial = min(len(stop_sequence) - 1, len(buffer))
        for length in range(max_partial, hold, -1):
            if stop_sequence.startswith(buffer[-length:]):
                hold = length
                break
    if hold == 0:
        return buffer, None, ""
    split = len(buffer) - hold
    return buffer[:split], None, buffer[split:]
