from collections.abc import Sequence


class IncrementalUtf8Decoder:
    """Stream-safe incremental UTF-8 decoder preventing character splitting and corruption.

    Strictly satisfies Chinese & Emoji multi-byte guidelines by accumulating split bytes
    across token boundaries until a valid unicode sequence can be decoded cleanly.
    """

    def __init__(self) -> None:
        self._pending_bytes: bytearray = bytearray()

    def decode_incremental(self, new_bytes: bytes) -> str:
        """Appends new bytes and returns newly decodable unicode text."""
        self._pending_bytes.extend(new_bytes)
        if not self._pending_bytes:
            return ""

        # Attempt to decode as much as possible from the front
        valid_length = 0
        for i in range(len(self._pending_bytes), 0, -1):
            try:
                bytes(self._pending_bytes[:i]).decode("utf-8")
                valid_length = i
                break
            except UnicodeDecodeError:
                continue

        if valid_length > 0:
            result = bytes(self._pending_bytes[:valid_length]).decode("utf-8")
            self._pending_bytes = self._pending_bytes[valid_length:]
            return result
        return ""

    def flush(self) -> str:
        """Forces flush of any remaining bytes, replacing incomplete sequences."""
        if not self._pending_bytes:
            return ""
        result = bytes(self._pending_bytes).decode("utf-8", errors="replace")
        self._pending_bytes.clear()
        return result


class EdgeStreamingGateway:
    """Peripheral edge gateway for local detokenization and SSE streaming."""

    def __init__(self, vocab_id_to_bytes: dict[int, bytes]) -> None:
        self._vocab_id_to_bytes = vocab_id_to_bytes
        self._decoder = IncrementalUtf8Decoder()
        self._generated_token_ids: list[int] = []

    @property
    def token_history(self) -> Sequence[int]:
        return tuple(self._generated_token_ids)

    def process_token(self, token_id: int) -> str:
        """Processes an incoming token from the compute node, returning new delta text."""
        self._generated_token_ids.append(token_id)
        token_bytes = self._vocab_id_to_bytes.get(token_id, b"")
        return self._decoder.decode_incremental(token_bytes)

    def finalize(self) -> str:
        """Finalizes the stream, returning any remaining buffered text."""
        return self._decoder.flush()

    def format_sse_chunk(self, delta_text: str, is_done: bool = False) -> str:
        """Formats string as standard Server-Sent Events (SSE) data frame."""
        import json

        payload = {"delta": delta_text, "done": is_done}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
