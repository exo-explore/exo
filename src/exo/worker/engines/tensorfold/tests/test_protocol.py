import pytest

from exo.shared.types.chunks import TokenChunk, ToolCallChunk
from exo.worker.engines.tensorfold.protocol import StreamProtocol
from exo.worker.engines.tensorfold.sse import EventDecoder, Heartbeat, ProtocolError
from exo.worker.engines.tensorfold.tests.fixtures import MODEL, raw, semantic


def test_streams_semantics_before_completion_without_counting_chunks_as_tokens() -> (
    None
):
    stream = StreamProtocol(MODEL)
    assert stream.feed(semantic(0, None, {"role": "assistant"})) == []
    assert stream.feed(raw([1, 2, 3])) == []
    thought = stream.feed(semantic(3, 3, {"reasoning_content": "Checking"}))
    answer = stream.feed(semantic(3, 3, {"content": "Found it."}))
    assert isinstance(thought[0], TokenChunk) and thought[0].is_thinking
    assert isinstance(answer[0], TokenChunk) and answer[0].token_id == 3
    assert answer[0].text == "Found it."
    assert stream.feed(semantic(3, 3, finish="stop", tokens=[1, 2, 3])) == []
    with pytest.raises(ProtocolError, match="completion marker"):
        stream.finish(50)
    stream.feed("[DONE]")
    final = stream.finish(50)
    assert isinstance(final, TokenChunk)
    assert final.usage is not None and final.usage.completion_tokens == 3
    assert final.stats is not None and final.stats.peak_memory_usage.in_bytes == 50


def test_keeps_utf8_semantics_separate_from_logprob_byte_fragments() -> None:
    stream = StreamProtocol(MODEL)
    stream.feed(raw([77, 88]))
    output = stream.feed(
        semantic(
            2,
            88,
            {"content": "€"},
            logprob_positions=[0, 1],
            logprobs=[
                {"token": "�", "bytes": [226], "logprob": -0.1, "top_logprobs": []},
                {
                    "token": "��",
                    "bytes": [130, 172],
                    "logprob": -0.2,
                    "top_logprobs": [],
                },
            ],
        )
    )
    assert len(output) == 1 and isinstance(output[0], TokenChunk)
    assert output[0].text == "€"
    rows = output[0].logprobs_content
    assert rows is not None and rows[0].bytes == [226] and rows[1].bytes == [130, 172]
    assert rows[0].token == "�" and rows[1].logprob == -0.2


def test_empty_native_completion_preserves_zero_usage_without_inventing_a_token() -> (
    None
):
    stream = StreamProtocol(MODEL)
    assert stream.feed(semantic(0, None, {"role": "assistant"})) == []
    assert stream.feed(semantic(0, None, finish="stop", tokens=[])) == []
    stream.feed("[DONE]")
    final = stream.finish(0)
    assert isinstance(final, TokenChunk)
    assert final.text == "" and final.token_id is None
    assert final.finish_reason == "stop" and final.usage is not None
    assert final.usage.completion_tokens == 0 and final.usage.total_tokens == 4


def test_entries_of_one_token_shown_in_parts_share_its_position() -> None:
    stream = StreamProtocol(MODEL)
    stream.feed(raw([77, 88]))
    for text, position in (("a", 0), ("b", 0), ("c", 1)):
        output = stream.feed(
            semantic(
                2,
                88,
                {"content": text},
                logprob_positions=[position],
                logprobs=[{"token": text, "logprob": -1.0}],
            )
        )
        assert len(output) == 1 and isinstance(output[0], TokenChunk)
        assert output[0].logprobs_content is not None


@pytest.mark.parametrize(
    "positions, message",
    [
        ([1, 0], "token order"),
        ([0, 2], "token order"),
        ([-1, 0], "token order"),
        ([0], "token positions"),
        (None, "token positions"),
    ],
)
def test_rejects_logprob_rows_that_do_not_match_raw_positions(
    positions: list[int] | None, message: str
) -> None:
    stream = StreamProtocol(MODEL)
    stream.feed(raw([77, 88]))
    with pytest.raises(ProtocolError, match=message):
        stream.feed(
            semantic(
                2,
                88,
                {"content": "ab"},
                logprob_positions=positions,
                logprobs=[
                    {"token": "a", "logprob": -1.0},
                    {"token": "b", "logprob": -2.0},
                ],
            )
        )


def test_rejects_logprob_rows_that_go_back_across_chunks() -> None:
    stream = StreamProtocol(MODEL)
    stream.feed(raw([77, 88]))
    stream.feed(
        semantic(
            2,
            88,
            {"content": "b"},
            logprob_positions=[1],
            logprobs=[{"token": "b", "logprob": -2.0}],
        )
    )
    with pytest.raises(ProtocolError, match="token order"):
        stream.feed(
            semantic(
                2,
                88,
                {"content": "a"},
                logprob_positions=[0],
                logprobs=[{"token": "a", "logprob": -1.0}],
            )
        )


def test_tool_arguments_are_assembled_once_and_keep_native_call_identity() -> None:
    stream = StreamProtocol(MODEL)
    stream.feed(raw([5, 6]))
    stream.feed(
        semantic(
            2,
            6,
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-native",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":'},
                    }
                ]
            },
        )
    )
    stream.feed(
        semantic(
            2, 6, {"tool_calls": [{"index": 0, "function": {"arguments": '"hello"}'}}]}
        )
    )
    stream.feed(semantic(2, 6, finish="tool_calls", tokens=[5, 6]))
    stream.feed("[DONE]")
    final = stream.finish(0)
    assert isinstance(final, ToolCallChunk)
    assert final.tool_calls[0].id == "call-native"
    assert final.tool_calls[0].name == "lookup"
    assert final.tool_calls[0].arguments == '{"q":"hello"}'


@pytest.mark.parametrize(
    "terminal",
    [
        semantic(2, 11, finish="stop", tokens=[10, 12]),
        semantic(2, 11, finish="stop", tokens=[10, 11], completion_tokens=3),
    ],
)
def test_never_accepts_corrupt_final_receipts(terminal: str) -> None:
    stream = StreamProtocol(MODEL)
    stream.feed(raw([10, 11]))
    stream.feed(terminal)
    with pytest.raises(ProtocolError):
        stream.feed("[DONE]")
    assert not stream.done


def test_rejects_discontinuous_offsets_and_uncommitted_semantic_text() -> None:
    stream = StreamProtocol(MODEL)
    with pytest.raises(ProtocolError, match="discontinuous"):
        stream.feed(raw([1], offset=1))
    with pytest.raises(ProtocolError, match="before committing"):
        stream.feed(semantic(0, None, {"content": "invented"}))
    with pytest.raises(ProtocolError, match="tool call before a token"):
        stream.feed(semantic(0, None, {"tool_calls": [{"index": 0, "id": "invented"}]}))
    stream.feed(raw([5]))
    with pytest.raises(ProtocolError, match="anchor"):
        stream.feed(semantic(1, 6, {"content": "wrong token"}))


def test_fragmented_event_bytes_preserve_utf8_and_report_heartbeats() -> None:
    decoder = EventDecoder()
    wire = ': alive\r\n\r\ndata: {"content":"€"}\r\n\r\n'.encode()
    events: list[str | Heartbeat] = []
    for byte in wire:
        events.extend(decoder.feed(bytes([byte])))
    decoder.finish()
    assert isinstance(events[0], Heartbeat)
    assert events[1:] == ['{"content":"€"}']


def test_unterminated_or_oversized_events_fail_with_bounded_buffers() -> None:
    decoder = EventDecoder(limit=8)
    with pytest.raises(ProtocolError, match="size limit"):
        decoder.feed(b"data:" + b"x" * 20)
    decoder = EventDecoder()
    decoder.feed(b"data: partial")
    with pytest.raises(ProtocolError, match="inside an event"):
        decoder.finish()
