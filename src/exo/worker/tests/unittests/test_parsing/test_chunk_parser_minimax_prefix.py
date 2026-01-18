# pyright: reportAny=false

from exo.worker.parsing.stream import ChunkParser, ChunkParserConfig


def test_chunkparser_minimax_prefix_parses_reasoning_from_first_chunk() -> None:
    # MiniMaxM2ReasoningParser needs a redacted reasoning prefix on the first chunk.
    # ChunkParser prepends the opening token to the *first* delta so that
    # downstream logic can treat it like a normal Hermes-style <think> block.
    parser = ChunkParser(
        ChunkParserConfig(reasoning_parser_name="minimax_m2", tool_parser_name=None)
    )

    out1 = parser.feed("R1")
    assert out1 == [{"reasoning_content": "R1"}]

    # Close tag arrives later; we should resume emitting normal content after it.
    out2 = parser.feed("</think>OK")
    assert out2 == ["OK"]
