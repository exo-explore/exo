# pyright: reportAny=false

from exo.worker.parsing.stream import ChunkParser, ChunkParserConfig


def test_chunkparser_hermes_reasoning_then_content_across_chunks() -> None:
    parser = ChunkParser(
        ChunkParserConfig(reasoning_parser_name="hermes", tool_parser_name=None)
    )

    out1 = parser.feed("<think>hello ")
    assert out1 == [{"reasoning_content": "hello "}]

    out2 = parser.feed("world</think>FINAL")
    assert out2 == [{"reasoning_content": "world"}, "FINAL"]
