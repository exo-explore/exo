"""What a model wrote (tool-call text, arguments, text it failed to parse) stays out of the logs
and out of error messages, whether the call parsed or not. `EXO_LOG_REQUEST_CONTENT=true` brings
it back."""

import sys
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Any

import pytest
import zstandard
from loguru import logger

from exo.api.types import ToolCallItem
from exo.shared import logging as exo_logging
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.models.deepseek_v3.parser import parse_dsml_stream
from exo.worker.engines.mlx.models.deepseek_v4 import encoding as dsv4
from exo.worker.engines.mlx.models.deepseek_v41 import encoding as dsv41
from exo.worker.engines.mlx.models.llama.parser import parse_llama3
from exo.worker.runner.llm_inference.model_output_parsers import parse_tool_calls
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser

SENTINEL = "ZQSENTINEL5e07a3b9"


@pytest.fixture
def logs(tmp_path: Path) -> Iterator[Path]:
    log_file = tmp_path / "exo.log"
    exo_logging.logger_setup(log_file, verbosity=1)  # -v: debug lines too
    yield log_file
    logger.remove()
    logger.add(sys.stderr)


def _everything_logged(log_file: Path) -> str:
    exo_logging.logger_cleanup()
    text = log_file.read_text()
    for archive in log_file.parent.glob("exo.*.log.zst"):
        text += zstandard.ZstdDecompressor().decompress(archive.read_bytes()).decode()
    return text


def _stream(*texts: str) -> Generator[GenerationResponse]:
    for i, text in enumerate(texts):
        yield GenerationResponse(
            text=text,
            token=i,
            finish_reason="stop" if i == len(texts) - 1 else None,
            usage=None,
        )


def _tool_parser(
    result: Callable[[str], list[ToolCallItem] | None],
):
    def inner(text: str, _tools: Any = None) -> list[ToolCallItem] | None:  # pyright: ignore[reportAny]
        return result(text)

    return make_mlx_parser("<tc>", "</tc>", inner)  # pyright: ignore[reportArgumentType]


def _run_every_parser_on_text_carrying_the_sentinel() -> None:
    # A tool call that parses, and one that does not
    parsed = _tool_parser(
        lambda text: [ToolCallItem(name="lookup", arguments=f'{{"ref": "{text}"}}')]
    )
    list(parse_tool_calls(_stream("<tc>", f"fn {SENTINEL}", "</tc>"), parsed, None))
    unparsed = _tool_parser(lambda _text: None)
    list(parse_tool_calls(_stream("<tc>", f"fn {SENTINEL}", "</tc>"), unparsed, None))
    # DeepSeek's and Llama 3's parsers, on a call they cannot read
    list(
        parse_dsml_stream(
            _stream("<S>", f"body {SENTINEL}", "</S>"), "<S>", "</S>", lambda _t: None
        )
    )
    list(
        parse_llama3(
            _stream(f"<|python_tag|>lookup(ref={SENTINEL}"), frozenset({"lookup"})
        )
    )


def test_no_parser_logs_what_the_model_wrote(logs: Path) -> None:
    _run_every_parser_on_text_carrying_the_sentinel()
    text = _everything_logged(logs)
    assert "parsed a tool call of" in text
    assert "DSML tool call parsing failed" in text
    assert "Llama 3 tool call parsing failed" in text
    assert SENTINEL not in text


def test_the_flag_brings_the_model_s_text_back(
    logs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXO_LOG_REQUEST_CONTENT", "true")
    _run_every_parser_on_text_carrying_the_sentinel()
    assert SENTINEL in _everything_logged(logs)


@pytest.mark.parametrize("encoding", [dsv4, dsv41], ids=["dsv4", "dsv41"])
def test_a_malformed_deepseek_tool_call_does_not_put_its_text_in_the_error(
    encoding: Any,  # pyright: ignore[reportAny]
) -> None:
    d: str = encoding.dsml_token  # pyright: ignore[reportAny]
    for text in (
        f">\n<{d}invoke name={SENTINEL}>\n",
        f'>\n<{d}invoke name="t">\n<{d}parameter name={SENTINEL}</{d}parameter>',
        f'>\n<{d}invoke name="t">\n<{d}parameter name="a" string="true">1</{d}parameter>{SENTINEL}\n</{d}invoke>',
    ):
        with pytest.raises(ValueError) as error:
            encoding.parse_tool_calls(0, text)  # pyright: ignore[reportAny]
        assert "format error" in str(error.value)
        assert SENTINEL not in str(error.value)
