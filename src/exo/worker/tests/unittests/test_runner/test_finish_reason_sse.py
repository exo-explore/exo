import json
from collections.abc import Generator
from typing import Any

from exo.api.types import CompletionTokensDetails, PromptTokensDetails, Usage
from exo.shared.types.worker.runner_response import (
    FinishReason,
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.vendor.dsml_encoding import (
    DSML_TOKEN,
    THINKING_END,
    THINKING_START,
    TOOL_CALLS_END,
    TOOL_CALLS_START,
)
from exo.worker.runner.llm_inference.model_output_parsers import (
    count_reasoning_tokens,
    parse_deepseek_v32,
    parse_thinking_models,
    parse_tool_calls,
)
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser


def _make_response(
    text: str, token: int, finish_reason: FinishReason | None = None
) -> GenerationResponse:
    return GenerationResponse(
        text=text, token=token, finish_reason=finish_reason, usage=None
    )


def _queue_source(
    tokens: list[GenerationResponse],
) -> Generator[GenerationResponse | None]:
    for token in tokens:
        yield token
        yield None
    while True:
        yield None


def _step_until_finish(
    parser_gen: Generator[GenerationResponse | ToolCallResponse | None],
    max_steps: int = 200,
) -> list[GenerationResponse | ToolCallResponse]:
    results: list[GenerationResponse | ToolCallResponse] = []
    for _ in range(max_steps):
        try:
            result = next(parser_gen)
        except StopIteration:
            break
        if result is None:
            continue
        results.append(result)
        if isinstance(result, GenerationResponse) and result.finish_reason is not None:
            return results
        if isinstance(result, ToolCallResponse):
            return results
    return results


def _got_finish(results: list[GenerationResponse | ToolCallResponse]) -> bool:
    for r in results:
        if isinstance(r, ToolCallResponse):
            return True
        if r.finish_reason is not None:
            return True
    return False


# ── parse_deepseek_v32 ──────────────────────────────────────────


class TestDeepSeekV32FinishReason:
    def test_finish_reason_with_buffered_dsml_prefix(self):
        tokens = [
            _make_response("Hello! The answer is x", 0),
            _make_response("<", 1),
            _make_response("", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)
        full_text = "".join(
            r.text for r in results if isinstance(r, GenerationResponse)
        )
        assert "Hello" in full_text
        assert "<" in full_text

    def test_finish_reason_completes_tool_call_block(self):
        tokens = [
            _make_response(TOOL_CALLS_START, 0),
            _make_response("\n", 1),
            _make_response(f'<{DSML_TOKEN}invoke name="get_weather">\n', 2),
            _make_response(
                f'<{DSML_TOKEN}parameter name="city" string="true">Tokyo</{DSML_TOKEN}parameter>\n',
                3,
            ),
            _make_response(f"</{DSML_TOKEN}invoke>\n", 4),
            _make_response(TOOL_CALLS_END, 5, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_finish_reason_mid_tool_call_before_close(self):
        tokens = [
            _make_response(TOOL_CALLS_START, 0),
            _make_response("\n", 1),
            _make_response(
                f'<{DSML_TOKEN}invoke name="get_weather">\n', 2, finish_reason="stop"
            ),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)

    def test_finish_reason_single_token_complete_dsml_block(self):
        dsml_block = (
            f"{TOOL_CALLS_START}\n"
            f'<{DSML_TOKEN}invoke name="get_weather">\n'
            f'<{DSML_TOKEN}parameter name="city" string="true">Tokyo</{DSML_TOKEN}parameter>\n'
            f"</{DSML_TOKEN}invoke>\n"
            f"{TOOL_CALLS_END}"
        )
        tokens = [_make_response(dsml_block, 0, finish_reason="stop")]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_finish_reason_during_thinking(self):
        tokens = [
            _make_response(THINKING_START, 0),
            _make_response("I need to think about this", 1),
            _make_response(" carefully before responding", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)

    def test_finish_reason_after_thinking_then_tool_call(self):
        tokens = [
            _make_response(THINKING_START, 0),
            _make_response("Let me check the weather.", 1),
            _make_response(THINKING_END, 2),
            _make_response("\n\n", 3),
            _make_response(TOOL_CALLS_START, 4),
            _make_response("\n", 5),
            _make_response(f'<{DSML_TOKEN}invoke name="get_weather">\n', 6),
            _make_response(
                f'<{DSML_TOKEN}parameter name="city" string="true">NYC</{DSML_TOKEN}parameter>\n',
                7,
            ),
            _make_response(f"</{DSML_TOKEN}invoke>\n", 8),
            _make_response(TOOL_CALLS_END, 9, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_finish_reason_normal_text_no_buffering(self):
        tokens = [
            _make_response("Hello", 0),
            _make_response(" world", 1),
            _make_response("!", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)
        full_text = "".join(
            r.text for r in results if isinstance(r, GenerationResponse)
        )
        assert full_text == "Hello world!"

    def test_finish_reason_multiple_buffered_prefix_tokens(self):
        tokens = [
            _make_response("text ", 0),
            _make_response("<", 1),
            _make_response("not a tag", 2),
            _make_response(" more<", 3),
            _make_response("", 4, finish_reason="stop"),
        ]
        results = _step_until_finish(parse_deepseek_v32(_queue_source(tokens)))
        assert _got_finish(results)


# ── parse_thinking_models ────────────────────────────────────────


class TestThinkingModelsFinishReason:
    def test_finish_reason_during_thinking(self):
        tokens = [
            _make_response("<think>", 0),
            _make_response("reasoning here", 1),
            _make_response("more reasoning", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _got_finish(results)
        last_gen = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(last_gen) == 1
        assert last_gen[0].is_thinking is False

    def test_finish_reason_after_thinking(self):
        tokens = [
            _make_response("<think>", 0),
            _make_response("hmm", 1),
            _make_response("</think>", 2),
            _make_response("The answer is 42.", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _got_finish(results)

    def test_finish_reason_starts_in_thinking(self):
        tokens = [
            _make_response("still thinking", 0),
            _make_response("</think>", 1),
            _make_response("done", 2, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=True,
            )
        )
        assert _got_finish(results)

    def test_reasoning_tokens_counted(self):
        """reasoning_tokens in Usage reflects the number of thinking tokens."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
        )
        tokens = [
            _make_response("<think>", 0),
            _make_response("let me", 1),
            _make_response(" think", 2),
            _make_response("</think>", 3),
            GenerationResponse(text="42", token=4, finish_reason="stop", usage=usage),
        ]
        results = _step_until_finish(
            count_reasoning_tokens(
                parse_thinking_models(
                    _queue_source(tokens),
                    think_start="<think>",
                    think_end="</think>",
                    starts_in_thinking=False,
                )
            )
        )
        final = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(final) == 1
        assert final[0].usage is not None
        assert final[0].usage.completion_tokens_details.reasoning_tokens == 2

    def test_reasoning_tokens_starts_in_thinking(self):
        """reasoning_tokens counts correctly when starts_in_thinking=True."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=3,
            total_tokens=13,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
        )
        tokens = [
            _make_response("hmm", 0),
            _make_response("ok", 1),
            _make_response("</think>", 2),
            GenerationResponse(
                text="answer", token=3, finish_reason="stop", usage=usage
            ),
        ]
        results = _step_until_finish(
            count_reasoning_tokens(
                parse_thinking_models(
                    _queue_source(tokens),
                    think_start="<think>",
                    think_end="</think>",
                    starts_in_thinking=True,
                )
            )
        )
        final = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(final) == 1
        assert final[0].usage is not None
        assert final[0].usage.completion_tokens_details.reasoning_tokens == 2


# ── parse_tool_calls (generic) ──────────────────────────────────


def _dummy_parser_fn(text: str) -> dict[str, Any]:
    return {"name": "test_fn", "arguments": {"arg": text}}


_dummy_parser = make_mlx_parser("<tool_call>", "</tool_call>", _dummy_parser_fn)


class TestGenericToolCallsFinishReason:
    def test_finish_reason_after_complete_tool_call(self):
        tokens = [
            _make_response("<tool_call>", 0),
            _make_response("body", 1),
            _make_response("</tool_call>", 2),
            _make_response("extra text", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_tool_calls(
                _queue_source(tokens),
                _dummy_parser,
                tools=None,
            )
        )
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1

    def test_finish_reason_mid_tool_call_unclosed(self):
        tokens = [
            _make_response("<tool_call>", 0),
            _make_response("partial content", 1, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_tool_calls(
                _queue_source(tokens),
                _dummy_parser,
                tools=None,
            )
        )
        assert _got_finish(results)

    def test_finish_reason_no_tool_calls(self):
        tokens = [
            _make_response("Just", 0),
            _make_response(" a", 1),
            _make_response(" normal", 2),
            _make_response(" response.", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_tool_calls(
                _queue_source(tokens),
                _dummy_parser,
                tools=None,
            )
        )
        assert _got_finish(results)


# ── Double parser chain (parse_thinking_models → parse_deepseek_v32) ──


class TestDeepSeekV32StartsInThinking:
    """Regression tests for deepseek v3.2 where the chat template appends
    <think> to the prompt so the model starts already inside a thinking block.
    """

    def test_reasoning_tagged_when_starts_in_thinking(self):
        tokens = [
            _make_response("let me", 0),
            _make_response(" think", 1),
            _make_response(THINKING_END, 2),
            _make_response("\n", 3),
            _make_response("42", 4, finish_reason="stop"),
        ]
        thinking = parse_thinking_models(
            _queue_source(tokens),
            think_start=THINKING_START,
            think_end=THINKING_END,
            starts_in_thinking=True,
        )
        results = _step_until_finish(parse_deepseek_v32(thinking))
        gens = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is None
        ]
        texts = [(r.text, r.is_thinking) for r in gens]
        assert texts == [("let me", True), (" think", True), ("\n", False)]
        final = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(final) == 1
        assert final[0].text == "42"
        assert final[0].is_thinking is False

    def test_starts_in_thinking_then_tool_call(self):
        tokens = [
            _make_response("need weather", 0),
            _make_response(THINKING_END, 1),
            _make_response("\n\n", 2),
            _make_response(TOOL_CALLS_START, 3),
            _make_response("\n", 4),
            _make_response(f'<{DSML_TOKEN}invoke name="get_weather">\n', 5),
            _make_response(
                f'<{DSML_TOKEN}parameter name="city" string="true">NYC</{DSML_TOKEN}parameter>\n',
                6,
            ),
            _make_response(f"</{DSML_TOKEN}invoke>\n", 7),
            _make_response(TOOL_CALLS_END, 8, finish_reason="stop"),
        ]
        thinking = parse_thinking_models(
            _queue_source(tokens),
            think_start=THINKING_START,
            think_end=THINKING_END,
            starts_in_thinking=True,
        )
        results = _step_until_finish(parse_deepseek_v32(thinking))
        reasoning_gens = [
            r
            for r in results
            if isinstance(r, GenerationResponse)
            and r.finish_reason is None
            and r.is_thinking
        ]
        assert [r.text for r in reasoning_gens] == ["need weather"]
        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_calls[0].name == "get_weather"

    def test_reasoning_tokens_counted_starts_in_thinking(self):
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
        )
        tokens = [
            _make_response("reasoning", 0),
            _make_response(" more", 1),
            _make_response(THINKING_END, 2),
            _make_response("\n", 3),
            GenerationResponse(text="42", token=4, finish_reason="stop", usage=usage),
        ]
        thinking = parse_thinking_models(
            _queue_source(tokens),
            think_start=THINKING_START,
            think_end=THINKING_END,
            starts_in_thinking=True,
        )
        results = _step_until_finish(
            count_reasoning_tokens(parse_deepseek_v32(thinking))
        )
        final = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is not None
        ]
        assert len(final) == 1
        assert final[0].usage is not None
        assert final[0].usage.completion_tokens_details.reasoning_tokens == 2


class TestBatchGeneratorSingleNext:
    def test_finish_reason_with_buffered_tokens_drain_loop(self):
        from exo.worker.runner.llm_inference.batch_generator import GeneratorQueue

        queue: GeneratorQueue[GenerationResponse] = GeneratorQueue()
        parser = parse_deepseek_v32(queue.gen())

        tokens = [
            _make_response("Hello ", 0),
            _make_response(" `<", 1),
            _make_response("", 2, finish_reason="stop"),
        ]

        collected: list[GenerationResponse | ToolCallResponse] = []
        for token in tokens:
            queue.push(token)
            while (parsed := next(parser, None)) is not None:
                collected.append(parsed)
            if token.finish_reason is not None:
                break

        assert _got_finish(collected), (
            f"No finish_reason in collected: {[(type(r).__name__, getattr(r, 'finish_reason', None) if isinstance(r, GenerationResponse) else 'tool') for r in collected]}"
        )


# ── parse_thinking_models prefix buffering ──────────────────────


def _drain_text(
    results: list[GenerationResponse | ToolCallResponse],
) -> str:
    return "".join(
        r.text
        for r in results
        if isinstance(r, GenerationResponse) and r.finish_reason is None
    )


class TestThinkingModelsPrefixBuffering:
    def test_lone_lt_is_preserved(self):
        tokens = [
            _make_response("<", 0),
            _make_response("function", 1),
            _make_response(">", 2),
            _make_response("", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _drain_text(results) == "<function>"
        gens = [r for r in results if isinstance(r, GenerationResponse)]
        assert all(not r.is_thinking for r in gens)

    def test_lone_lt_slash_is_preserved(self):
        tokens = [
            _make_response("</", 0),
            _make_response("parameter", 1),
            _make_response(">", 2),
            _make_response("", 3, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _drain_text(results) == "</parameter>"

    def test_partial_prefix_then_diverge(self):
        tokens = [
            _make_response("<", 0),
            _make_response("t", 1),
            _make_response("h", 2),
            _make_response("other", 3),
            _make_response("", 4, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        assert _drain_text(results) == "<thother"

    def test_real_think_tag_still_swallowed(self):
        tokens = [
            _make_response("<", 0),
            _make_response("think", 1),
            _make_response(">", 2),
            _make_response("body", 3),
            _make_response("</", 4),
            _make_response("think", 5),
            _make_response(">", 6),
            _make_response("after", 7),
            _make_response("", 8, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        gens = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and r.finish_reason is None
        ]
        texts = [(r.text, r.is_thinking) for r in gens]
        assert texts == [("body", True), ("after", False)]

    def test_finish_reason_flushes_buffer(self):
        tokens = [
            _make_response("<", 0),
            _make_response("", 1, finish_reason="stop"),
        ]
        results = _step_until_finish(
            parse_thinking_models(
                _queue_source(tokens),
                think_start="<think>",
                think_end="</think>",
                starts_in_thinking=False,
            )
        )
        gens = [r for r in results if isinstance(r, GenerationResponse)]
        assert len(gens) == 2
        assert gens[0].text == "<"
        assert gens[0].is_thinking is False
        assert gens[0].finish_reason is None
        assert gens[1].finish_reason == "stop"
        assert gens[1].is_thinking is False

    def test_tool_call_after_prefix_tokens_parses(self):
        def _capture_parser(text: str) -> dict[str, Any]:
            return {"name": "captured", "arguments": {"raw": text}}

        tool_parser = make_mlx_parser("<tool_call>", "</tool_call>", _capture_parser)

        tokens = [
            _make_response("<tool_call>", 0),
            _make_response("\n", 1),
            _make_response("<", 2),
            _make_response("function", 3),
            _make_response("=glob", 4),
            _make_response(">", 5),
            _make_response("\n", 6),
            _make_response("<", 7),
            _make_response("parameter", 8),
            _make_response("=pattern", 9),
            _make_response(">", 10),
            _make_response("**/*", 11),
            _make_response("</", 12),
            _make_response("parameter", 13),
            _make_response(">", 14),
            _make_response("</", 15),
            _make_response("function", 16),
            _make_response(">", 17),
            _make_response("</tool_call>", 18, finish_reason="stop"),
        ]

        thinking = parse_thinking_models(
            _queue_source(tokens),
            think_start="<think>",
            think_end="</think>",
            starts_in_thinking=False,
        )
        results = _step_until_finish(
            parse_tool_calls(thinking, tool_parser, tools=None)
        )

        tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
        assert len(tool_results) == 1
        raw = json.loads(tool_results[0].tool_calls[0].arguments)["raw"]  # pyright: ignore[reportAny]
        assert "<function=glob>" in raw
        assert "<parameter=pattern>" in raw
        assert "</parameter>" in raw
        assert "</function>" in raw
