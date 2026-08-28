"""Tests for streaming-safe stop-sequence scanning.

These cover the multi-token leak bug: a stop sequence spanning more than one
decoded token must not leak its leading bytes into the output before the whole
sequence is recognised.
"""

from exo.worker.engines.mlx.generator.stop_sequences import scan_stop_sequences


def _run_stream(tokens: list[str], stop_sequences: list[str]) -> tuple[str, str | None]:
    """Mirror the generator loop's use of scan_stop_sequences.

    Feeds ``tokens`` one at a time, accumulating the emitted text exactly as
    generate.py / batch_generate.py do, and flushing any held-back text on a
    natural end-of-stream. Returns ``(emitted_text, matched_stop_sequence)``.
    """
    pending = ""
    emitted: list[str] = []
    matched: str | None = None
    for token in tokens:
        pending += token
        emit, hit, pending = scan_stop_sequences(pending, stop_sequences)
        emitted.append(emit)
        if hit is not None:
            matched = hit
            break
    else:
        # Natural EOS: flush whatever was held back — it is real output.
        emitted.append(pending)
    return "".join(emitted), matched


class TestScanStopSequencesContract:
    def test_no_stop_sequences_passes_through(self):
        assert scan_stop_sequences("hello", []) == ("hello", None, "")

    def test_empty_stop_sequence_is_ignored(self):
        assert scan_stop_sequences("hello", [""]) == ("hello", None, "")

    def test_full_match_trims_and_reports(self):
        assert scan_stop_sequences("abEND", ["END"]) == ("ab", "END", "")

    def test_match_inside_single_buffer(self):
        assert scan_stop_sequences("Hello END world", ["END"]) == (
            "Hello ",
            "END",
            "",
        )

    def test_partial_suffix_is_held_back(self):
        # "EN" could still grow into "END" on the next token, so hold it.
        assert scan_stop_sequences("abEN", ["END"]) == ("ab", None, "EN")

    def test_non_matching_tail_is_not_held(self):
        assert scan_stop_sequences("abXY", ["END"]) == ("abXY", None, "")

    def test_earliest_full_match_wins(self):
        emit, matched, pending = scan_stop_sequences("aBBBcQQQ", ["QQQ", "BBB"])
        assert (emit, matched, pending) == ("a", "BBB", "")

    def test_longest_partial_across_sequences_is_held(self):
        # "XY" is a prefix of "XYZ"; "Q" is unrelated. Hold "XY".
        assert scan_stop_sequences("aXY", ["XYZ", "Q"]) == ("a", None, "XY")


class TestScanStopSequencesStreaming:
    def test_multi_token_stop_does_not_leak(self):
        # "END" arriving as "E" then "ND" must not emit the "E".
        assert _run_stream(["E", "ND"], ["END"]) == ("", "END")

    def test_text_then_multi_token_stop(self):
        assert _run_stream(["ABC", "DEF", "E", "ND"], ["END"]) == ("ABCDEF", "END")

    def test_single_token_stop(self):
        assert _run_stream(["END"], ["END"]) == ("", "END")

    def test_false_partial_is_released_on_continuation(self):
        # "E" is held, then "X" arrives proving it was not a stop sequence.
        assert _run_stream(["E", "X", "tra"], ["END"]) == ("EXtra", None)

    def test_held_partial_is_flushed_on_natural_eos(self):
        # Generation ends while a partial match is buffered: it is real output.
        assert _run_stream(["partialEN"], ["END"]) == ("partialEN", None)

    def test_no_stop_present(self):
        assert _run_stream(["foo", "bar"], ["END"]) == ("foobar", None)

    def test_stop_split_across_three_tokens(self):
        assert _run_stream(["12", "34", "56"], ["3456"]) == ("12", "3456")

    def test_multiple_stops_second_matches(self):
        assert _run_stream(["A", "B"], ["XYZ", "B"]) == ("A", "B")

    def test_pending_is_bounded_by_longest_stop(self):
        # However long the run, never hold back more than len(stop) - 1.
        _emit, _matched, pending = scan_stop_sequences("x" * 100 + "EN", ["END"])
        assert pending == "EN"
