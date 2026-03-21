import pytest
from exo_core.types.text_generation import resolve_reasoning_params


def test_both_none_returns_none_none() -> None:
    assert resolve_reasoning_params(None, None) == (None, None)


def test_both_set_passes_through_unchanged() -> None:
    assert resolve_reasoning_params("high", True) == ("high", True)
    assert resolve_reasoning_params("none", True) == ("none", True)
    assert resolve_reasoning_params("low", False) == ("low", False)


def test_enable_thinking_true_derives_medium() -> None:
    assert resolve_reasoning_params(None, True) == ("medium", True)


def test_enable_thinking_false_derives_none() -> None:
    assert resolve_reasoning_params(None, False) == ("none", False)


def test_reasoning_effort_none_derives_thinking_false() -> None:
    assert resolve_reasoning_params("none", None) == ("none", False)


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh"])
def test_non_none_effort_derives_thinking_true(effort: str) -> None:
    assert resolve_reasoning_params(effort, None) == (effort, True)  # pyright: ignore[reportArgumentType]
