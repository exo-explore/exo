"""Tests for the warn-on-extra-fields mixin."""

from collections.abc import Iterator

import pytest
from loguru import logger
from pydantic import AliasChoices, AliasPath, BaseModel, Field

from exo.utils.extra_fields_warner import (
    WarnExtraModel,
    _reset_rate_limit_state,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def captured_warnings() -> Iterator[list[str]]:
    """Capture loguru warnings for the duration of one test.

    Each test gets a fresh rate-limit cache so `(model, field)` pair state
    does not leak across tests.
    """
    _reset_rate_limit_state()
    messages: list[str] = []
    sink_id = logger.add(
        lambda msg: messages.append(str(msg)),
        level="WARNING",
        format="{message}",
    )
    try:
        yield messages
    finally:
        logger.remove(sink_id)
        _reset_rate_limit_state()


class _ChatLikeRequest(WarnExtraModel):
    """Stand-in for a real public API request model."""

    model: str
    temperature: float | None = None
    max_tokens: int | None = None


def test_known_fields_parse_without_warning(captured_warnings: list[str]) -> None:
    parsed = _ChatLikeRequest.model_validate(
        {"model": "qwen", "temperature": 0.5, "max_tokens": 100}
    )

    assert parsed.model == "qwen"
    assert parsed.temperature == 0.5
    assert parsed.max_tokens == 100
    assert captured_warnings == []


def test_unknown_field_is_dropped_and_warned(captured_warnings: list[str]) -> None:
    # Mistyped field name — the historical bug case (e.g. `instance_meta`
    # arriving as `instanceMeta` on a snake_case model).
    parsed = _ChatLikeRequest.model_validate(
        {"model": "qwen", "tempreture": 0.5}  # typo, codespell-ignore
    )

    # Behavior is unchanged: extra is dropped, request still parses.
    assert parsed.model == "qwen"
    assert parsed.temperature is None
    assert not hasattr(parsed, "tempreture")  # codespell-ignore

    # But a warning IS emitted, naming both the model and the field.
    assert len(captured_warnings) == 1
    msg = captured_warnings[0]
    assert "tempreture" in msg  # codespell-ignore
    assert "_ChatLikeRequest" in msg


def test_repeat_unknown_field_is_rate_limited(
    captured_warnings: list[str],
) -> None:
    for _ in range(50):
        _ChatLikeRequest.model_validate({"model": "qwen", "bogus": True})

    # All 50 requests parse, but only one warning is logged for this
    # (model, field) pair within the rate-limit window.
    assert len(captured_warnings) == 1
    assert "bogus" in captured_warnings[0]


def test_distinct_unknown_fields_each_warn_once(
    captured_warnings: list[str],
) -> None:
    _ChatLikeRequest.model_validate({"model": "qwen", "alpha": 1})
    _ChatLikeRequest.model_validate({"model": "qwen", "alpha": 2})  # repeat
    _ChatLikeRequest.model_validate({"model": "qwen", "beta": 3})

    # Two distinct fields → two warnings; the repeat of "alpha" is muted.
    assert len(captured_warnings) == 2
    fields_warned_on = {
        "alpha" if "alpha" in m else "beta" if "beta" in m else "?"
        for m in captured_warnings
    }
    assert fields_warned_on == {"alpha", "beta"}


def test_aliased_field_does_not_warn(captured_warnings: list[str]) -> None:
    class _Aliased(WarnExtraModel):
        model_id: str = Field(alias="model")

    parsed = _Aliased.model_validate({"model": "qwen"})
    assert parsed.model_id == "qwen"
    assert captured_warnings == []


def test_alias_choices_does_not_warn(captured_warnings: list[str]) -> None:
    class _MultiAlias(WarnExtraModel):
        thinking: bool = Field(
            default=False,
            validation_alias=AliasChoices("thinking", "enable_thinking"),
        )

    # Either declared name should be accepted without warning.
    parsed_a = _MultiAlias.model_validate({"thinking": True})
    parsed_b = _MultiAlias.model_validate({"enable_thinking": True})
    assert parsed_a.thinking is True
    assert parsed_b.thinking is True
    assert captured_warnings == []


def test_alias_path_top_level_key_does_not_warn(
    captured_warnings: list[str],
) -> None:
    class _Pathed(WarnExtraModel):
        nested_value: int = Field(
            default=0, validation_alias=AliasPath("outer", "inner")
        )

    # The validator can only see the top-level keys of the input dict;
    # "outer" is the top-level key for an AliasPath that drills into
    # `outer.inner`, so it must not trigger the unknown-field warning.
    parsed = _Pathed.model_validate({"outer": {"inner": 7}})
    assert parsed.nested_value == 7
    assert captured_warnings == []


def test_subclass_inherits_validator(captured_warnings: list[str]) -> None:
    # BenchChatCompletionRequest pattern: a subclass adds new fields and
    # should still warn on unknowns inherited or added.
    class _BenchChatLike(_ChatLikeRequest):
        use_prefix_cache: bool = False

    parsed = _BenchChatLike.model_validate(
        {"model": "qwen", "use_prefix_cache": True, "wrong_key": "x"}
    )

    assert parsed.use_prefix_cache is True
    assert len(captured_warnings) == 1
    assert "wrong_key" in captured_warnings[0]
    # Subclass name appears in the warning, not the parent's.
    assert "_BenchChatLike" in captured_warnings[0]


def test_non_dict_input_is_passthrough(captured_warnings: list[str]) -> None:
    # When pydantic dispatches the validator on a model instance (e.g.,
    # nested validation), the input is not a dict and should pass through
    # without inspection.
    inst = _ChatLikeRequest(model="qwen")
    again = _ChatLikeRequest.model_validate(inst)
    assert again.model == "qwen"
    assert captured_warnings == []


def test_does_not_affect_unrelated_basemodels(
    captured_warnings: list[str],
) -> None:
    # Sanity: a plain BaseModel still has the historical default (no
    # warning, no error). The mixin is opt-in.
    class _Plain(BaseModel):
        x: int

    parsed = _Plain.model_validate({"x": 1, "extra": "ignored"})
    assert parsed.x == 1
    assert captured_warnings == []
