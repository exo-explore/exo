"""Tests for per-model sampling defaults (SamplingDefaults + apply_sampling_defaults).

Covers three invariants:
  1. Explicit request values are never overwritten by card defaults.
  2. Card defaults fill *absent* (None) fields in the request.
  3. A card without sampling_defaults is a no-op (zero allocation).
"""

import pytest

from exo.shared.models.model_cards import SamplingDefaults
from exo.worker.engines.mlx.utils_mlx import apply_sampling_defaults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(**kwargs):
    """Build a minimal TextGenerationTaskParams with the given overrides."""
    from unittest.mock import MagicMock
    from exo.shared.types.text_generation import TextGenerationTaskParams
    from exo.shared.types.common import ModelId

    defaults = dict(
        model=ModelId("mlx-community/test-model"),
        input=[],
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        repetition_context_size=None,
    )
    defaults.update(kwargs)
    return TextGenerationTaskParams.model_validate(defaults)


CARD_DEFAULTS = SamplingDefaults(
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestApplySamplingDefaults:
    def test_fills_absent_fields(self):
        """Card defaults must populate None slots in the request."""
        params = _make_params()
        result = apply_sampling_defaults(params, CARD_DEFAULTS)

        assert result.temperature == 1.0
        assert result.top_p == 0.95
        assert result.top_k == 40
        assert result.repetition_penalty == 1.1

    def test_explicit_request_value_wins(self):
        """An explicit caller value must never be overwritten."""
        params = _make_params(temperature=0.3, top_p=0.9)
        result = apply_sampling_defaults(params, CARD_DEFAULTS)

        assert result.temperature == 0.3   # caller's value preserved
        assert result.top_p == 0.9         # caller's value preserved
        assert result.top_k == 40          # card default applied (was None)

    def test_noop_when_no_defaults(self):
        """Empty SamplingDefaults must return the *exact same* object (fast path)."""
        params = _make_params()
        empty = SamplingDefaults()
        result = apply_sampling_defaults(params, empty)

        assert result is params  # identity check — no copy created

    def test_noop_when_all_explicit(self):
        """When every field is already set, defaults are irrelevant."""
        params = _make_params(
            temperature=0.7,
            top_p=0.8,
            top_k=50,
            repetition_penalty=1.05,
        )
        result = apply_sampling_defaults(params, CARD_DEFAULTS)

        assert result.temperature == 0.7
        assert result.top_p == 0.8
        assert result.top_k == 50
        assert result.repetition_penalty == 1.05

    @pytest.mark.parametrize("field,value", [
        ("temperature", 0.5),
        ("top_p", 0.7),
        ("top_k", 20),
        ("repetition_penalty", 1.05),
        ("repetition_context_size", 30),
    ])
    def test_individual_field_override(self, field: str, value: float | int):
        """Each field is independently preserved when explicitly supplied."""
        params = _make_params(**{field: value})
        result = apply_sampling_defaults(params, CARD_DEFAULTS)

        assert getattr(result, field) == value


class TestSamplingDefaultsFillMissing:
    def test_merge_prefers_incoming(self):
        """fill_missing must keep existing values from the overrides dict."""
        defaults = SamplingDefaults(temperature=1.0, top_p=0.95)
        overrides = {"temperature": 0.3}

        merged = defaults.fill_missing(overrides)

        assert merged["temperature"] == 0.3   # existing value kept
        assert merged["top_p"] == 0.95         # default applied

    def test_empty_overrides(self):
        """fill_missing({}) returns all non-None defaults."""
        defaults = SamplingDefaults(temperature=1.0)
        merged = defaults.fill_missing({})

        assert merged == {"temperature": 1.0}
