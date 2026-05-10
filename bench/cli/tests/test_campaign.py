"""Unit tests for ``bench.cli.campaign``.

The pure helpers (defaults+run merge, table-lookup, str-or-none) are
tested here. End-to-end campaign execution requires a real eco cluster
and is exercised manually via ``bench campaign <toml>``.
"""

from __future__ import annotations

from bench.cli.campaign import (
    _merge,  # type: ignore[reportPrivateUsage]
    _str_or_none,  # type: ignore[reportPrivateUsage]
    _table,  # type: ignore[reportPrivateUsage]
)

# ---------------------------------------------------------------------------
# _table
# ---------------------------------------------------------------------------


class TestTable:
    def test_present_table(self) -> None:
        data = {"defaults": {"nodes": 4}}
        assert _table(data, "defaults") == {"nodes": 4}

    def test_missing_key_returns_empty(self) -> None:
        assert _table({}, "absent") == {}

    def test_non_table_value_returns_empty(self) -> None:
        # `nodes = 4` is an int at top level, not a table; treat as empty.
        assert _table({"nodes": 4}, "nodes") == {}

    def test_list_value_returns_empty(self) -> None:
        assert _table({"runs": [{"a": 1}]}, "runs") == {}


# ---------------------------------------------------------------------------
# _merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_run_wins_on_conflict(self) -> None:
        defaults = {"nodes": 4, "tg": 64}
        run = {"nodes": 2}
        assert _merge(defaults, run) == {"nodes": 2, "tg": 64}

    def test_disjoint_keys(self) -> None:
        defaults = {"nodes": 4}
        run = {"model": "test/foo"}
        assert _merge(defaults, run) == {"nodes": 4, "model": "test/foo"}

    def test_run_only(self) -> None:
        assert _merge({}, {"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_defaults_only(self) -> None:
        assert _merge({"a": 1}, {}) == {"a": 1}

    def test_tags_deep_merged_defaults_only(self) -> None:
        defaults = {"tags": {"operator": "ciaranbor"}}
        run = {"model": "test/foo"}
        merged = _merge(defaults, run)
        assert merged["tags"] == {"operator": "ciaranbor"}

    def test_tags_deep_merged_run_only(self) -> None:
        defaults = {"nodes": 4}
        run = {"tags": {"model_short": "llama-3b"}}
        merged = _merge(defaults, run)
        assert merged["tags"] == {"model_short": "llama-3b"}

    def test_tags_deep_merged_both(self) -> None:
        defaults = {"tags": {"operator": "ciaranbor", "campaign": "smoke"}}
        run = {"tags": {"model_short": "llama-3b"}}
        merged = _merge(defaults, run)
        assert merged["tags"] == {
            "operator": "ciaranbor",
            "campaign": "smoke",
            "model_short": "llama-3b",
        }

    def test_run_tags_override_defaults_tags(self) -> None:
        defaults = {"tags": {"operator": "ciaranbor"}}
        run = {"tags": {"operator": "alice"}}
        merged = _merge(defaults, run)
        assert merged["tags"] == {"operator": "alice"}

    def test_no_tags_table_means_no_tags_key(self) -> None:
        # When neither side has tags, we don't synthesise an empty dict.
        merged = _merge({"nodes": 4}, {"model": "test/foo"})
        assert "tags" not in merged


# ---------------------------------------------------------------------------
# _str_or_none
# ---------------------------------------------------------------------------


class TestStrOrNone:
    def test_str_passes_through(self) -> None:
        assert _str_or_none("hello") == "hello"

    def test_none_returns_none(self) -> None:
        assert _str_or_none(None) is None

    def test_int_returns_none(self) -> None:
        assert _str_or_none(42) is None

    def test_list_returns_none(self) -> None:
        assert _str_or_none(["a", "b"]) is None
