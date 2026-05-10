"""Unit tests for the argparse boundary helpers in ``bench.cli._common``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from bench.cli._common import (
    _config_to_argv,  # type: ignore[reportPrivateUsage]
    _parse_csv,  # type: ignore[reportPrivateUsage]
    _parse_tags,  # type: ignore[reportPrivateUsage]
    expand_config_in_argv,
    get_arg,
    get_arg_optional,
)

# ---------------------------------------------------------------------------
# _parse_csv
# ---------------------------------------------------------------------------


class TestParseCsv:
    def test_simple_list(self) -> None:
        assert _parse_csv("a,b,c") == ["a", "b", "c"]

    def test_strips_whitespace(self) -> None:
        assert _parse_csv("  a , b , c  ") == ["a", "b", "c"]

    def test_skips_empty_entries(self) -> None:
        assert _parse_csv("a,,b,") == ["a", "b"]
        assert _parse_csv(",,,") == []

    def test_empty_string_returns_empty(self) -> None:
        assert _parse_csv("") == []

    def test_single_value(self) -> None:
        assert _parse_csv("only") == ["only"]


# ---------------------------------------------------------------------------
# _parse_tags
# ---------------------------------------------------------------------------


class TestParseTags:
    def test_empty_input_returns_empty_dict(self) -> None:
        assert _parse_tags([]) == {}

    def test_single_tag(self) -> None:
        assert _parse_tags(["operator=ciaranbor"]) == {"operator": "ciaranbor"}

    def test_multiple_tags(self) -> None:
        assert _parse_tags(["a=1", "b=2", "c=3"]) == {"a": "1", "b": "2", "c": "3"}

    def test_strips_whitespace_around_key_and_value(self) -> None:
        assert _parse_tags(["  key  =  value  "]) == {"key": "value"}

    def test_value_can_contain_equals(self) -> None:
        assert _parse_tags(["url=http://example.com/?a=b"]) == {
            "url": "http://example.com/?a=b"
        }

    def test_later_duplicate_key_wins(self) -> None:
        # Standard dict behaviour; explicit so we notice if it changes.
        assert _parse_tags(["k=v1", "k=v2"]) == {"k": "v2"}

    def test_missing_equals_raises(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="key=value"):
            _ = _parse_tags(["malformed"])

    def test_one_malformed_in_list_raises(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            _ = _parse_tags(["good=1", "bad", "alsogood=2"])


# ---------------------------------------------------------------------------
# get_arg / get_arg_optional
# ---------------------------------------------------------------------------


class TestGetArg:
    def test_str_passes_through(self) -> None:
        ns = argparse.Namespace(name="hello")
        assert get_arg(ns, "name", str) == "hello"

    def test_int_passes_through(self) -> None:
        ns = argparse.Namespace(count=42)
        assert get_arg(ns, "count", int) == 42

    def test_int_coerces_from_string(self) -> None:
        ns = argparse.Namespace(count="42")
        assert get_arg(ns, "count", int) == 42

    def test_float_passes_through(self) -> None:
        ns = argparse.Namespace(rate=3.14)
        assert get_arg(ns, "rate", float) == 3.14

    def test_float_coerces_from_int(self) -> None:
        ns = argparse.Namespace(rate=3)
        assert get_arg(ns, "rate", float) == 3.0

    def test_float_coerces_from_string(self) -> None:
        ns = argparse.Namespace(rate="3.14")
        assert get_arg(ns, "rate", float) == 3.14

    def test_bool_passes_through(self) -> None:
        ns = argparse.Namespace(flag=True)
        assert get_arg(ns, "flag", bool) is True

    def test_wrong_type_raises(self) -> None:
        ns = argparse.Namespace(name=42)
        with pytest.raises(TypeError, match="expected str"):
            _ = get_arg(ns, "name", str)

    def test_missing_attribute_raises(self) -> None:
        ns = argparse.Namespace()
        with pytest.raises(AttributeError):
            _ = get_arg(ns, "missing", str)


class TestGetArgOptional:
    def test_missing_returns_none(self) -> None:
        ns = argparse.Namespace()
        assert get_arg_optional(ns, "missing", str) is None

    def test_explicit_none_returns_none(self) -> None:
        ns = argparse.Namespace(value=None)
        assert get_arg_optional(ns, "value", str) is None

    def test_present_value_returns_typed(self) -> None:
        ns = argparse.Namespace(value="present")
        assert get_arg_optional(ns, "value", str) == "present"

    def test_int_coerces_from_string(self) -> None:
        ns = argparse.Namespace(value="42")
        assert get_arg_optional(ns, "value", int) == 42

    def test_float_coerces_from_int(self) -> None:
        ns = argparse.Namespace(value=42)
        assert get_arg_optional(ns, "value", float) == 42.0

    def test_wrong_type_raises(self) -> None:
        ns = argparse.Namespace(value=[1, 2, 3])
        with pytest.raises(TypeError, match="expected str or None"):
            _ = get_arg_optional(ns, "value", str)


# ---------------------------------------------------------------------------
# _config_to_argv
# ---------------------------------------------------------------------------


class TestConfigToArgv:
    def test_empty(self) -> None:
        assert _config_to_argv({}) == []

    def test_string_value(self) -> None:
        assert _config_to_argv({"model": "mlx/foo"}) == ["--model", "mlx/foo"]

    def test_int_and_float_values(self) -> None:
        out = _config_to_argv({"num_steps": 32, "fraction_of_max": 0.5})
        assert out == ["--num-steps", "32", "--fraction-of-max", "0.5"]

    def test_underscore_keys_become_hyphenated_flags(self) -> None:
        out = _config_to_argv({"min_memory_gb": 21.0})
        assert out == ["--min-memory-gb", "21.0"]

    def test_bool_true_emits_flag(self) -> None:
        assert _config_to_argv({"auto_constrain": True}) == ["--auto-constrain"]

    def test_bool_false_emits_no_form(self) -> None:
        assert _config_to_argv({"auto_constrain": False}) == ["--no-auto-constrain"]

    def test_none_value_skipped(self) -> None:
        assert _config_to_argv({"chip": None, "model": "foo"}) == [
            "--model",
            "foo",
        ]

    def test_list_joined_as_csv(self) -> None:
        out = _config_to_argv({"hosts": ["s4", "s9"], "cold_controls": [1024, 2048]})
        assert out == [
            "--hosts",
            "s4,s9",
            "--cold-controls",
            "1024,2048",
        ]

    def test_tags_table_expands_to_repeated_tag_args(self) -> None:
        out = _config_to_argv({"tags": {"operator": "ciaranbor", "run": "full"}})
        # Order within a TOML table is preserved by tomllib
        assert out == [
            "--tag",
            "operator=ciaranbor",
            "--tag",
            "run=full",
        ]


# ---------------------------------------------------------------------------
# expand_config_in_argv
# ---------------------------------------------------------------------------


class TestExpandConfigInArgv:
    def test_no_config_flag_passthrough(self) -> None:
        argv = ["context-scaling", "--model", "foo"]
        assert expand_config_in_argv(argv) == argv

    def test_config_at_end(self, tmp_path: Path) -> None:
        cfg = tmp_path / "run.toml"
        _ = cfg.write_text('model = "from_config"\nnum_steps = 16\n')
        argv = ["context-scaling", "--config", str(cfg)]
        # Config flags are inserted right after the subcommand
        assert expand_config_in_argv(argv) == [
            "context-scaling",
            "--model",
            "from_config",
            "--num-steps",
            "16",
        ]

    def test_explicit_cli_overrides_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "run.toml"
        _ = cfg.write_text('model = "from_config"\nnum_steps = 16\n')
        # User overrides --num-steps explicitly. Argparse takes the last
        # occurrence for non-append actions, so the user's 32 wins.
        argv = ["context-scaling", "--config", str(cfg), "--num-steps", "32"]
        out = expand_config_in_argv(argv)
        assert out == [
            "context-scaling",
            "--model",
            "from_config",
            "--num-steps",
            "16",
            "--num-steps",
            "32",
        ]

    def test_missing_path_arg_raises(self) -> None:
        with pytest.raises(ValueError, match="--config requires a path"):
            _ = expand_config_in_argv(["context-scaling", "--config"])

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _ = expand_config_in_argv(
                ["context-scaling", "--config", str(tmp_path / "missing.toml")]
            )

    def test_bool_false_in_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "run.toml"
        _ = cfg.write_text("auto_constrain = false\n")
        argv = ["context-scaling", "--config", str(cfg)]
        assert expand_config_in_argv(argv) == [
            "context-scaling",
            "--no-auto-constrain",
        ]

    def test_tags_table(self, tmp_path: Path) -> None:
        cfg = tmp_path / "run.toml"
        _ = cfg.write_text(
            'model = "foo"\n[tags]\noperator = "ciaranbor"\nrun = "full"\n'
        )
        argv = ["context-scaling", "--config", str(cfg)]
        assert expand_config_in_argv(argv) == [
            "context-scaling",
            "--model",
            "foo",
            "--tag",
            "operator=ciaranbor",
            "--tag",
            "run=full",
        ]
