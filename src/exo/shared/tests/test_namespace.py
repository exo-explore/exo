import sys

import pytest

from exo import __version__
from exo.main import Args, default_namespace


def test_defaults_to_version() -> None:
    assert default_namespace({}) == __version__


def test_zenoh_namespace_env() -> None:
    assert default_namespace({"EXO_ZENOH_NAMESPACE": "my-cluster"}) == "my-cluster"


def test_legacy_libp2p_namespace_env_still_isolates() -> None:
    assert default_namespace({"EXO_LIBP2P_NAMESPACE": "old-cluster"}) == "old-cluster"


def test_zenoh_namespace_wins_over_legacy() -> None:
    environ = {"EXO_ZENOH_NAMESPACE": "new", "EXO_LIBP2P_NAMESPACE": "old"}
    assert default_namespace(environ) == "new"


def test_empty_env_falls_back_to_version() -> None:
    assert default_namespace({"EXO_ZENOH_NAMESPACE": ""}) == __version__


def test_args_parse_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The macOS app sets EXO_ZENOH_NAMESPACE rather than passing --namespace
    monkeypatch.setenv("EXO_ZENOH_NAMESPACE", "from-env")
    monkeypatch.setattr(sys, "argv", ["exo"])
    assert Args.parse().namespace == "from-env"


def test_args_parse_flag_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_ZENOH_NAMESPACE", "from-env")
    monkeypatch.setattr(sys, "argv", ["exo", "--namespace", "from-flag"])
    assert Args.parse().namespace == "from-flag"
