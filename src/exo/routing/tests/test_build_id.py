from importlib.metadata import PackageNotFoundError
from types import ModuleType

import pytest

from exo import __version__
from exo.routing.wire import _installed, build_id  # pyright: ignore[reportPrivateUsage]


def test_the_app_names_its_own_build() -> None:
    assert build_id({"EXO_BUILD_ID": " 1.0.72 (4f1c2a9) "}) == "1.0.72 (4f1c2a9)"


def test_a_source_build_is_named_by_its_package_and_runner_versions() -> None:
    # No git commit: one lockfile on several machines is one build
    name = build_id({})
    assert name == build_id({"EXO_BUILD_ID": ""})
    assert name.startswith(f"exo {__version__}, mlx ")
    assert ", mlx-lm " in name


def _no_metadata(distribution: str) -> str:
    raise PackageNotFoundError(distribution)


def test_a_frozen_build_reads_the_runner_versions_from_the_modules() -> None:
    # The release bundle has mlx.core and mlx_lm but not their package metadata
    modules = {
        "mlx.core": "0.32.4.dev20261102+36969c09",
        "mlx_lm": "0.32.3.dev16+g6b572ab1b",
    }

    def load(name: str) -> ModuleType:
        module = ModuleType(name)
        module.__version__ = modules[name]  # pyright: ignore[reportAttributeAccessIssue]
        return module

    assert _installed("mlx", _no_metadata, load) == "0.32.4.dev20261102+36969c09"
    assert _installed("mlx-lm", _no_metadata, load) == "0.32.3.dev16+g6b572ab1b"


@pytest.mark.parametrize("version_attribute", [None, "", 32])
def test_a_runner_without_metadata_or_a_version_is_none(
    version_attribute: object,
) -> None:
    def missing(name: str) -> ModuleType:
        raise ModuleNotFoundError(name)

    def without_version(name: str) -> ModuleType:
        module = ModuleType(name)
        if version_attribute is not None:
            module.__version__ = version_attribute  # pyright: ignore[reportAttributeAccessIssue]
        return module

    assert _installed("mlx", _no_metadata, missing) == "none"
    assert _installed("mlx", _no_metadata, without_version) == "none"


def test_package_metadata_comes_first() -> None:
    def never(name: str) -> ModuleType:
        raise AssertionError(f"imported {name}")

    assert _installed("mlx", lambda _: "0.32.4", never) == "0.32.4"
