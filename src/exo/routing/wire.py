"""What decides whether two exo nodes can talk: the wire version and the build.

Nodes connect only when they run the same ``WIRE_VERSION`` and the same build (see
``rust/networking/src/wire.rs``). Every message model is strict (``extra="forbid"``), so nodes
whose schemas differ can't read each other: a node that meets an event it can't read stalls at
that index until it restarts, and a node can't read a state snapshot that has a field it
doesn't know. Two builds can also disagree inside a model, between its ranks: MLX, mlx-lm and
exo's sharding code all change between builds, and a model split across them can hang part
way through or give wrong tokens. So different builds never cluster; each is told the other
runs another build instead.

Bump ``WIRE_VERSION`` when what nodes send each other changes. ``wire_schema.py`` keeps a
snapshot of the schema and a ledger of versions, and a test fails when the code no longer
matches them; its docstring says what to do.
"""

import os
from collections.abc import Callable, Mapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import Final, Literal, final

from exo import __version__
from exo.utils.pydantic_ext import FrozenModel

# Bumped by hand for a change in how nodes behave towards each other that their schema doesn't
# show (each version's schema hash is in wire_versions.txt):
# - 2: an instance's runners restart in place by a new rule. A node replaces its runner only
#   once the other nodes' runners that had joined it are going, and reports its own as going
#   until then; a node of version 1 never reports a runner it stops, so the two wait on each
#   other for good.
WIRE_VERSION: Final = 6

# The cluster name when none is given. Not the package version: the wire version and the build
# decide which nodes can talk, and one name for every version lets a node name a device that
# runs another version, instead of seeing another cluster. Give a cluster its own name to keep
# it apart.
DEFAULT_CLUSTER_NAME: Final = "exo"

# Set by the macOS app to its release and commit
BUILD_ID_ENV: Final = "EXO_BUILD_ID"


# The module whose ``__version__`` names a runner package's version when its package metadata is
# missing: the release bundle carries the modules but not their metadata
_VERSION_MODULES: Final = {"mlx": "mlx.core", "mlx-lm": "mlx_lm"}


def _installed(
    distribution: str,
    metadata_version: Callable[[str], str] = version,
    load_module: Callable[[str], ModuleType] = import_module,
) -> str:
    try:
        return metadata_version(distribution)
    except PackageNotFoundError:
        pass
    try:
        module = load_module(_VERSION_MODULES[distribution])
    except ImportError:
        return "none"
    found: object = getattr(module, "__version__", None)
    return found if isinstance(found, str) and found else "none"


def build_id(environ: Mapping[str, str] = os.environ) -> str:
    """Which build of exo this node runs: nodes connect only to the same build.

    The macOS app passes its own in ``EXO_BUILD_ID``. A source build is named by exo's package
    version and the exact MLX and mlx-lm versions it runs (a git build's version names its
    revision; a frozen build without package metadata reads them from the modules), not by a
    git commit: developers running one lockfile on several machines connect. That assumes
    source builds of one package version run the same exo code; two checkouts that change what
    a model's ranks send each other must not run together.
    """
    given = environ.get(BUILD_ID_ENV, "").strip()
    if given:
        return given
    return f"exo {__version__}, mlx {_installed('mlx')}, mlx-lm {_installed('mlx-lm')}"


@final
class DeviceOnOtherVersion(FrozenModel):
    """A device discovery heard under this cluster's name that can't join it."""

    # Every address it was heard on: one per network interface
    addresses: list[str]
    # older_exo: from before wire versions; wire_version: another wire version (given);
    # build: this wire version, but another build
    reason: Literal["older_exo", "wire_version", "build"]
    wire_version: int | None
