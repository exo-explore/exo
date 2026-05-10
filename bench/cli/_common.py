"""Shared CLI argument parsing for the bench command-line interface.

Every benchmark subcommand inherits the same model / cluster / output
arguments via :func:`add_shared_args` and consumes them through
:class:`SharedOptions`. argparse's ``Namespace.<attr>`` is fundamentally
typed ``Any``; the :func:`get_arg` / :func:`get_arg_optional` helpers are
the single boundary where we coerce to typed values.

A ``--config <path>.toml`` flag lets the caller capture a run definition
in a TOML file. :func:`expand_config_in_argv` rewrites argv in place,
substituting the config's keys as CLI flags placed *before* any explicit
user args so that explicit CLI flags always win.
"""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from exo_tools.cluster import Chip, Thunderbolt
from exo_tools.harness import Comm, Sharding

_T = TypeVar("_T")


def get_arg(args: argparse.Namespace, name: str, type_: type[_T]) -> _T:
    """Return ``args.<name>``, asserting it's an instance of ``type_``.

    For ``int`` and ``float`` we additionally accept inputs that ``int(.)`` /
    ``float(.)`` would parse, since argparse's ``type=int`` already coerces
    cleanly on input but post-`set_defaults` callers may pass raw values.
    """
    raw: Any = getattr(args, name)  # type: ignore[reportAny]
    if isinstance(raw, type_):
        return raw
    if type_ is int and isinstance(raw, (int, str)):
        return int(raw)  # type: ignore[return-value]
    if type_ is float and isinstance(raw, (int, float, str)):
        return float(raw)  # type: ignore[return-value]
    raise TypeError(
        f"argparse field {name!r} expected {type_.__name__}, got {type(raw).__name__}"  # type: ignore[reportUnknownArgumentType]
    )


def get_arg_optional(args: argparse.Namespace, name: str, type_: type[_T]) -> _T | None:
    """Like :func:`get_arg` but allows the field to be missing or None."""
    raw = getattr(args, name, None)
    if raw is None:
        return None
    if isinstance(raw, type_):
        return raw
    if type_ is int and isinstance(raw, (int, str)):
        return int(raw)  # type: ignore[return-value]
    if type_ is float and isinstance(raw, (int, float, str)):
        return float(raw)  # type: ignore[return-value]
    raise TypeError(
        f"argparse field {name!r} expected {type_.__name__} or None, "
        f"got {type(raw).__name__}"  # type: ignore[reportUnknownArgumentType]
    )


@dataclass(frozen=True)
class SharedOptions:
    """Parsed shared CLI options for any benchmark."""

    model: str
    hosts: tuple[str, ...]
    nodes: int
    thunderbolt: Thunderbolt | None
    chip: Chip | None
    min_memory_gb: float | None
    max_memory_gb: float | None
    min_disk_gb: float | None
    max_disk_gb: float | None
    evict_downloads: bool
    sharding: Sharding
    comm: Comm
    min_nodes: int
    output_dir: Path
    tags: dict[str, str]
    cleanup_instance: bool
    user_prefix: str

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> SharedOptions:
        hosts_raw = get_arg_optional(args, "hosts", str)
        thunderbolt_raw = get_arg_optional(args, "thunderbolt", str)
        chip_raw = get_arg_optional(args, "chip", str)
        tag_list_raw: object = getattr(args, "tag", None) or []
        if isinstance(tag_list_raw, list):
            tag_list: list[str] = [
                str(t)  # type: ignore[reportUnknownArgumentType]
                for t in tag_list_raw  # type: ignore[reportUnknownVariableType]
            ]
        else:
            tag_list = []
        return cls(
            model=get_arg(args, "model", str),
            hosts=tuple(_parse_csv(hosts_raw)) if hosts_raw else (),
            nodes=get_arg(args, "nodes", int),
            thunderbolt=Thunderbolt(thunderbolt_raw) if thunderbolt_raw else None,
            chip=Chip(chip_raw) if chip_raw else None,
            min_memory_gb=get_arg_optional(args, "min_memory_gb", float),
            max_memory_gb=get_arg_optional(args, "max_memory_gb", float),
            min_disk_gb=get_arg_optional(args, "min_disk_gb", float),
            max_disk_gb=get_arg_optional(args, "max_disk_gb", float),
            evict_downloads=get_arg(args, "evict_downloads", bool),
            sharding=Sharding(get_arg(args, "sharding", str)),
            comm=Comm(get_arg(args, "comm", str)),
            min_nodes=get_arg(args, "min_nodes", int),
            output_dir=Path(get_arg(args, "output_dir", str)),
            tags=_parse_tags(tag_list),
            cleanup_instance=get_arg(args, "cleanup_instance", bool),
            user_prefix=get_arg(args, "eco_user_prefix", str),
        )


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared-arg group on ``parser``.

    The bool flags (``--auto-constrain``, ``--evict-downloads``,
    ``--cleanup-instance``) all default to True and use
    :class:`argparse.BooleanOptionalAction` so callers opt out via the
    ``--no-X`` form (or set ``X = false`` in a TOML config).
    """
    g_config = parser.add_argument_group("config file")
    g_config.add_argument(
        "--config",
        default=None,
        help="TOML file with run parameters. CLI flags placed after --config "
        "override values from the file.",
    )

    g_model = parser.add_argument_group("model")
    g_model.add_argument(
        "--model",
        required=True,
        help="HuggingFace model id. To run multiple models in one go, use "
        "the 'campaign' subcommand with a TOML file listing each as a "
        "separate [[runs]] entry.",
    )
    g_model.add_argument(
        "--sharding",
        default=Sharding.TENSOR.value,
        choices=[s.value for s in Sharding],
        help="Sharding mode for the placed instance. Default 'Tensor' (splits "
        "layers within nodes; pairs with --comm MlxJaccl for high throughput "
        "on TB-connected clusters). Use 'Pipeline' for layer-per-node sharding "
        "(typical for single-node smoke tests).",
    )
    g_model.add_argument(
        "--comm",
        default=Comm.JACCL.value,
        choices=[c.value for c in Comm],
        help="Inter-node communication mode. Default 'MlxJaccl' (RDMA over "
        "Thunderbolt; pairs with --sharding Tensor and --thunderbolt a2a). "
        "Use 'MlxRing' for ring all-reduce over the regular network.",
    )
    g_model.add_argument("--min-nodes", type=int, default=1)

    g_cluster = parser.add_argument_group("cluster")
    g_cluster.add_argument(
        "--hosts",
        default=None,
        help="Comma-separated host list (e.g. s4,s9). Bypasses constraint search.",
    )
    g_cluster.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of cluster nodes (hosts) to deploy on. "
        "Distinct from --min-nodes which controls the model's instance placement.",
    )
    g_cluster.add_argument(
        "--thunderbolt",
        default=Thunderbolt.A2A.value,
        choices=[t.value for t in Thunderbolt],
        help="Thunderbolt topology required: 'a2a' (clique, default; needed "
        "for tensor parallelism + JACCL), 'ring' (cycle; for pipeline + JACCL), "
        "or 'none' (exclude TB-connected hosts; pair with --sharding Pipeline "
        "--comm MlxRing).",
    )
    g_cluster.add_argument(
        "--chip",
        default=None,
        choices=[c.value for c in Chip],
        help="Chip required (e.g. 'M3 Ultra')",
    )
    g_cluster.add_argument(
        "--min-memory-gb",
        type=float,
        default=None,
        help="Min RAM (GB) on each host. If unset, auto-derived from the HF "
        "model size (×1.30 + 1 GiB).",
    )
    g_cluster.add_argument(
        "--max-memory-gb",
        type=float,
        default=None,
        help="Max RAM (GB) on each host. Useful to leave bigger machines "
        "free for other workloads.",
    )
    g_cluster.add_argument(
        "--min-disk-gb",
        type=float,
        default=None,
        help="Min free disk (GB) on each host. If unset, auto-derived from "
        "the HF model size (×1.10 + 1 GiB).",
    )
    g_cluster.add_argument(
        "--max-disk-gb",
        type=float,
        default=None,
        help="Max disk (GB) on each host.",
    )

    g_runtime = parser.add_argument_group("runtime")
    g_runtime.add_argument(
        "--evict-downloads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-evict existing models (smallest first) when disk is short to "
        "make room for the bench model. Default on; pass --no-evict-downloads "
        "to keep existing downloads.",
    )
    g_runtime.add_argument(
        "--cleanup-instance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean up the placed instance after the benchmark exits. "
        "Default on; pass --no-cleanup-instance to leave it running for debugging.",
    )
    g_runtime.add_argument(
        "--eco-user-prefix",
        default="bench",
        help="USER prefix for the eco session (default: 'bench').",
    )

    g_output = parser.add_argument_group("output")
    g_output.add_argument(
        "--output-dir",
        default="bench/results",
        help="Base directory for JSON results. Subcommands may add a sub-folder.",
    )
    g_output.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Add a 'key=value' tag to metadata.tags (repeatable).",
    )


# ---------------------------------------------------------------------------
# TOML config expansion
# ---------------------------------------------------------------------------


def expand_config_in_argv(argv: list[str]) -> list[str]:
    """If ``--config <path>`` appears in ``argv``, splice the TOML's contents in.

    The TOML file's keys are converted to CLI flags (``foo_bar`` →
    ``--foo-bar``) and inserted *before* the user's other args, so explicit
    CLI flags always override the config. The ``--config <path>`` pair
    itself is removed from argv. The first arg (the subcommand name) is
    preserved at index 0.

    Special handling:
      - ``[tags]`` table → repeated ``--tag key=value`` occurrences
      - lists → joined as a comma-separated value (matches the parser's
        CSV handling for ``--hosts``)
      - bool true/false → ``--key`` / ``--no-key`` (assumes the underlying
        flag uses :class:`argparse.BooleanOptionalAction`)
    """
    if "--config" not in argv:
        return list(argv)

    idx = argv.index("--config")
    if idx + 1 >= len(argv):
        raise ValueError("--config requires a path argument")
    config_path = Path(argv[idx + 1])
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as f:
        config_data: dict[str, Any] = tomllib.load(f)

    expanded = _config_to_argv(config_data)
    stripped = list(argv[:idx]) + list(argv[idx + 2 :])
    if not stripped:
        return expanded
    # The subcommand name must come first; insert config-derived args
    # right after it so that the user's later explicit args override.
    return [stripped[0]] + expanded + stripped[1:]


def _config_to_argv(data: dict[str, Any]) -> list[str]:
    """Convert a TOML-loaded dict to a list of argv-style CLI flags."""
    out: list[str] = []
    for key in data:
        value: Any = data[key]  # type: ignore[reportAny]
        if key == "tags" and isinstance(value, dict):
            for tag_key, tag_value in value.items():  # type: ignore[reportUnknownVariableType]
                out.extend(["--tag", f"{tag_key}={tag_value}"])
            continue
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            out.append(flag if value else f"--no-{key.replace('_', '-')}")
        elif isinstance(value, list):
            joined = ",".join(
                str(x)  # type: ignore[reportUnknownArgumentType]
                for x in value  # type: ignore[reportUnknownVariableType]
            )
            out.extend([flag, joined])
        else:
            out.extend([flag, str(value)])  # type: ignore[reportAny]
    return out


def _parse_csv(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_tags(raw: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in raw:
        if "=" not in entry:
            raise argparse.ArgumentTypeError(
                f"--tag must be 'key=value', got {entry!r}"
            )
        k, v = entry.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@dataclass
class CommandResult:
    """Return value from a benchmark CLI handler."""

    output_path: Path | None = None
    extra: dict[str, str] = field(default_factory=dict)
