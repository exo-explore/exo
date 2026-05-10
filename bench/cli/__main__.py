"""``python -m bench.cli`` — dispatcher for benchmark subcommands.

To add a new benchmark:

  1. Implement the methodology in ``bench.lib.<name>`` exposing a typed
     ``run(session, params, bundle)`` callable (no argparse, no eco I/O).
  2. Implement a ``bench.cli.<name>`` module with an ``add_subparser`` and
     a ``run(args) -> Path`` handler.
  3. Add an ``import + add_subparser(subparsers)`` line below.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from bench.cli import campaign, context_scaling, plot
from bench.cli._common import expand_config_in_argv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.cli",
        description=(
            "Composable, eco-managed benchmarks for exo. "
            "Pick a subcommand and pass its model / cluster options."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="subcommand",
        required=True,
        metavar="SUBCOMMAND",
    )
    context_scaling.add_subparser(subparsers)
    plot.add_subparser(subparsers)
    campaign.add_subparser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    expanded = expand_config_in_argv(raw_argv)
    args = _build_parser().parse_args(expanded)
    handler = getattr(args, "handler", None)
    if not callable(handler):
        subcommand = getattr(args, "subcommand", "<unknown>")
        raise SystemExit(f"subcommand {subcommand!r} did not register a handler")
    cast("Callable[[argparse.Namespace], Path]", handler)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
