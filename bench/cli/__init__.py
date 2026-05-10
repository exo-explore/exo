"""CLI front-ends for bench library benchmarks.

Each benchmark is a sub-package / module with two pieces:

  - a ``run(...)`` callable in ``bench.lib.<name>`` that does the actual
    measurement (no argparse, no eco, no I/O)
  - an ``add_subparser(subparsers)`` helper here that wires CLI args to a
    handler invoking the lib

The main entry point dispatches to the requested subcommand:

    uv run python -m bench.cli context-scaling --hosts s4 \\
        --model mlx-community/Qwen3-30B-A3B-4bit
"""
