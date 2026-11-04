"""exo's single-machine cluster test kit: a stub engine (no MLX) and a multi-process harness.

Nothing in exo's production code imports this package. The harness starts real exo nodes
through `exo.testkit.node`, which swaps the runner's engine for the stub before exo starts.
See README.md in this directory.
"""
