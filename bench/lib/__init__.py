"""Composable bench library for exo.

Provides reusable building blocks for benchmarks:

- :class:`bench.lib.session.BenchSession` — cluster + instance + client wrapper
- :class:`bench.lib.results.ResultsBundle` — structured results + JSON writer
- :func:`bench.lib.cluster.managed_cluster` /
  :func:`bench.lib.cluster.managed_instance` — eco-managed lifecycle ctx-managers
- :func:`bench.lib.model_meta.fetch_model_meta` — HF metadata fetcher driving
  cluster constraints + auto-derived context ramps
- :mod:`bench.lib.context_scaling` — prompt-TPS / decode-TPS vs context-size sweep

CLI entrypoints under ``bench/cli/`` consume this library via
``python -m bench.cli <subcommand>``. Adding a new benchmark = (i) write
``bench/lib/<name>.py`` exposing a typed ``run(session, params, bundle)``
callable, (ii) write ``bench/cli/<name>.py`` with an ``add_subparser`` and
a handler, (iii) register it in ``_REGISTRY`` in ``bench/cli/__main__.py``.
"""
