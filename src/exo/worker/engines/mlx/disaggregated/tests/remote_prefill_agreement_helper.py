"""Child process for the remote-prefill rank agreement test.

Spawned once per MLX rank with the ``ring`` distributed backend on localhost. Each
rank computes, for a set of scenarios, the remote-prefill fallback decision that
``ExoBatchGenerator.submit()`` makes: every rank of a multi-node instance must agree on
whether any rank's remote fetch failed, otherwise one rank prefills locally (running
collectives) while another does not, which hangs.

The decision is ``mx_any(not fetch_ok, group)``: True when any rank failed (fall back and
prefill locally together), False when every rank's fetch succeeded (keep the remote
cache).

Run with ``--scenarios <json>`` where the json is a list of
``{"name": str, "rank0_ok": bool, "rank1_ok": bool, "expected": bool}``. This process
picks its own ``fetch_ok`` from its ``MLX_RANK``. MLX ring needs ``MLX_RANK`` and
``MLX_HOSTFILE`` in the environment. Results (one json object per line with each
scenario's name mapped to this rank's observed decision) go to stdout; if the ring
backend cannot initialise, a single ``{"init_failed": true}`` line is emitted instead.
"""

import json
import os
import sys
from typing import TypedDict, cast

import mlx.core as mx

from exo.worker.engines.mlx.utils_mlx import mx_any


class Scenario(TypedDict):
    name: str
    rank0_ok: bool
    rank1_ok: bool
    expected: bool


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload), flush=True)


def _scenarios_json() -> str:
    # sys.argv is list[str], so indexing yields a typed str (argparse returns Any).
    index = sys.argv.index("--scenarios")
    return sys.argv[index + 1]


def main() -> int:
    scenarios: list[Scenario] = cast(list[Scenario], json.loads(_scenarios_json()))

    rank = int(os.environ["MLX_RANK"])
    try:
        group = mx.distributed.init(backend="ring", strict=True)
    except Exception:
        _emit({"init_failed": True})
        return 2

    observed: dict[str, bool] = {}
    for scenario in scenarios:
        fetch_ok = scenario["rank0_ok"] if rank == 0 else scenario["rank1_ok"]
        # The submit() fallback decision: fall back when any rank's fetch failed.
        decision = mx_any(not fetch_ok, group)
        observed[scenario["name"]] = decision
        if decision != scenario["expected"]:
            _emit(
                {
                    "name": scenario["name"],
                    "decision": decision,
                    "expected": scenario["expected"],
                }
            )
            return 1

    _emit({"rank": rank, "world_size": group.size(), "observed": observed})
    return 0


if __name__ == "__main__":
    sys.exit(main())
