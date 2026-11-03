"""Ranks of a multi-node decode instance must agree on the remote-prefill outcome.

After each rank fetches its remote prefill independently, ``ExoBatchGenerator.submit()``
checks ``mx_any(not remote_prefilled, self.group)``: if any rank's fetch failed, every
rank discards its cache and prefills the full prompt locally together. If the ranks
disagreed, one rank would run local-prefill collectives while another would not, which
hangs.

This runs the agreement across two real MLX processes over the ``ring`` distributed
backend on localhost. Driving the whole ``submit()`` path needs a real loaded model, so
the processes exercise the agreement primitive ``mx_any`` directly, on both the True/False
combinations the caller submits, and check that both ranks observe the same, correct
fallback decision (the exact predicate submit() uses).
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict, cast

import pytest

from exo.worker.engines.mlx.disaggregated.tests.remote_prefill_agreement_helper import (
    Scenario,
)

_HELPER = Path(__file__).with_name("remote_prefill_agreement_helper.py")


class ObservedRanks(TypedDict):
    rank: int
    world_size: int
    observed: dict[str, bool]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = cast(tuple[str, int], sock.getsockname())[1]
        return int(port)


@pytest.mark.slow
def test_ranks_agree_on_remote_prefill_fallback() -> None:
    # Ring backend reads MLX_HOSTFILE as the path of a JSON hostfile: a per-rank list of
    # "ip:port" listen addresses.
    base_port = _free_port()
    hostfile = [f"127.0.0.1:{base_port + rank}" for rank in range(2)]

    scenarios: list[Scenario] = [
        {
            "name": "only_rank0_failed",
            "rank0_ok": False,
            "rank1_ok": True,
            "expected": True,
        },
        {
            "name": "only_rank1_failed",
            "rank0_ok": True,
            "rank1_ok": False,
            "expected": True,
        },
        {"name": "both_failed", "rank0_ok": False, "rank1_ok": False, "expected": True},
        {"name": "both_ok", "rank0_ok": True, "rank1_ok": True, "expected": False},
    ]
    scenarios_json = json.dumps(scenarios)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as hostfile_file:
        json.dump(hostfile, hostfile_file)
        hostfile_path = hostfile_file.name

    base_env = os.environ.copy()
    base_env.pop("MLX_HOSTFILE", None)
    base_env.pop("MLX_RANK", None)

    processes: list[subprocess.Popen[str]] = []
    try:
        for rank in range(2):
            env = base_env.copy()
            env["MLX_HOSTFILE"] = hostfile_path
            env["MLX_RANK"] = str(rank)
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(_HELPER), "--scenarios", scenarios_json],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )

        init_failed = False
        rank_messages: list[ObservedRanks] = []
        for process in processes:
            try:
                stdout, _stderr = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                pytest.fail("ring-backend agreement did not finish within 60 s")
            if process.returncode not in (0, 2):  # 2 = the ring backend did not start
                pytest.fail(
                    f"rank process failed (rc={process.returncode}): {stdout.strip()}"
                )
            for line in stdout.splitlines():
                if not line.strip():
                    continue
                payload = cast(dict[str, object], json.loads(line))
                if "init_failed" in payload:
                    init_failed = True
                elif "observed" in payload:
                    rank_messages.append(cast(ObservedRanks, cast(object, payload)))

        if init_failed:
            pytest.skip("MLX ring backend could not initialise on localhost")

        assert len(rank_messages) == 2, (
            f"expected two rank results, got {len(rank_messages)}"
        )

        observed_by_rank = {msg["rank"]: msg["observed"] for msg in rank_messages}
        assert set(observed_by_rank) == {0, 1}

        for scenario in scenarios:
            name = scenario["name"]
            rank0, rank1 = observed_by_rank[0][name], observed_by_rank[1][name]
            assert rank0 == rank1, (
                f"ranks disagree on '{name}': rank0={rank0}, rank1={rank1}"
            )
            assert rank0 == scenario["expected"], (
                f"both ranks got wrong decision for '{name}': {rank0}"
            )
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
        os.unlink(hostfile_path)
