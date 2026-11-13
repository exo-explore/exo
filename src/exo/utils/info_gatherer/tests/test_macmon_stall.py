import json
import os
import sys
from pathlib import Path

import anyio
import pytest

from exo.utils.channels import channel
from exo.utils.info_gatherer import info_gatherer
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer

_MACMON_LINE = json.dumps(
    {
        "timestamp": "2026-01-01T00:00:00Z",
        "temp": {"cpu_temp_avg": 40.0, "gpu_temp_avg": 35.0},
        "memory": {
            "ram_total": 64 * 1024**3,
            "ram_usage": 16 * 1024**3,
            "swap_total": 0,
            "swap_usage": 0,
        },
        "ecpu_usage": [1000, 0.1],
        "pcpu_usage": [3000, 0.2],
        "gpu_usage": [1000, 0.0],
        "all_power": 10.0,
        "ane_power": 0.0,
        "cpu_power": 5.0,
        "gpu_power": 1.0,
        "gpu_ram_power": 0.0,
        "ram_power": 0.5,
        "sys_power": 12.0,
    }
)

# Answers the preflight, then in pipe mode records its pid, writes one reading and stalls
_FAKE_MACMON = """
import os, sys, time
line = {line!r}
if "--samples" in sys.argv:
    print(line)
    sys.exit(0)
with open({starts!r}, "a") as starts:
    starts.write(f"{{os.getpid()}}\\n")
print(line, flush=True)
time.sleep(1000)
"""


def _started_pids(starts: Path) -> list[int]:
    if not starts.exists():
        return []
    return [int(pid) for pid in starts.read_text().split()]


async def test_a_stalled_macmon_is_killed_and_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts = tmp_path / "starts"
    script = tmp_path / "fake_macmon.py"
    script.write_text(_FAKE_MACMON.format(line=_MACMON_LINE, starts=str(starts)))
    macmon = tmp_path / "macmon"
    macmon.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    macmon.chmod(0o755)
    monkeypatch.setenv("EXO_MACMON_PATH", str(macmon))
    monkeypatch.setattr(info_gatherer, "MACMON_MIN_READ_TIMEOUT", 0.5)

    info_sender, info_receiver = channel[GatheredInfo]()
    gatherer = InfoGatherer(info_sender)
    with info_sender, info_receiver:
        async with gatherer._tg as task_group:  # pyright: ignore[reportPrivateUsage]
            task_group.start_soon(gatherer._monitor_macmon, 0.05)  # pyright: ignore[reportPrivateUsage]
            with anyio.fail_after(10):
                while len(_started_pids(starts)) < 2:
                    await anyio.sleep(0.05)
            task_group.cancel_scope.cancel()

    stalled_pid = _started_pids(starts)[0]
    with pytest.raises(ProcessLookupError):
        os.kill(stalled_pid, 0)
    assert gatherer._psutil_enabled  # pyright: ignore[reportPrivateUsage]
