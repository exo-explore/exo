import subprocess

import pytest

import exo.utils.info_gatherer.macmon as macmon
from exo.utils.info_gatherer.macmon import read_macmon_metrics_once


def _macmon_output() -> str:
    return """
{"timestamp":"2026-05-14T12:00:00Z","temp":{"cpu_temp_avg":45.0,"gpu_temp_avg":46.0},"memory":{"ram_total":1000,"ram_usage":275,"swap_total":500,"swap_usage":125},"ecpu_usage":[1000,1.0],"pcpu_usage":[2000,2.0],"gpu_usage":[1200,3.0],"all_power":1.0,"ane_power":2.0,"cpu_power":3.0,"gpu_power":4.0,"gpu_ram_power":5.0,"ram_power":6.0,"sys_power":7.0}
{"timestamp":"2026-05-14T12:00:01Z","temp":{"cpu_temp_avg":45.0,"gpu_temp_avg":46.0},"memory":{"ram_total":1000,"ram_usage":999,"swap_total":500,"swap_usage":125},"ecpu_usage":[1000,1.0],"pcpu_usage":[2000,2.0],"gpu_usage":[1200,3.0],"all_power":1.0,"ane_power":2.0,"cpu_power":3.0,"gpu_power":4.0,"gpu_ram_power":5.0,"ram_power":6.0,"sys_power":7.0}
"""


def test_read_macmon_metrics_once_uses_first_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        capture_output: bool,
        check: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert capture_output
        assert not check
        assert text
        assert timeout == 3
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=_macmon_output(),
            stderr="",
        )

    monkeypatch.setattr(macmon.subprocess, "run", fake_run)

    metrics = read_macmon_metrics_once("/usr/local/bin/macmon", timeout=3)

    assert metrics is not None
    assert metrics.memory.ram_available.in_bytes == 725
    assert calls == [
        ["/usr/local/bin/macmon", "pipe", "--samples", "1", "--interval", "100"]
    ]


def test_read_macmon_metrics_once_returns_none_for_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        args: list[str],
        *,
        capture_output: bool,
        check: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(macmon.subprocess, "run", fake_run)

    assert read_macmon_metrics_once("/usr/local/bin/macmon") is None


def test_read_macmon_metrics_once_returns_none_for_missing_binary() -> None:
    assert read_macmon_metrics_once("/does/not/exist/macmon") is None
