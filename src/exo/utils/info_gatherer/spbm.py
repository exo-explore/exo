import re
import shutil

import anyio
from anyio import fail_after
from loguru import logger

from exo.shared.types.profiling import SystemPerformanceProfile
from exo.utils.pydantic_ext import TaggedModel

_SPBM_CHIP = "spbm-acpi-0"

_FIELD_RE = re.compile(r"^\s+(\w+):\s+(-?\d+(?:\.\d+)?)\s*$")


class SpbmMetrics(TaggedModel):
    system_profile: SystemPerformanceProfile


async def has_spbm() -> bool:
    if shutil.which("sensors") is None:
        return False
    try:
        with fail_after(5):
            proc = await anyio.run_process(["sensors", "-u", _SPBM_CHIP], check=False)
    except (TimeoutError, OSError):
        return False
    return proc.returncode == 0 and b"power1_input" in proc.stdout


def _parse_sensors_output(output: str) -> dict[str, dict[str, float]]:
    """Parse `sensors -u` output, keyed by category then channel label.

    Returns {"power": {label: watts}, "temp": {label: celsius},
             "energy": {label: joules}}. Multiple sections may share a label
     (e.g. "gpu" appears under power, temp, and energy); categorising by the
     numeric field's prefix disambiguates them.
    """
    result: dict[str, dict[str, float]] = {"power": {}, "temp": {}, "energy": {}}
    current_label: str | None = None
    for line in output.splitlines():
        if not line:
            continue
        if line[0] not in " \t":
            stripped = line.rstrip()
            current_label = stripped[:-1] if stripped.endswith(":") else None
            continue
        match = _FIELD_RE.match(line)
        if match is None or current_label is None:
            continue
        field = match.group(1)
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if not field.endswith("_input"):
            continue
        for category in result:
            if field.startswith(category):
                result[category][current_label] = value
                break
    return result


async def gather_spbm_metrics() -> SpbmMetrics | None:
    try:
        with fail_after(5):
            proc = await anyio.run_process(["sensors", "-u", _SPBM_CHIP], check=False)
    except (TimeoutError, OSError) as e:
        logger.opt(exception=e).debug("spbm sensors invocation failed")
        return None
    if proc.returncode != 0:
        return None

    parsed = _parse_sensors_output(proc.stdout.decode("utf-8", errors="replace"))
    sys_total = parsed["power"].get("sys_total")
    if sys_total is None:
        return None
    tj_max = parsed["temp"].get("tj_max")
    gpu_temp = parsed["temp"].get("gpu")

    temp = tj_max if tj_max is not None else (gpu_temp if gpu_temp is not None else 0.0)

    return SpbmMetrics(
        system_profile=SystemPerformanceProfile(
            temp=temp,
            sys_power=sys_total,
        ),
    )
