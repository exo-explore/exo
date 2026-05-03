from exo.utils.info_gatherer.macmon import MacmonMetrics


def test_macmon_metrics_preserves_ane_power() -> None:
    raw_json = """
    {
      "timestamp": "2026-01-01T00:00:00Z",
      "temp": {"cpu_temp_avg": 50.0, "gpu_temp_avg": 55.0},
      "memory": {
        "ram_total": 1000,
        "ram_usage": 400,
        "swap_total": 200,
        "swap_usage": 50
      },
      "ecpu_usage": [1200, 10.0],
      "pcpu_usage": [2400, 20.0],
      "gpu_usage": [800, 30.0],
      "all_power": 18.0,
      "ane_power": 2.5,
      "cpu_power": 4.0,
      "gpu_power": 6.0,
      "gpu_ram_power": 1.0,
      "ram_power": 2.0,
      "sys_power": 15.0
    }
    """

    metrics = MacmonMetrics.from_raw_json(raw_json)

    assert metrics.system_profile.sys_power == 15.0
    assert metrics.system_profile.ane_power == 2.5
