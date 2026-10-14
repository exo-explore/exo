from exo.utils.info_gatherer.net_profile import latency_changed_materially


def test_first_measurement_is_material():
    assert latency_changed_materially(None, 0.5)


def test_small_absolute_jitter_is_ignored():
    # sub-noise-floor changes never republish, even when the factor is large
    assert not latency_changed_materially(0.4, 1.9)
    assert not latency_changed_materially(1.9, 0.4)


def test_small_relative_change_is_ignored():
    assert not latency_changed_materially(40.0, 55.0)
    assert not latency_changed_materially(55.0, 40.0)


def test_large_change_is_material():
    # e.g. traffic silently rerouted from thunderbolt to a WAN path
    assert latency_changed_materially(0.8, 45.0)
    assert latency_changed_materially(45.0, 0.8)
