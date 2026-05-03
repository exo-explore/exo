from exo.utils.keyed_backoff import KeyedBackoff


def test_tracked_keys_reports_and_resets_backoff_state() -> None:
    backoff = KeyedBackoff[str]()

    backoff.record_attempt("instance-a")

    assert backoff.tracked_keys() == {"instance-a"}

    backoff.reset("instance-a")

    assert backoff.tracked_keys() == set()
