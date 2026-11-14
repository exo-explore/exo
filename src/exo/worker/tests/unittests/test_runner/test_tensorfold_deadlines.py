import pytest

from exo.shared.models.model_cards import ModelId
from exo.worker.runner.supervisor import runner_load_timeout, runner_warmup_timeout
from exo.worker.tests.unittests.conftest import get_pipeline_shard_metadata


def test_native_load_and_warmup_use_the_configured_process_deadlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = get_pipeline_shard_metadata(ModelId("test/native"), 0).model_card
    monkeypatch.setenv("EXO_TENSORFOLD_STARTUP_TIMEOUT", "1200")
    monkeypatch.setenv("EXO_TENSORFOLD_REQUEST_TIMEOUT", "75")
    assert runner_load_timeout(card, tensorfold=True) == 1210
    assert runner_warmup_timeout(card, on_metal=False, tensorfold=True) == 85
    assert runner_warmup_timeout(card, on_metal=True, tensorfold=True) == 85
    assert runner_load_timeout(card) != 1210


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_native_timeout_is_not_an_unbounded_supervisor_deadline(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    card = get_pipeline_shard_metadata(ModelId("test/native"), 0).model_card
    monkeypatch.setenv("EXO_TENSORFOLD_STARTUP_TIMEOUT", value)
    with pytest.raises(ValueError):
        runner_load_timeout(card, tensorfold=True)
