"""A request's text stays out of the logs of the worker's side of serving it: the supervisor that
sends it to a runner, and the runner that serves it, on the success path, a refused duplicate, a
runner that has gone away, and a cancellation."""

import sys
from collections.abc import Iterator
from pathlib import Path

import anyio
import pytest
import zstandard
from loguru import logger

from exo.shared import logging as exo_logging
from exo.shared.types.common import CommandId
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.text_generation import (
    TextGenerationTaskParams,
)
from exo.worker.tests.unittests.test_runner.test_event_ordering import (  # noqa: F401
    INIT_TASK,
    LOAD_TASK,
    SHUTDOWN_TASK,
    WARMUP_TASK,
    _run,  # pyright: ignore[reportPrivateUsage]
    patch_out_mlx,  # pyright: ignore[reportUnusedImport]
)
from exo.worker.tests.unittests.test_runner.test_supervisor_acknowledgement import (
    BOUND,
    supervisor,
)

SENTINEL = "ZQSENTINEL4b81d0e7"


def _sentinel_task() -> TextGeneration:
    return TextGeneration(
        instance_id=BOUND.instance.instance_id,
        command_id=CommandId(),
        task_params=TextGenerationTaskParams.model_validate(
            {
                "model": str(BOUND.bound_shard.model_card.model_id),
                "input": [{"role": "user", "content": f"user text {SENTINEL}"}],
                "instructions": f"system text {SENTINEL}",
                "tools": [{"type": "function", "function": {"name": SENTINEL}}],
                "chat_template_messages": [
                    {"role": "user", "content": f"user text {SENTINEL}"}
                ],
                "stream": True,
                "max_output_tokens": 4,
            }
        ),
    )


@pytest.fixture
def logs(tmp_path: Path) -> Iterator[Path]:
    log_file = tmp_path / "exo.log"
    exo_logging.logger_setup(log_file, verbosity=1)  # -v: debug lines too
    yield log_file
    logger.remove()
    logger.add(sys.stderr)


def _everything_logged(log_file: Path) -> str:
    exo_logging.logger_cleanup()
    text = log_file.read_text()
    for archive in log_file.parent.glob("exo.*.log.zst"):
        text += zstandard.ZstdDecompressor().decompress(archive.read_bytes()).decode()
    return text


@pytest.mark.anyio
async def test_the_supervisor_does_not_log_the_request(logs: Path) -> None:
    async with supervisor() as (sup, _):
        task = _sentinel_task()
        with anyio.fail_after(2):
            await sup.start_task(task)  # "Starting task ..."
            await sup.start_task(task)  # "Skipping invalid task ... already submitted"
        sup.completed.add(task.task_id)
        sup.pending.pop(task.task_id, None)
        sup.in_progress.pop(task.task_id, None)
        with anyio.fail_after(2):
            await sup.start_task(task)  # "... already been completed"
            await sup.cancel_task(task.task_id)  # "Unable to cancel ..."

        gone = _sentinel_task()
        sup._task_sender.close()  # pyright: ignore[reportPrivateUsage]
        with anyio.fail_after(2):
            await sup.start_task(
                gone
            )  # "Task ... dropped, runner closed communication."
    text = _everything_logged(logs)
    assert "Starting task" in text
    assert "already" in text
    assert SENTINEL not in text


def test_the_runner_does_not_log_the_request(
    patch_out_mlx: pytest.MonkeyPatch,  # noqa: F811
    logs: Path,
) -> None:
    task = _sentinel_task().model_copy(update={"instance_id": INIT_TASK.instance_id})
    _run([INIT_TASK, LOAD_TASK, WARMUP_TASK, task], send_after_ready=[SHUTDOWN_TASK])
    text = _everything_logged(logs)
    assert "received chat request" in text
    assert "<InputMessageContent:" in text  # what it logs is the shape of the request
    assert SENTINEL not in text


def test_the_runner_logs_the_request_with_the_flag_on(
    patch_out_mlx: pytest.MonkeyPatch,  # noqa: F811
    logs: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_LOG_REQUEST_CONTENT", "true")
    task = _sentinel_task().model_copy(update={"instance_id": INIT_TASK.instance_id})
    _run([INIT_TASK, LOAD_TASK, WARMUP_TASK, task], send_after_ready=[SHUTDOWN_TASK])
    assert SENTINEL in _everything_logged(logs)
