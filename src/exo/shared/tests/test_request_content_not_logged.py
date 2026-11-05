"""The text of a request (prompt, tool schemas, tool arguments, reply) never reaches a log.

A request logged as a whole, as a task, a command or an event, shows how long its text was and
not what it said, validation errors don't quote their input, and no log sink prints the
variables of a logged exception. `EXO_LOG_REQUEST_CONTENT=true` brings the text back for
debugging; every test here that asserts absence has its counterpart with the flag on.
"""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import zstandard
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from exo.api.types import ImageEditsTaskParams, ImageGenerationTaskParams, ToolCallItem
from exo.routing.topics import GLOBAL_EVENTS
from exo.shared import logging as exo_logging
from exo.shared.request_content import (
    LOG_REQUEST_CONTENT_ENV,
    describe_text,
    logging_request_content,
)
from exo.shared.types.chunks import TokenChunk, ToolCallChunk
from exo.shared.types.commands import TextGeneration as TextGenerationCommand
from exo.shared.types.common import CommandId, ModelId, NodeId, SessionId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    GlobalForwarderEvent,
    TaskCreated,
)
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    Base64Image,
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId

SENTINEL = "ZQSENTINEL7f3a9c1e"


def _params() -> TextGenerationTaskParams:
    """A request with the sentinel in every field that carries the caller's text."""
    return TextGenerationTaskParams.model_validate(
        {
            "model": "test/model",
            "input": [
                {"role": "user", "content": f"user text {SENTINEL}"},
                {"role": "assistant", "content": f"assistant text {SENTINEL}"},
            ],
            "instructions": f"system text {SENTINEL}",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": f"schema text {SENTINEL}",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "stop": [f"stop {SENTINEL}"],
            "chat_template_messages": [
                {"role": "system", "content": f"system text {SENTINEL}"},
                {"role": "user", "content": f"user text {SENTINEL}"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": json.dumps({"ref": SENTINEL}),
                            },
                        }
                    ],
                },
                {"role": "tool", "content": f"tool result {SENTINEL}"},
            ],
            "images": [f"aW1hZ2U{SENTINEL}"],
        }
    )


def _task() -> TextGeneration:
    return TextGeneration(
        task_id=TaskId("task"),
        command_id=CommandId("command"),
        instance_id=InstanceId("instance"),
        task_params=_params(),
    )


def _logged_forms(request: TextGenerationTaskParams) -> dict[str, str]:
    """Every form in which the code base puts a request or its reply into a log line: repr and
    str of the params, of the task, of the command and of the events that carry them (what
    `f"{task}"`, `str(event)[:100]` and `{command}` print), and the events as they come back
    from the wire."""
    task = TextGeneration(
        task_id=TaskId("task"),
        command_id=CommandId("command"),
        instance_id=InstanceId("instance"),
        task_params=request,
    )
    command = TextGenerationCommand(task_params=request)
    created = TaskCreated(task_id=task.task_id, task=task)
    token = ChunkGenerated(
        command_id=CommandId("command"),
        chunk=TokenChunk(
            model=ModelId("test/model"),
            text=f"reply {SENTINEL}",
            token_id=1,
            usage=None,
        ),
    )
    call = ChunkGenerated(
        command_id=CommandId("command"),
        chunk=ToolCallChunk(
            model=ModelId("test/model"),
            tool_calls=[
                ToolCallItem(name="lookup", arguments=json.dumps({"ref": SENTINEL}))
            ],
            usage=None,
        ),
    )
    adapter: TypeAdapter[Event] = TypeAdapter(Event)
    forms: dict[str, object] = {
        "params": request,
        "task": task,
        "command": command,
        "TaskCreated": created,
        "token chunk": token,
        "tool call chunk": call,
        "TaskCreated off the wire": adapter.validate_json(created.model_dump_json()),
        "token chunk off the wire": adapter.validate_json(token.model_dump_json()),
        "tool call chunk off the wire": adapter.validate_json(call.model_dump_json()),
    }
    out: dict[str, str] = {}
    for name, value in forms.items():
        out[f"repr of {name}"] = repr(value)
        out[f"str of {name}"] = str(value)
        out[f"f-string of {name}"] = f"{value}"
        out[f"truncated str of {name}"] = str(value)[:100]
    return out


def test_a_request_logged_as_a_whole_shows_lengths_and_not_text() -> None:
    forms = _logged_forms(_params())
    leaks = [name for name, text in forms.items() if SENTINEL in text]
    assert leaks == []
    # Still useful in a log: it says what kind of request it was and how big
    assert "chars" in forms["str of task"]
    assert "test/model" in forms["str of task"]


def test_the_wire_format_still_carries_the_text() -> None:
    # Hiding the text from repr must not touch what is sent between nodes or stored in state
    dumped = _task().model_dump_json()
    assert SENTINEL in dumped
    restored = TextGeneration.model_validate_json(dumped)
    assert restored == _task()
    assert restored.task_params.input[0].content.endswith(SENTINEL)
    assert restored.task_params.tools is not None
    assert SENTINEL in json.dumps(restored.task_params.tools)


def test_image_requests_show_neither_prompt_nor_negative_prompt() -> None:
    generation = ImageGenerationTaskParams.model_validate(
        {
            "prompt": f"a picture of {SENTINEL}",
            "model": "test/image",
            "advanced_params": {"negative_prompt": f"not {SENTINEL}"},
        }
    )
    edit = ImageEditsTaskParams(
        prompt=f"edit {SENTINEL}", model="test/image", image_data=f"data{SENTINEL}"
    )
    for item in (generation, edit):
        assert SENTINEL not in repr(item)
        assert SENTINEL not in str(item)
        assert SENTINEL not in str(item)[:500]


def test_the_flag_brings_the_text_back(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not logging_request_content()
    monkeypatch.setenv(LOG_REQUEST_CONTENT_ENV, "true")
    assert logging_request_content()
    # The start of each message, as before
    forms = _logged_forms(
        TextGenerationTaskParams(
            model=ModelId("test/model"),
            input=[
                InputMessage(
                    role="user", content=InputMessageContent(f"hello {SENTINEL}")
                )
            ],
        )
    )
    assert SENTINEL in forms["str of task"]
    assert describe_text(f"hello {SENTINEL}") == repr(f"hello {SENTINEL}")
    # Long text stays cut at the old length
    long = InputMessageContent("x" * 500 + SENTINEL)
    assert SENTINEL not in repr(long)
    assert repr(long).count("x") == InputMessageContent.truncate_length


def test_describe_text_gives_a_length_by_default() -> None:
    assert describe_text(f"hello {SENTINEL}") == f"<{len(f'hello {SENTINEL}')} chars>"
    assert repr(Base64Image(SENTINEL)) == f"<Base64Image: {len(SENTINEL)} chars>"


def test_a_validation_error_does_not_quote_the_request() -> None:
    bad_role = {"role": f"bogus {SENTINEL}", "content": f"user text {SENTINEL}"}
    with pytest.raises(ValidationError) as error:
        TextGenerationTaskParams.model_validate(
            {"model": "test/model", "input": [bad_role]}
        )
    assert SENTINEL not in str(error.value)
    assert "input" in str(error.value)  # it still says which field was wrong

    with pytest.raises(ValidationError) as error:
        TextGenerationTaskParams.model_validate(
            {
                "model": "test/model",
                "input": [{"role": "user", "content": f"user text {SENTINEL}"}],
                "temperature": f"hot {SENTINEL}",
            }
        )
    assert SENTINEL not in str(error.value)

    # An event that a node of another version sent (a wrong field, the request inside), read the
    # way the router reads what arrives on a topic
    created = TaskCreated(task_id=TaskId("task"), task=_task())
    wire = cast(
        dict[str, Any],
        json.loads(
            GlobalForwarderEvent(
                origin_idx=0,
                origin=NodeId("node"),
                session=SessionId(master_node_id=NodeId("node"), election_clock=0),
                event=created,
            ).model_dump_json()
        ),
    )
    wire["event"]["TaskCreated"]["task"]["TextGeneration"]["task_params"][
        "temperature"
    ] = f"hot {SENTINEL}"
    with pytest.raises(ValidationError) as error:
        GLOBAL_EVENTS.deserialize(json.dumps(wire).encode())
    assert SENTINEL not in str(error.value)
    assert "temperature" in str(error.value)


@pytest.fixture
def logs(tmp_path: Path) -> Iterator[Path]:
    """exo's real log setup at -v (debug lines included: the most that gets written)"""
    log_file = tmp_path / "exo.log"
    exo_logging.logger_setup(log_file, verbosity=1)
    yield log_file
    logger.remove()
    logger.add(sys.stderr)


def _everything_logged(log_file: Path) -> str:
    exo_logging.logger_cleanup()
    text = log_file.read_text()
    for archive in log_file.parent.glob("exo.*.log.zst"):
        text += zstandard.ZstdDecompressor().decompress(archive.read_bytes()).decode()
    return text


def _fail_holding_the_request(request: TextGenerationTaskParams) -> None:
    prompt = f"rendered prompt {SENTINEL}"
    reply = f"partial reply {SENTINEL}"
    # loguru's diagnose prints the variables named on the first line of the raise
    raise RuntimeError("the engine failed" if prompt and reply and request else "")


def test_a_logged_exception_does_not_print_the_variables_of_its_frames(
    logs: Path,
) -> None:
    request = _params()
    try:
        _fail_holding_the_request(request)
    except RuntimeError as error:
        logger.opt(exception=error).warning("Runner crashed")
    logger.info(f"received chat request: {_task()}")
    text = _everything_logged(logs)
    assert "Runner crashed" in text
    assert "the engine failed" in text  # the traceback itself is still there
    assert SENTINEL not in text


def test_loguru_would_print_them_by_default() -> None:
    """The reason for diagnose=False: this is what loguru does without it."""
    lines: list[str] = []
    handler = logger.add(lines.append, format="{message}", level="INFO", diagnose=True)
    try:
        try:
            _fail_holding_the_request(_params())
        except RuntimeError as error:
            logger.opt(exception=error).warning("Runner crashed")
    finally:
        logger.remove(handler)
    assert SENTINEL in "".join(lines)


def test_a_log_file_written_at_every_level_has_no_text(logs: Path) -> None:
    for form in _logged_forms(_params()).values():
        logger.debug(form)
        logger.info(form)
        logger.warning(form)
    assert SENTINEL not in _everything_logged(logs)
