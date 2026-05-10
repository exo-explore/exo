"""Tests for the drafter-task-routing gate in :mod:`exo.worker.main`.

Codex flagged (P1, PR #20 round 2) that
``Worker._start_runner_task`` resolved the runner via
``instance.all_node_to_runner[self.node_id]`` for *every* task,
which on the drafter node routed ``TextGeneration`` /
``ImageGeneration`` / ``ImageEdits`` to the drafter runner. The
drafter runner only accepts lifecycle tasks (``ConnectToGroup``,
``LoadModel``, ``StartWarmup``, ``Shutdown``) and raises
``ValueError`` for anything else, marking the runner failed and
cascading into instance shutdown during asymmetric serving.

These tests cover :func:`_should_drop_generation_task_at_drafter`
which gates the routing.

Tasks are constructed via ``model_construct`` so we don't have to
populate every required pydantic field; only the *type* of the task
matters for the routing gate (``isinstance`` check).
"""

from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.tasks import (
    ConnectToGroup,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.instances import DrafterPlacement, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.worker.main import (
    _should_drop_generation_task_at_drafter,  # pyright: ignore[reportPrivateUsage]
)
from exo.worker.runner.supervisor import RunnerSupervisor

DRAFTER_NODE = NodeId()
TARGET_NODE = NodeId()
DRAFTER_RUNNER = RunnerId()
TARGET_RUNNER = RunnerId()
INSTANCE = InstanceId()


def _drafter_placement() -> DrafterPlacement:
    return DrafterPlacement(
        drafter_node_id=DRAFTER_NODE,
        drafter_runner_id=DRAFTER_RUNNER,
        drafter_model_id=ModelId("mlx-community/gemma-4-e2b-it-8bit"),
        drafter_rank=2,
        drafter_socket_host="169.254.0.10",
        drafter_socket_port=60001,
    )


def _text_gen() -> TextGeneration:
    return TextGeneration.model_construct(
        task_id=TaskId(),
        instance_id=INSTANCE,
        command_id=CommandId(),
        task_status=TaskStatus.Pending,
    )


def _image_gen() -> ImageGeneration:
    return ImageGeneration.model_construct(
        task_id=TaskId(),
        instance_id=INSTANCE,
        command_id=CommandId(),
        task_status=TaskStatus.Pending,
    )


def _image_edits() -> ImageEdits:
    return ImageEdits.model_construct(
        task_id=TaskId(),
        instance_id=INSTANCE,
        command_id=CommandId(),
        task_status=TaskStatus.Pending,
    )


def _connect() -> ConnectToGroup:
    return ConnectToGroup.model_construct(
        task_id=TaskId(),
        instance_id=INSTANCE,
        task_status=TaskStatus.Pending,
    )


def _load_model() -> LoadModel:
    return LoadModel.model_construct(
        task_id=TaskId(),
        instance_id=INSTANCE,
        task_status=TaskStatus.Pending,
    )


def _start_warmup() -> StartWarmup:
    return StartWarmup.model_construct(
        task_id=TaskId(),
        instance_id=INSTANCE,
        task_status=TaskStatus.Pending,
    )


def _shutdown() -> Shutdown:
    return Shutdown.model_construct(
        task_id=TaskId(),
        instance_id=INSTANCE,
        task_status=TaskStatus.Pending,
        runner_id=DRAFTER_RUNNER,
    )


def test_drops_text_generation_at_drafter_node() -> None:
    """TextGeneration on the drafter node routed to the drafter runner
    must be dropped -- DrafterRunner._dispatch raises ValueError."""
    assert _should_drop_generation_task_at_drafter(
        task=_text_gen(),
        runner_id=DRAFTER_RUNNER,
        drafter_placement=_drafter_placement(),
        node_id=DRAFTER_NODE,
    )


def test_drops_image_generation_at_drafter_node() -> None:
    assert _should_drop_generation_task_at_drafter(
        task=_image_gen(),
        runner_id=DRAFTER_RUNNER,
        drafter_placement=_drafter_placement(),
        node_id=DRAFTER_NODE,
    )


def test_drops_image_edits_at_drafter_node() -> None:
    assert _should_drop_generation_task_at_drafter(
        task=_image_edits(),
        runner_id=DRAFTER_RUNNER,
        drafter_placement=_drafter_placement(),
        node_id=DRAFTER_NODE,
    )


def test_does_not_drop_lifecycle_tasks_at_drafter() -> None:
    """ConnectToGroup, LoadModel, StartWarmup, Shutdown must reach
    the drafter runner -- they're the only tasks DrafterRunner
    accepts."""
    placement = _drafter_placement()
    for task in (_connect(), _load_model(), _start_warmup(), _shutdown()):
        assert not _should_drop_generation_task_at_drafter(
            task=task,
            runner_id=DRAFTER_RUNNER,
            drafter_placement=placement,
            node_id=DRAFTER_NODE,
        ), f"{task.__class__.__name__} should reach drafter runner"


def test_does_not_drop_text_generation_at_target_node() -> None:
    """On the target node, TextGeneration routes to the target runner,
    not the drafter, so the gate must NOT fire."""
    assert not _should_drop_generation_task_at_drafter(
        task=_text_gen(),
        runner_id=TARGET_RUNNER,
        drafter_placement=_drafter_placement(),
        node_id=TARGET_NODE,
    )


def test_does_not_drop_when_no_drafter_placement() -> None:
    """Symmetric placement (no drafter) -- gate is a no-op."""
    assert not _should_drop_generation_task_at_drafter(
        task=_text_gen(),
        runner_id=TARGET_RUNNER,
        drafter_placement=None,
        node_id=TARGET_NODE,
    )


def test_does_not_drop_when_runner_id_does_not_match_drafter() -> None:
    """If the resolved runner is NOT the drafter runner, the task is
    target-bound and must not be dropped (defends against future
    refactors that change ``all_node_to_runner`` semantics)."""
    assert not _should_drop_generation_task_at_drafter(
        task=_text_gen(),
        runner_id=TARGET_RUNNER,  # not the drafter runner
        drafter_placement=_drafter_placement(),
        node_id=DRAFTER_NODE,  # drafter node, but target runner
    )


def test_does_not_drop_when_node_id_is_not_drafter_node() -> None:
    """If self.node_id isn't the drafter node, the task is target-
    bound on this worker. Belt-and-suspenders against
    ``all_node_to_runner`` returning the drafter runner from a
    non-drafter node (which shouldn't happen, but the gate is
    defensive)."""
    assert not _should_drop_generation_task_at_drafter(
        task=_text_gen(),
        runner_id=DRAFTER_RUNNER,  # would-be drafter runner...
        drafter_placement=_drafter_placement(),
        node_id=TARGET_NODE,  # ...but on a target node
    )


def test_mark_task_dropped_locally_records_completion_without_dispatch() -> None:
    """``RunnerSupervisor.mark_task_dropped_locally`` is the hook that
    short-circuits planner re-selection when a task reached this node
    but the runner cannot accept it. The contract is:

    * ``in_progress`` no longer contains the id (so the runner won't
      try to ack it later).
    * ``completed`` does contain the id so that ``plan.py`` skips it
      on the next 100ms tick.

    Codex P2 (PR #20) flagged that without this hook, the planner
    re-selects dropped generation tasks on every tick, re-emitting
    ``TaskCreated`` and re-running the drop branch for the lifetime
    of the request.
    """
    task = _text_gen()
    supervisor = RunnerSupervisor.__new__(RunnerSupervisor)
    supervisor.in_progress = {task.task_id: task}
    supervisor.completed = set()
    supervisor.mark_task_dropped_locally(task.task_id)
    assert task.task_id in supervisor.completed
    assert task.task_id not in supervisor.in_progress
    # Idempotent: a duplicate call must not raise or duplicate state.
    supervisor.mark_task_dropped_locally(task.task_id)
    assert task.task_id in supervisor.completed
    assert task.task_id not in supervisor.in_progress
