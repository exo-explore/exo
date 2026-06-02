from pydantic import ValidationError

from exo.master.placement_utils import find_ip_prioritised
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import ImageEdits, ImageGeneration, TextGeneration
from exo.shared.types.instance_link import InstanceLink
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import (
    Task,
    TaskId,
    TaskStatus,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.worker.instances import InstanceId


def decode_instance_for_text_generation(
    state: State, model_id: ModelId, instance_links: list[InstanceLink]
) -> InstanceId:
    prefill_only: set[InstanceId] = set()
    for link in instance_links:
        prefill_only.update(link.prefill_instances)
    for link in instance_links:
        prefill_only.difference_update(link.decode_instances)

    instance_task_counts = _instance_task_counts_for_model(state, model_id)
    for instance_id in prefill_only:
        instance_task_counts.pop(instance_id, None)

    if not instance_task_counts:
        raise ValueError(f"No instance found for model {model_id}")

    return min(
        instance_task_counts, key=lambda instance_id: instance_task_counts[instance_id]
    )


def instance_for_generation(state: State, model_id: ModelId) -> InstanceId:
    instance_task_counts = _instance_task_counts_for_model(state, model_id)
    if not instance_task_counts:
        raise ValueError(f"No instance found for model {model_id}")

    return min(
        instance_task_counts, key=lambda instance_id: instance_task_counts[instance_id]
    )


def text_generation_task(
    state: State, command: TextGeneration, instance_links: list[InstanceLink]
) -> TextGenerationTask:
    instance_id = decode_instance_for_text_generation(
        state, command.task_params.model, instance_links
    )
    task_params = command.task_params.model_copy(
        update={
            "prefill_endpoint": prefill_endpoint_for(
                state,
                instance_links,
                instance_id,
            ),
        }
    )
    return TextGenerationTask(
        task_id=TaskId(),
        command_id=command.command_id,
        instance_id=instance_id,
        task_status=TaskStatus.Pending,
        task_params=task_params,
    )


def image_generation_task(
    state: State, command: ImageGeneration
) -> ImageGenerationTask:
    return ImageGenerationTask(
        task_id=TaskId(),
        command_id=command.command_id,
        instance_id=instance_for_generation(state, ModelId(command.task_params.model)),
        task_status=TaskStatus.Pending,
        task_params=command.task_params,
    )


def image_edits_task(state: State, command: ImageEdits) -> ImageEditsTask:
    return ImageEditsTask(
        task_id=TaskId(),
        command_id=command.command_id,
        instance_id=instance_for_generation(state, ModelId(command.task_params.model)),
        task_status=TaskStatus.Pending,
        task_params=command.task_params,
    )


def task_from_command(
    state: State,
    command: TextGeneration | ImageGeneration | ImageEdits,
    instance_links: list[InstanceLink],
) -> Task:
    match command:
        case TextGeneration():
            return text_generation_task(state, command, instance_links)
        case ImageGeneration():
            return image_generation_task(state, command)
        case ImageEdits():
            return image_edits_task(state, command)


def instance_id_for_command(
    state: State,
    command: TextGeneration | ImageGeneration | ImageEdits,
    instance_links: list[InstanceLink],
) -> InstanceId:
    match command:
        case TextGeneration():
            return decode_instance_for_text_generation(
                state, command.task_params.model, instance_links
            )
        case ImageGeneration() | ImageEdits():
            return instance_for_generation(state, ModelId(command.task_params.model))


def load_instance_links(values: list[str]) -> list[InstanceLink]:
    instance_links: list[InstanceLink] = []
    for value in values:
        try:
            instance_links.append(InstanceLink.model_validate_json(value))
        except ValidationError:
            continue
    return instance_links


def prefill_endpoint_for(
    state: State, instance_links: list[InstanceLink], decode_instance_id: InstanceId
) -> str | None:
    decode = state.instances.get(decode_instance_id)
    if decode is None:
        return None
    decode_node = decode.shard_assignments.shards[
        decode.shard_assignments.primary_output_node
    ].node_id

    sources: set[InstanceId] = set()
    for link in instance_links:
        if decode_instance_id in link.decode_instances:
            sources.update(link.prefill_instances)
    sources.discard(decode_instance_id)

    in_flight = {TaskStatus.Pending, TaskStatus.Running}
    task_counts: dict[InstanceId, int] = {
        src_id: sum(
            1
            for task in state.tasks.values()
            if task.instance_id == src_id and task.task_status in in_flight
        )
        for src_id in sources
    }
    for src_id in sorted(sources, key=lambda sid: task_counts[sid]):
        instance = state.instances.get(src_id)
        if instance is None:
            continue
        for node_id, runner_id, _ in instance.shard_assignments.shards:
            port = state.prefill_server_ports.get(runner_id)
            if port is None:
                continue
            ip = find_ip_prioritised(
                decode_node, node_id, state.topology, state.node_network, ring=True
            )
            if ip is None:
                continue
            return f"{ip}:{port}"
    return None


def _instance_task_counts_for_model(
    state: State, model_id: ModelId
) -> dict[InstanceId, int]:
    in_flight = {TaskStatus.Pending, TaskStatus.Running}
    return {
        instance.instance_id: sum(
            1
            for task in state.tasks.values()
            if task.instance_id == instance.instance_id
            and task.task_status in in_flight
        )
        for instance in state.instances.values()
        if instance.shard_assignments.model_id == model_id
    }
