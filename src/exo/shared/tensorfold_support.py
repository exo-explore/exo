from collections.abc import Mapping, Sequence

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.tensorfold import NativeCapabilities
from exo.shared.types.worker.instances import Instance, InstanceId, TensorFoldInstance


def native_backend(
    card: ModelCard,
    capabilities: NativeCapabilities | None,
    backends: Sequence[Backend],
) -> Backend | None:
    if capabilities is None or not capabilities.chip:
        return None
    try:
        capabilities.require_streaming()
    except ValueError:
        return None
    if card.tasks != [ModelTask.TextGeneration] or card.model_type is None:
        return None
    if card.tensorfold_format not in capabilities.families.get(card.model_type, []):
        return None
    for native, backend, chip_prefix in (
        ("metal", Backend.TensorFoldMetal, "apple-"),
        ("cuda", Backend.TensorFoldCuda, "nvidia-"),
    ):
        if (
            native in capabilities.backends
            and backend in backends
            and backend in card.backends
            and capabilities.chip.startswith(chip_prefix)
        ):
            return backend
    return None


def runtime_reserve(card: ModelCard) -> int:
    return max(1024**3, card.resident_size.in_bytes // 10)


def available_bytes(usage: MemoryUsage, backend: Backend) -> int:
    if backend == Backend.TensorFoldCuda:
        return usage.gpu_available.in_bytes if usage.gpu_available is not None else 0
    if usage.gpu_working_set is None:
        return 0
    return min(usage.ram_available.in_bytes, usage.gpu_working_set.in_bytes)


def prefill_link_refusal(
    instance_ids: Sequence[InstanceId], instances: Mapping[InstanceId, Instance]
) -> str | None:
    if any(isinstance(instances.get(key), TensorFoldInstance) for key in instance_ids):
        return "TensorFold does not support remote prefill or KV-cache transfer"
    return None
