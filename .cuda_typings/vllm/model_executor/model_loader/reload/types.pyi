from _typeshed import Incomplete
from dataclasses import dataclass, field
from inspect import BoundArguments

__all__ = ["LayerTensors", "LayerReloadingInfo"]

LayerTensors: Incomplete

@dataclass
class LayerReloadingInfo:
    restore_metadata: LayerTensors = field(default_factory=Incomplete)
    kernel_tensors: LayerTensors = field(default_factory=Incomplete)
    load_numel: int = ...
    load_numel_total: int | None = ...
    loaded_weights: list[tuple[str, BoundArguments]] = field(default_factory=list)
    def reset(self) -> None: ...
    def can_process(self) -> bool: ...
