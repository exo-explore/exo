from collections.abc import Iterable, Mapping
from torch.nn import Module as Module

def is_activation_quantization_format(format: str) -> bool: ...
def should_ignore_layer(
    layer_name: str | None,
    ignore: Iterable[str] = ...,
    fused_mapping: Mapping[str, list[str]] = ...,
) -> bool: ...
def check_equal_or_regex_match(layer_name: str, targets: Iterable[str]) -> bool: ...
def find_matched_target(
    layer_name: str | None,
    module: Module,
    targets: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = ...,
) -> str: ...
