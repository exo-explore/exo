import torch
from collections.abc import Callable as Callable, Iterable
from typing import TypeAlias, overload
from vllm.multimodal.inputs import BatchedTensorInputs as BatchedTensorInputs

JSONTree: TypeAlias

def json_iter_leaves(value: JSONTree[_T]) -> Iterable[_T]: ...
@overload
def json_map_leaves(
    func: Callable[[torch.Tensor], "torch.Tensor"], value: BatchedTensorInputs
) -> BatchedTensorInputs: ...
@overload
def json_map_leaves(
    func: Callable[[_T], _U], value: _T | dict[str, _T]
) -> _U | dict[str, _U]: ...
@overload
def json_map_leaves(
    func: Callable[[_T], _U], value: _T | list[_T]
) -> _U | list[_U]: ...
@overload
def json_map_leaves(
    func: Callable[[_T], _U], value: _T | tuple[_T, ...]
) -> _U | tuple[_U, ...]: ...
@overload
def json_map_leaves(func: Callable[[_T], _U], value: JSONTree[_T]) -> JSONTree[_U]: ...
@overload
def json_reduce_leaves(
    func: Callable[[_T, _T], _T], value: _T | dict[str, _T], /
) -> _T: ...
@overload
def json_reduce_leaves(func: Callable[[_T, _T], _T], value: _T | list[_T], /) -> _T: ...
@overload
def json_reduce_leaves(
    func: Callable[[_T, _T], _T], value: _T | tuple[_T, ...], /
) -> _T: ...
@overload
def json_reduce_leaves(func: Callable[[_T, _T], _T], value: JSONTree[_T], /) -> _T: ...
@overload
def json_reduce_leaves(
    func: Callable[[_U, _T], _U], value: JSONTree[_T], initial: _U, /
) -> _U: ...
def json_count_leaves(value: JSONTree[_T]) -> int: ...
