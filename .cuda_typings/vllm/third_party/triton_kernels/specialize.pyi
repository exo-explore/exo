import triton
from _typeshed import Incomplete
from dataclasses import dataclass

def cacheable(f): ...
def define_kernel(src, module, attrs=None, **extra_globals): ...
@dataclass(frozen=True)
class FnSpecs:
    name: str
    fn: triton.runtime.jit.JITFunction
    fn_arg_names: tuple[str]
    fn_arg_do_not_specialize: tuple[str] = ...
    reduction_n: int = ...
    @staticmethod
    def default(): ...

def specialize(fn, module, constants, tuples, name=None, do_not_specialize=...): ...
@dataclass(frozen=True)
class ClosureArg:
    fn_name: str
    fn_params_name: str

class SpecializationModule:
    module_name: Incomplete
    kernels: Incomplete
    closure_args: Incomplete
    def __init__(
        self,
        module_name: str,
        kernels: list[tuple[str, object]],
        closure_args: dict[str, ClosureArg],
    ) -> None: ...
    def get(self, **kwargs): ...
