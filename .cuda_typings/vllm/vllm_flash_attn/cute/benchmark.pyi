def benchmark_forward(
    fn,
    *inputs,
    repeats: int = 10,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype=...,
    **kwinputs,
): ...
def benchmark_backward(
    fn,
    *inputs,
    grad=None,
    repeats: int = 10,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype=...,
    **kwinputs,
): ...
def benchmark_combined(
    fn,
    *inputs,
    grad=None,
    repeats: int = 10,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype=...,
    **kwinputs,
): ...
def benchmark_fwd_bwd(
    fn,
    *inputs,
    grad=None,
    repeats: int = 10,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype=...,
    **kwinputs,
): ...
def benchmark_all(
    fn,
    *inputs,
    grad=None,
    repeats: int = 10,
    desc: str = "",
    verbose: bool = True,
    amp: bool = False,
    amp_dtype=...,
    **kwinputs,
): ...
def pytorch_profiler(
    fn,
    *inputs,
    trace_filename=None,
    backward: bool = False,
    amp: bool = False,
    amp_dtype=...,
    cpu: bool = False,
    verbose: bool = True,
    **kwinputs,
) -> None: ...
def benchmark_memory(fn, *inputs, desc: str = "", verbose: bool = True, **kwinputs): ...
