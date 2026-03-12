from .target_info import is_cuda as is_cuda, is_hip as is_hip
from dataclasses import dataclass

@dataclass
class PerfRecord:
    time_ns: float
    flops: float
    bytes: float

def parse_profile(profile_path, useful_op_regex): ...
def write_csv(xs, perfs, fpath): ...
def compute_roofline(
    *args,
    bench_fn,
    intensity_proxy_name,
    intensity_proxy_values,
    out_path,
    verbose,
    **kwargs,
): ...
def get_memset_tbps(): ...
def get_cublas_tflops(dtype): ...
def load_perf_csv(path): ...
def validate_perfs(perfs) -> None: ...
def plot_roofline(
    series,
    flops_dtype,
    out_path,
    max_tbps: str = "memset",
    max_tflops: str = "cublas",
    title: str = "",
    xlabel: str = "",
    labels=None,
    points_of_interest=None,
): ...
