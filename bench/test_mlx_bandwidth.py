# type: ignore
import argparse
import json
import os
import statistics
import sys
import tempfile
import time

import mlx.core as mx

DTYPE_MAP = {
    "float32": (mx.float32, 4),
    "float16": (mx.float16, 2),
    "bfloat16": (mx.bfloat16, 2),
}

SIZES = [
    1 * 1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1 * 1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
    1 * 1024 * 1024 * 1024,
    2 * 1024 * 1024 * 1024,
    4 * 1024 * 1024 * 1024,
    8 * 1024 * 1024 * 1024,
]


def format_bytes(n: int) -> str:
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.0f} GB"
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def format_time(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    if seconds >= 0.001:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds * 1_000_000:.1f} us"


def format_bandwidth(bytes_per_sec: float) -> str:
    if bytes_per_sec >= 1024 * 1024 * 1024:
        return f"{bytes_per_sec / (1024 * 1024 * 1024):.2f} GB/s"
    if bytes_per_sec >= 1024 * 1024:
        return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"
    return f"{bytes_per_sec / 1024:.1f} KB/s"


def barrier(group: mx.distributed.Group) -> None:
    mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group))


def init_ring(
    rank: int, self_ip: str, peer_ip: str, port: int, tmpdir: str
) -> mx.distributed.Group:
    if rank == 0:
        hosts = [f"{self_ip}:{port}", f"{peer_ip}:{port}"]
    else:
        hosts = [f"{peer_ip}:{port}", f"{self_ip}:{port}"]

    hostfile = os.path.join(tmpdir, "hosts.json")
    with open(hostfile, "w") as f:
        json.dump(hosts, f)

    for var in ("MLX_HOSTFILE", "MLX_RANK", "MLX_IBV_DEVICES", "MLX_JACCL_COORDINATOR"):
        os.environ.pop(var, None)

    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    return mx.distributed.init(backend="ring", strict=True)


def init_jaccl(
    rank: int, interface: str, coordinator: str, port: int, tmpdir: str
) -> mx.distributed.Group:
    devices = [[None, interface], [interface, None]]
    devfile = os.path.join(tmpdir, "devices.json")
    with open(devfile, "w") as f:
        json.dump(devices, f)

    for var in ("MLX_HOSTFILE", "MLX_RANK", "MLX_IBV_DEVICES", "MLX_JACCL_COORDINATOR"):
        os.environ.pop(var, None)

    os.environ["MLX_IBV_DEVICES"] = devfile
    os.environ["MLX_RANK"] = str(rank)
    if rank == 0:
        os.environ["MLX_JACCL_COORDINATOR"] = f"0.0.0.0:{port}"
    else:
        os.environ["MLX_JACCL_COORDINATOR"] = coordinator

    return mx.distributed.init(backend="jaccl", strict=True)


def bench_unidirectional(
    group: mx.distributed.Group,
    rank: int,
    size_bytes: int,
    dtype: mx.Dtype,
    element_size: int,
    warmup: int,
    iterations: int,
) -> list[float]:
    n_elements = size_bytes // element_size
    tensor = mx.random.normal(shape=(n_elements,)).astype(dtype)
    mx.eval(tensor)

    for _ in range(warmup):
        if rank == 0:
            sent = mx.distributed.send(tensor, dst=1, group=group)
            mx.eval(sent)
        else:
            received = mx.distributed.recv_like(tensor, src=0, group=group)
            mx.eval(received)
        barrier(group)

    times: list[float] = []
    for _ in range(iterations):
        barrier(group)
        t0 = time.perf_counter()
        if rank == 0:
            sent = mx.distributed.send(tensor, dst=1, group=group)
            mx.eval(sent)
        else:
            received = mx.distributed.recv_like(tensor, src=0, group=group)
            mx.eval(received)
        barrier(group)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def bench_rtt(
    group: mx.distributed.Group,
    rank: int,
    size_bytes: int,
    dtype: mx.Dtype,
    element_size: int,
    warmup: int,
    iterations: int,
) -> list[float]:
    n_elements = size_bytes // element_size
    tensor = mx.random.normal(shape=(n_elements,)).astype(dtype)
    mx.eval(tensor)

    for _ in range(warmup):
        if rank == 0:
            sent = mx.distributed.send(tensor, dst=1, group=group)
            mx.eval(sent)
            received = mx.distributed.recv_like(tensor, src=1, group=group)
            mx.eval(received)
        else:
            received = mx.distributed.recv_like(tensor, src=0, group=group)
            mx.eval(received)
            sent = mx.distributed.send(received, dst=0, group=group)
            mx.eval(sent)
        barrier(group)

    times: list[float] = []
    for _ in range(iterations):
        barrier(group)
        t0 = time.perf_counter()
        if rank == 0:
            sent = mx.distributed.send(tensor, dst=1, group=group)
            mx.eval(sent)
            received = mx.distributed.recv_like(tensor, src=1, group=group)
            mx.eval(received)
        else:
            received = mx.distributed.recv_like(tensor, src=0, group=group)
            mx.eval(received)
            sent = mx.distributed.send(received, dst=0, group=group)
            mx.eval(sent)
        barrier(group)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def bench_all_gather(
    group: mx.distributed.Group,
    rank: int,
    size_bytes: int,
    dtype: mx.Dtype,
    element_size: int,
    warmup: int,
    iterations: int,
) -> list[float]:
    n_elements = (size_bytes // 2) // element_size
    tensor = mx.random.normal(shape=(n_elements,)).astype(dtype)
    mx.eval(tensor)

    for _ in range(warmup):
        gathered = mx.distributed.all_gather(tensor, group=group)
        mx.eval(gathered)
        barrier(group)

    times: list[float] = []
    for _ in range(iterations):
        barrier(group)
        t0 = time.perf_counter()
        gathered = mx.distributed.all_gather(tensor, group=group)
        mx.eval(gathered)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def print_table(title: str, rows: list[dict[str, str]]) -> None:
    print(f"\n=== {title} ===")
    headers = ["Size", "Median", "Min", "Max", "Bandwidth"]
    widths = [
        max(len(h), max((len(r[h]) for r in rows), default=0)) + 2 for h in headers
    ]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(row[h].ljust(w) for h, w in zip(headers, widths, strict=True)))


def run_bench(
    name: str,
    bench_fn,
    group: mx.distributed.Group,
    rank: int,
    dtype: mx.Dtype,
    element_size: int,
    warmup: int,
    iterations: int,
    bw_multiplier: int = 1,
) -> None:
    rows: list[dict[str, str]] = []
    for size in SIZES:
        if rank == 0:
            print(f"  {name}: {format_bytes(size)}...", end="", flush=True)
        times = bench_fn(group, rank, size, dtype, element_size, warmup, iterations)
        if rank == 0:
            med = statistics.median(times)
            mn = min(times)
            mx_ = max(times)
            bw = (size * bw_multiplier) / med
            rows.append(
                {
                    "Size": format_bytes(size),
                    "Median": format_time(med),
                    "Min": format_time(mn),
                    "Max": format_time(mx_),
                    "Bandwidth": format_bandwidth(bw),
                }
            )
            print(f" {format_bandwidth(bw)}")
    if rank == 0:
        print_table(name, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLX Distributed Communication Benchmark"
    )
    subparsers = parser.add_subparsers(dest="backend", required=True)

    ring_parser = subparsers.add_parser("ring")
    ring_parser.add_argument("--rank", type=int, required=True, choices=[0, 1])
    ring_parser.add_argument("--self-ip", required=True)
    ring_parser.add_argument("--peer-ip", required=True)
    ring_parser.add_argument("--port", type=int, default=5555)

    jaccl_parser = subparsers.add_parser("jaccl")
    jaccl_parser.add_argument("--rank", type=int, required=True, choices=[0, 1])
    jaccl_parser.add_argument("--interface", required=True)
    jaccl_parser.add_argument(
        "--coordinator",
        type=str,
        default=None,
        help="IP:PORT of rank 0 (required for rank 1)",
    )
    jaccl_parser.add_argument(
        "--port", type=int, default=9999, help="Coordinator port (rank 0 only)"
    )

    for p in [ring_parser, jaccl_parser]:
        p.add_argument("--warmup", type=int, default=3)
        p.add_argument("--iterations", type=int, default=10)
        p.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="float32")

    args = parser.parse_args()

    if args.backend == "jaccl" and args.rank == 1 and args.coordinator is None:
        jaccl_parser.error("--coordinator is required for rank 1")

    return args


def main() -> int:
    args = parse_args()
    dtype, element_size = DTYPE_MAP[args.dtype]

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.backend == "ring":
            print(f"Initializing ring backend (rank {args.rank})...")
            group = init_ring(args.rank, args.self_ip, args.peer_ip, args.port, tmpdir)
        else:
            print(f"Initializing jaccl backend (rank {args.rank})...")
            group = init_jaccl(
                args.rank, args.interface, args.coordinator or "", args.port, tmpdir
            )

        print(f"Rank {group.rank()} of {group.size()} initialized")
        barrier(group)

        if args.rank == 0:
            print("\nMLX Distributed Communication Benchmark")
            print(
                f"Backend: {args.backend} | Dtype: {args.dtype} | Warmup: {args.warmup} | Iterations: {args.iterations}"
            )

        run_bench(
            "Unidirectional (rank 0 -> rank 1)",
            bench_unidirectional,
            group,
            args.rank,
            dtype,
            element_size,
            args.warmup,
            args.iterations,
        )
        run_bench(
            "Round-Trip (ping-pong)",
            bench_rtt,
            group,
            args.rank,
            dtype,
            element_size,
            args.warmup,
            args.iterations,
            bw_multiplier=2,
        )
        run_bench(
            "All-Gather",
            bench_all_gather,
            group,
            args.rank,
            dtype,
            element_size,
            args.warmup,
            args.iterations,
        )

        if args.rank == 0:
            print("\nDone.")
        else:
            print("Rank 1 complete.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
