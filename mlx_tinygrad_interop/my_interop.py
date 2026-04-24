from tinygrad.helpers import GlobalCounters
from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import _from_torch_dtype, _from_np_dtype
import torch
import time
import statistics

def pytorch_to_tinygrad(tensor: torch.Tensor) -> Tensor:
    return Tensor.from_blob(tensor.data_ptr(), tensor.shape, dtype=_from_torch_dtype(tensor.dtype), device="METAL")

def main() -> None:
    for i in range(8):
        N = 256 * (4 ** i)
        x = torch.zeros(N, device=torch.device("mps"))
        
        vals = []
        for j in range(1000):
            x = x.uniform_()
            torch.mps.synchronize()

            old = time.perf_counter_ns()
            pytorch_to_tinygrad(x)
            new = time.perf_counter_ns()
            vals.append(new - old)
        print( f"result: {N*4:8d} pytorch to tinygrad in {statistics.mean(vals):.2f}ns with stddev {statistics.stdev(vals):.2f}ns" )


if __name__=="__main__":
    main()