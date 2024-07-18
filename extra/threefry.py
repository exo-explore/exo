import os
if "DEBUG" not in os.environ: os.environ["DEBUG"] = "2"
if "THREEFRY" not in os.environ: os.environ["THREEFRY"] = "1"

from tinygrad import Tensor, GlobalCounters

for N in [10_000_000, 100_000_000, 1_000_000_000]:
  GlobalCounters.reset()
  Tensor.rand(N).realize()
  print(f"N {N:>20_}, global_ops {GlobalCounters.global_ops:>20_}, global_mem {GlobalCounters.global_mem:>20_}")