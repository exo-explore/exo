import os
import statistics
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import GlobalCounters

A = os.environ["A"]
B = os.environ["B"]

for i in range(8):
  N = 256 * (4 ** i)
  x = Tensor.zeros(N, device=A, dtype=dtypes.float32)
  vals = []
  for j in range(100):
    # random increment on current device
    inc0 = Tensor.uniform(N, low=0.0, high=1.0, device=A, dtype=dtypes.float32)
    x = (x + inc0).realize()

    Device[A].synchronize()
    Device[B].synchronize()
    GlobalCounters.reset()
    with Context(DEBUG=2):
      # move to other device
      x = x.to(B).realize()
      Device[B].synchronize()
    vals.append(GlobalCounters.time_sum_s * 1_000_000)
    x = x.to(A).realize()
    Device[A].synchronize()
  print( f"result: {N*4:8d} bytes moved from {A} to {B} in {statistics.mean(vals):.2f}us with stddev {statistics.stdev(vals):.2f}us" )
