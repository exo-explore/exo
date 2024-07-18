#!/usr/bin/env python
import unittest
from tinygrad.tensor import Tensor
from tinygrad.codegen.kernel import Kernel
from tinygrad.renderer.cstyle import OpenCLRenderer
from tinygrad.engine.graph import graph_uops
from tinygrad.engine.schedule import create_schedule
from tinygrad.nn import Conv2d

class TestUopsGraph(unittest.TestCase):
  def test_matmul(self):
    N = 1024
    a = Tensor.rand(N,N)
    b = Tensor.rand(N,N)
    si = create_schedule([(a@b).lazydata])[-1]
    lin = Kernel(si.ast)
    lin.hand_coded_optimizations()
    print(lin.colored_shape())
    uops = lin.linearize().uops
    graph_uops(uops)
    for u in uops: print(u)
    print(OpenCLRenderer("matmul", uops)[0])

  def test_reduce(self):
    a = Tensor.rand(1024*1024)
    si = create_schedule([a.sum().lazydata])[-1]
    lin = Kernel(si.ast)
    lin.hand_coded_optimizations()
    uops = lin.linearize().uops
    graph_uops(uops)
    #print(OpenCLRenderer("reduce", uops)[0])

  def test_conv(self):
    x = Tensor.rand(1,3,16,16)
    c = Conv2d(3, 16, (3,3))
    si = create_schedule([c(x).elu().lazydata])[-1]
    lin = Kernel(si.ast)
    lin.hand_coded_optimizations()
    uops = lin.linearize().uops
    graph_uops(uops)
    print(lin.colored_shape())
    print(OpenCLRenderer("conv", uops)[0])

if __name__ == '__main__':
  unittest.main()
