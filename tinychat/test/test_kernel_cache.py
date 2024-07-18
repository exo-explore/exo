#!/usr/bin/env python
import unittest
from tinygrad.tensor import Tensor
from tinygrad import Device

class TestKernelCache(unittest.TestCase):
  def test_kernel_cache_in_action(self):
    if Device.DEFAULT not in ["CLANG"]:
      self.skipTest("No custom kernel cache is implemented")

    a = Tensor.rand(4,4)
    b = Tensor.rand(4,4)
    x = a + b
    x.realize()

    orig_compile_func = Device['CLANG'].compiler
    Device['CLANG'].compiler = None # making it not callable

    a1 = Tensor.rand(4,4)
    b1 = Tensor.rand(4,4)
    x1 = a1 + b1
    x1.realize() # Same kernel should be from cache.

    Device['CLANG'].compiler = orig_compile_func

if __name__ == "__main__":
  unittest.main()
