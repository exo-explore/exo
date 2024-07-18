#!/usr/bin/env python
import unittest
from tinygrad.shape.view import View

class TestView(unittest.TestCase):
  def test_canonicalize_empty_mask(self):
    v = View.create(shape=(2,2,2), strides=(4,2,1), mask=((0,2),(0,2),(0,2)))
    assert v.mask is None
    v = View.create(shape=(4,3,2), strides=(1,4,10), mask=((0,4),(0,3),(0,2)))
    assert v.mask is None

  def test_minify_zero_strided_dims(self):
    target = View.create(shape=(2,2), strides=(30,2), offset=7, mask=None)
    v = View.create(shape=(2,1,2), strides=(30,0,2), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(1,2,2), strides=(0,30,2), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(2,2,1), strides=(30,2,0), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(2,1,1,2), strides=(30,0,0,2), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(1,1,2,2), strides=(0,0,30,2), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(2,2,1,1), strides=(30,2,0,0), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(1,2,2,1), strides=(0,30,2,0), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(1,2,1,2), strides=(0,30,0,2), offset=7, mask=None)
    assert v.minify() == target

  def test_empty_mask_contiguous(self):
    v1 = View.create(shape=(2,2,2), strides=(4,2,1), mask=None)
    v2 = View.create(shape=(2,2,2), strides=(4,2,1), mask=((0,2),(0,2),(0,2)))
    assert v1.contiguous == v2.contiguous
    v1 = View.create(shape=(1,1,1,4), strides=(0,0,0,1), offset=0, mask=None)
    v2 = View.create(shape=(1,1,1,4), strides=(0,0,0,1), offset=0, mask=((0,1),(0,1),(0,1),(0,4)))
    assert v1.contiguous == v2.contiguous
    v = View.create(shape=(2,3,4), mask=((0,2),(0,3),(0,4)))
    assert v.contiguous

if __name__ == '__main__':
  unittest.main()
