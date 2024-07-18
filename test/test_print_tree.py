#%%
import unittest
from tinygrad.engine.graph import print_tree

from tinygrad import Tensor, dtypes
from tinygrad.codegen.uops import UOp

import sys, io

class TestPrintTree(unittest.TestCase):

  def _capture_print(self, fn):
    capturedOutput = io.StringIO()
    sys.stdout = capturedOutput
    fn()
    sys.stdout = sys.__stdout__
    return capturedOutput.getvalue()

  def test_print_uop(self):
    x = Tensor.arange(10).schedule()[-1].ast.src[0]
    output =  self._capture_print(lambda: print_tree(x))
    assert output == '\
  0 ━┳ BufferOps.STORE MemBuffer(idx=0, dtype=dtypes.int, \
st=ShapeTracker(views=(View(shape=(10, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)))\n\
  1  ┗━┳ BinaryOps.ADD None\n\
  2    ┣━┳ ReduceOps.SUM (1,)\n\
  3    ┃ ┗━━ BufferOps.CONST ConstBuffer(val=1, dtype=dtypes.int, st=ShapeTrac\
ker(views=(View(shape=(11, 19), strides=(0, 0), offset=0, mask=((0, 11), (9, 19))\
, contiguous=False), View(shape=(10, 10), strides=(1, 20), offset=0, mask=None, contiguous=False))))\n\
  4    ┗━━ BufferOps.CONST ConstBuffer(val=-1, dtype=dtypes.int, st=ShapeTracker(views=(View(shape=(10,\
 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)))\n'

    x = UOp.var("x", dtypes.int)
    x = (x + x) - UOp.const(dtypes.int, 2)
    output = self._capture_print(lambda: print_tree(x))
    assert output == '\
  0 ━┳ UOps.ALU BinaryOps.ADD\n\
  1  ┣━┳ UOps.ALU BinaryOps.ADD\n\
  2  ┃ ┣━━ UOps.VAR x\n\
  3  ┃ ┗━━ UOps.VAR x\n\
  4  ┗━┳ UOps.ALU UnaryOps.NEG\n\
  5    ┗━━ UOps.CONST 2\n'

    """
    x = UPat(UOp.alu(BinaryOps.ADD, UOp.var("x", dtypes.int), UOp.var("x", dtypes.int)))
    assert self._capture_print(lambda: print_tree(x)) == '\
  0 ━━ UOps.ALU            : dtypes.int                [<UOps.VAR: 2>, <UOps.VAR: 2>]   BinaryOps.ADD None\n'

    x = UPat.compile(UOp.store(UOp.var("buf"), UOp.var("idx"),
                               UOp(UOps.CAST, src=tuple(UOp(UOps.GEP, arg=i, src=(UOp.var("val"),)) for i in range(4)))), UOp.store)
    assert self._capture_print(lambda: print_tree(x)) == '\
  0 ━┳ UOps.STORE None\n\
  1  ┣━━ None None\n\
  2  ┣━━ None None\n\
  3  ┗━┳ UOps.CAST None\n\
  4    ┣━┳ UOps.GEP 0\n\
  5    ┃ ┗━━ None None\n\
  6    ┣━┳ UOps.GEP 1\n\
  7    ┃ ┗━━ None None\n\
  8    ┣━┳ UOps.GEP 2\n\
  9    ┃ ┗━━ None None\n\
 10    ┗━┳ UOps.GEP 3\n\
 11      ┗━━ None None\n'
    """

if __name__ == "__main__":
  unittest.main()