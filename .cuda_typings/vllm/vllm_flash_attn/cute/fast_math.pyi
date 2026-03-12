import cutlass.cute as cute
from cutlass import Int32

@cute.jit
def clz(x: Int32) -> Int32: ...
