import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cutlass_dsl import dsl_user_op

@dsl_user_op
def ld_acquire(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32: ...
@dsl_user_op
def red_relaxed(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None: ...
@dsl_user_op
def red_release(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None: ...
@cute.jit
def wait_eq(
    lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: Int32
) -> None: ...
@cute.jit
def arrive_inc(
    lock_ptr: cute.Pointer,
    thread_idx: int | Int32,
    flag_offset: int,
    val: cutlass.Constexpr[Int32],
) -> None: ...
