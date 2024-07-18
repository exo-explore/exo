from __future__ import annotations
from typing import Optional, Union, Any, Tuple, List, Dict
import functools, itertools, operator
from tinygrad.helpers import all_same, all_int, dedup, round_up, prod, DEBUG, RING
from tinygrad.dtype import DType, ConstType
from tinygrad.ops import BinaryOps, MetaOps, UnaryOps, TernaryOps, ReduceOps
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import sint

def all_reduce(op: ReduceOps, lbs: List[LazyBuffer]) -> List[LazyBuffer]:
  assert all_int(lbs[0].shape), f"does not support symbolic shape {lbs[0].shape}"
  assert all_same([lb.shape[0] for lb in lbs]), "allreduce with uneven shards is undefined"
  bop = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX}[op]

  n_lbs, dim = len(lbs), prod(lbs[0].shape)
  # Ring allreduce doesn't provide a benefit with only 2 nodes or where number of elements is less than 256k (empirically)
  # so just fallback to naive allreduce to save on kernel dispatch, chunking and reassembling chunks.
  use_ring = (RING >= 2 or (n_lbs > 2 and dim > 256_000 and RING >= 1))
  if DEBUG >= 2: print(f"{'RING ALLREDUCE' if use_ring else 'NAIVE ALLREDUCE'} {n_lbs}x{dim} | {lbs[0].dtype}")
  if not use_ring:
    return [functools.reduce(lambda x,y: x.e(bop, y), [x.copy_to_device(lb.device) for x in lbs]) for lb in lbs]
  factor = max(f for f in [32, 16, 8, 4, 2, 1] if dim % f == 0)
  base, left = (dim // factor) // n_lbs, (dim // factor) % n_lbs
  c_lens = [(base + 1) * factor if i < left else base * factor for i in range(n_lbs)]
  acc = 0
  chunks = [(acc, (acc := acc + i)) for i in c_lens if i > 0]
  chunked = [[lb.reshape((dim,)).shrink(((s,e),)) for s,e in chunks] for lb in lbs]

  # Scatter-reduce step
  for step in range(n_lbs - 1):
    for i in range(len(chunks)):
      s, r = (i+step)%n_lbs, (i+step+1)%n_lbs
      chunked[r][i] = chunked[r][i].e(bop, chunked[s][i].copy_to_device(chunked[r][i].device, force=True))

  # Allgather step
  for step in range(n_lbs - 1):
    for i in range(len(chunks)):
      s, r = (i+step-1)%n_lbs, (i+step)%n_lbs
      chunked[r][i] = chunked[s][i].copy_to_device(chunked[r][i].device, force=True)

  # Assemble chunks back
  pads = [((s,dim-e),) for s,e in chunks]
  return [functools.reduce(lambda x,y: x.e(BinaryOps.ADD, y), [c.pad(pads[i]) for i,c in enumerate(lb_c)]).reshape(lbs[0].shape) for lb_c in chunked]

def to_sharded(lbs:List[LazyBuffer], axis:int) -> List[LazyBuffer]:
  if DEBUG >= 3 and lbs[0].shape[axis] % len(lbs) != 0: print(f"multi axis uneven: {lbs[0].shape=} {axis=} {len(lbs)=}")
  sz = round_up(lbs[0].shape[axis], len(lbs)) // len(lbs)
  return [lb.shrink(tuple((0,s) if a != axis else (min(s,sz*i),min(s,sz*(i+1))) for a,s in enumerate(lb.shape))) for i,lb in enumerate(lbs)]

class MultiLazyBuffer:
  def __init__(self, lbs:List[LazyBuffer], axis:Optional[int], real:Optional[List[bool]]=None):
    assert all(isinstance(x, LazyBuffer) for x in lbs) and len(lbs), "all lbs must be LazyBuffers, and we need at least one of them"
    assert all_same([x.dtype for x in lbs]), f"all multilazybuffer needs same dtype, getting {[x.dtype for x in lbs]}"
    self.lbs, self.axis, self.dtype, self.device, self.real = lbs, axis, lbs[0].dtype, tuple(x.device for x in lbs), real or [True]*len(lbs)
    if axis is not None:
      splits = list(itertools.accumulate([lb.shape[axis] for lb in lbs], initial=0))
      self.bounds = list(zip(splits, splits[1:]))

  @property
  def shape(self):
    return tuple(sum(y.shape[a] for y in self.real_lbs) if a == self.axis else s for a,s in enumerate(self.real_lbs[0].shape))

  @property
  def size(self): return sum(x.size for x in self.real_lbs)

  @property
  def real_lbs(self): return [lb for lb,r in zip(self.lbs, self.real) if r]

  def __repr__(self):
    return f"<MLB {self.axis=} {self.real=} {chr(10)}{chr(10).join([f'{x.device} {x.st}' for x in self.lbs])}>"

  @staticmethod
  def from_sharded(lb:LazyBuffer, devices:Tuple[str, ...], axis:Optional[int]=None):
    lbs = [lb] * len(devices)
    sharded_lbs = [lb.copy_to_device(d) for lb,d in zip(to_sharded(lbs, axis) if axis is not None else lbs, devices)]
    return MultiLazyBuffer([lb if lb.is_unrealized_unmasked_const() else lb.contiguous(allow_buffer_view=False) for lb in sharded_lbs], axis)

  def copy_to_device(self, device:str) -> LazyBuffer:
    if self.axis is None:
      # if we already have a copy on the device, return that
      for lb in self.real_lbs:
        if lb.device == device: return lb
      return self.real_lbs[0].copy_to_device(device)
    # copy lbs to device, pad to final shape, and sum
    llbs:List[LazyBuffer] = []
    for lb,real,(start,end) in zip(self.lbs, self.real, self.bounds):
      if not real: continue
      pad_arg = tuple((0,0) if a != self.axis else (start, self.bounds[-1][1]-end) for a in range(len(lb.shape)))
      llbs.append(lb.copy_to_device(device).pad(pad_arg))
    return functools.reduce(lambda x,y: x.e(BinaryOps.ADD, y), llbs)

  # passthroughs
  def is_realized(self) -> bool: return all(lb.base.realized is not None for lb, r in zip(self.lbs, self.real) if r is True)
  def cast(self, dtype:DType, bitcast:bool=False, allow_buffer_view=True):
    return MultiLazyBuffer([x.cast(dtype, bitcast, allow_buffer_view) for x in self.lbs], self.axis, self.real)
  def const(self, val:ConstType) -> MultiLazyBuffer: return MultiLazyBuffer([x.const(val) for x in self.lbs], self.axis, self.real)
  def assign(self, x:MultiLazyBuffer): return MultiLazyBuffer([s.assign(d) for s,d in zip(self.lbs, x.lbs)], self.axis, self.real)
  def contiguous(self): return MultiLazyBuffer([x.contiguous() for x in self.lbs], self.axis, self.real)

  # elementwise is simple
  def e(self, op:Union[MetaOps, UnaryOps, BinaryOps, TernaryOps], *in_srcs:MultiLazyBuffer, arg:Optional[Any]=None) -> MultiLazyBuffer:
    msrcs = (self,)+in_srcs
    assert all(isinstance(x, MultiLazyBuffer) for x in msrcs), f"all buffers must be MultiLazyBuffer {msrcs}"
    assert all_same([x.device for x in msrcs]), f"all buffers must have the same device {[x.device for x in msrcs]}"

    # NOTE: they all have to share an axis, we always choose [-1]
    axis = axes[-1] if len(axes := dedup([x.axis for x in msrcs if x.axis is not None])) else None
    srcs:List[List[LazyBuffer]] = []
    not_all_real = any(not all(mlb.real) for mlb in msrcs)
    new_real = [all(transposed) for transposed in zip(*[mlb.real for mlb in msrcs])] if not_all_real else self.real
    assert any(new_real), "output contains no real lb"
    for mlb in msrcs:
      if mlb.axis == axis or not_all_real: srcs.append(mlb.lbs)
      elif mlb.axis is None and axis is not None: srcs.append(to_sharded(mlb.lbs, axis))
      else: srcs.append(to_sharded([mlb.copy_to_device(lb.device) for lb in mlb.lbs], axis))
    new_real_lbs:Dict[int,LazyBuffer] = {i:lsrcs[0].e(op, *lsrcs[1:], arg=arg) for i,(lsrcs,r) in enumerate(zip(zip(*srcs), new_real)) if r}
    # NOTE: const dtype should match real
    real_dtype = next(iter(new_real_lbs.values())).dtype
    return MultiLazyBuffer([new_real_lbs.get(i, lsrcs[0].const(0).cast(real_dtype)) for i,lsrcs in enumerate(zip(*srcs))], axis, new_real)

  def r(self, op:ReduceOps, axis:Tuple[int, ...]) -> MultiLazyBuffer:
    if self.axis is not None and self.axis in axis:
      # all-reduce on sharded axes
      reduced_parts = [(x if r else x.const(0)).r(op, axis) for x,r in zip(self.lbs, self.real)]
      if all(self.real): return MultiLazyBuffer(all_reduce(op, reduced_parts), None)
      return MultiLazyBuffer(reduced_parts, None, self.real)
    # reduce on non sharded axes, piecewise is fine. if axis is None this is also correct
    return MultiLazyBuffer([x.r(op, axis) for x in self.lbs], self.axis, self.real)

  # *** movement ops ***

  def _shape_to_single_shard(self, shape:Tuple[sint, ...], lb:LazyBuffer) -> Tuple[sint, ...]:
    return tuple(lb.shape[self.axis] if a == self.axis else s for a,s in enumerate(shape))

  def reshape(self, arg:Tuple[sint, ...]):
    if self.axis is None: return MultiLazyBuffer([x.reshape(arg) for x in self.lbs], None, self.real)
    arg_acc:List[sint] = list(itertools.accumulate(arg, operator.mul, initial=1))
    # new_axis is the last one that preserves prod(prior to new_axis) and must not move items between shards
    # todo: what to do about shrinking to self.shape[self.axis]==1 len(self.real_lbs)==1?
    new_axis = len(arg_acc) - arg_acc[::-1].index(prod(self.shape[:self.axis])) - 1
    if arg[new_axis] != self.shape[self.axis]:
      assert self.shape[self.axis] % len(self.real_lbs) == 0, f"cannot reshape on-axis for uneven shard {self.axis} {self.shape} {len(self.real_lbs)}"
      assert arg[new_axis] % len(self.real_lbs) == 0, f"new on-axis shape must divide evenly between devices {new_axis} {arg} {len(self.real_lbs)}"
    return MultiLazyBuffer([x.reshape(tuple(s if a != new_axis else
                              x.shape[self.axis] if s == self.shape[self.axis] else
                              s // len(self.real_lbs) for a,s in enumerate(arg))) for x in self.lbs],
                           new_axis, self.real)

  def pad(self, arg:Tuple[Tuple[sint, sint], ...]):
    assert self.axis is None or arg[self.axis] == (0,0) or not all(self.real), f"padding not supported for {arg=}"
    # pad on shard axis -> fill others with zeros and set real to all True
    if self.axis is not None and arg[self.axis] != (0,0):
      # pad back to whole axis, remove real mask
      assert all(arg[i] == (0, 0) or i == self.axis for i in range(len(self.shape))), "cannot pad sharded and non-sharded axis at the same time"
      assert arg[self.axis] == (sum(lb.shape[self.axis] for i,lb in enumerate(self.lbs) if i < self.real.index(True)), \
                                sum(lb.shape[self.axis] for i,lb in enumerate(self.lbs) if i > self.real.index(True))), "can only pad to whole axis"
      return MultiLazyBuffer([x if r else x.const(0) for x,r in zip(self.lbs, self.real)], self.axis)
    return MultiLazyBuffer([x.pad(arg) for x in self.lbs], self.axis, self.real)

  def expand(self, arg:Tuple[sint, ...]):
    # NOTE: this assert isn't needed, sharded axis can have dim 1
    assert self.axis is None or arg[self.axis] == self.shape[self.axis], f"expand not supported on sharded axis {arg=}"
    return MultiLazyBuffer([x.expand(self._shape_to_single_shard(arg, x)) for x in self.lbs], self.axis, self.real)

  def permute(self, arg:Tuple[int, ...]):
    # all permutes supported!
    return MultiLazyBuffer([x.permute(arg) for x in self.lbs], arg.index(self.axis) if self.axis is not None else None, self.real)

  def shrink(self, arg:Tuple[Tuple[sint, sint], ...]):
    assert self.axis is None or arg[self.axis] == (0, self.shape[self.axis]) or arg[self.axis] in self.bounds, f"shrinking not supported for {arg=}"
    if self.axis is not None and arg[self.axis] in self.bounds and arg[self.axis] != (0, self.shape[self.axis]):
      assert all(arg[i] == (0, s) or i == self.axis for i,s in enumerate(self.shape)), "cannot shrink sharded and non-sharded axis at the same time"
      idx = self.bounds.index(arg[self.axis])
      # zero out other lbs to not create lb reference
      return MultiLazyBuffer([lb if i==idx else lb.const(0) for i,lb in enumerate(self.lbs)], self.axis, [i==idx for i in range(len(self.lbs))])
    return MultiLazyBuffer([x.shrink(tuple((0, x.shape[self.axis]) if a == self.axis else s for a,s in enumerate(arg))) for x in self.lbs],
                           self.axis, self.real)

  def stride(self, arg:Tuple[int, ...]):
    assert self.axis is None or arg[self.axis] == 1, "flipping not supported on sharded axis"
    return MultiLazyBuffer([x.stride(arg) for x in self.lbs], self.axis, self.real)
