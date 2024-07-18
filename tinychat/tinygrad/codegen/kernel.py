from __future__ import annotations
import itertools, functools
from dataclasses import replace
from collections import defaultdict
from typing import Optional, List, Tuple, cast, Dict, Union, Final, DefaultDict
from tinygrad.engine.graph import print_tree
from tinygrad.ops import LazyOp, UnaryOps, BinaryOps, ReduceOps, MemBuffer, ConstBuffer, BufferOps, MetaOps, UNSAFE_PAD_OPS, \
                         verify_lazyop, KernelInfo, get_lazyop_info
from tinygrad.device import Device
from tinygrad.renderer import Renderer, TensorCore, Program
from tinygrad.dtype import dtypes, ImageDType
from tinygrad.helpers import all_same, colored, ansilen, dedup, getenv, prod, DEBUG, TC_OPT, USE_TC, round_up, all_int, \
                             get_contraction, to_function_name, diskcache_put, ContextVar
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import sint
from tinygrad.shape.view import strides_for_shape
from tinygrad.codegen.uops import UOps, flops_mem
from tinygrad.codegen.uopgraph import UOpGraph
from tinygrad.codegen.lowerer import lazyop_to_uop
from dataclasses import dataclass
from enum import Enum, auto

class OptOps(Enum):
  TC = auto(); UPCAST = auto(); UPCASTMID = auto(); UNROLL = auto(); LOCAL = auto() # noqa: E702
  GROUP = auto(); GROUPTOP = auto(); NOLOCALS = auto(); PADTO = auto(); MERGE = auto() # noqa: E702
  def __lt__(self, x:OptOps): return self.value < x.value

class KernelOptError(Exception): pass

def check(cond:bool, msg:str=""):
  if not cond: raise KernelOptError(msg)

@dataclass(frozen=True, order=True)
class Opt:
  op: OptOps
  axis: Optional[int] = None
  amt: Optional[int] = None
  def __repr__(self): return f"Opt(op={self.op}, axis={self.axis}, amt={self.amt})"
  def real_axis(self, k:Kernel):
    if self.axis is None: return -1
    if self.op is OptOps.UNROLL: return k.first_reduce+self.axis
    if self.op in {OptOps.GROUP, OptOps.GROUPTOP}: return k.first_reduce+k.group_for_reduces+self.axis
    return self.axis

@dataclass
class TensorCoreOptions:
  axes: Tuple[int, ...] # the location of the original N and M axes if still in the shape
  axes_exist: Tuple[bool, ...] # true if the original N and M axes are still in the shape
  axis_pads: Tuple[Tuple[int, int], ...]
  def fix_axes(self, removed_axis:int): # adjust the TC axes if necesssary when a dimension is removed
    axes, axes_exist = list(self.axes), list(self.axes_exist)
    for tc_dim in [i for i in range(2) if axes_exist[i]]:
      if removed_axis < axes[tc_dim]: axes[tc_dim] -= 1
      elif removed_axis == axes[tc_dim]: axes_exist[tc_dim] = False
    self.axes, self.axes_exist = tuple(axes), tuple(axes_exist)

class Kernel:
  def __init__(self, *ast:LazyOp, opts:Optional[Renderer]=None):
    if len(ast) > 1 or ast[0].op is BufferOps.STORE:
      assert all(x.op is BufferOps.STORE for x in ast)
      self.ast = LazyOp(MetaOps.KERNEL, ast)
    else:
      assert len(ast) == 1 and ast[0].op is MetaOps.KERNEL
      self.ast = ast[0]

    self.opts = opts if opts is not None else Device[Device.DEFAULT].renderer
    try: lazyop_sts_map = verify_lazyop(self.ast)
    except AssertionError as e:
      print("INVALID AST")
      for op in ast: print_tree(op)
      raise e

    cached_ordered_lazyops: Dict[LazyOp, List[LazyOp]] = {}
    def ordered_lazyops(op):
      if op not in cached_ordered_lazyops: cached_ordered_lazyops[op] = dedup([item for x in op.src for item in ordered_lazyops(x)] + [op])
      return cached_ordered_lazyops[op]
    self.reduceops = dedup([x for x in ordered_lazyops(self.ast) if x.op in ReduceOps])

    self.vars = self.ast.vars()
    self.bufs: List[Union[MemBuffer, ConstBuffer]] = dedup([x.arg for x in self.ast.lazyops if x.op in BufferOps])

    # get earlybufs, before any reduceops
    earlybufs = [x.arg for reduceop in self.reduceops for x in reduceop.lazyops if x.op in BufferOps]
    self.full_buf_index: int = self.bufs.index(earlybufs[0]) if earlybufs else 0
    # NOTE: full_shape can be wrong if there's a tree of reduces

    # create new shapetrackers inside this kernel, we will permute them
    self.sts: List[ShapeTracker] = [x.st for x in self.bufs]

    # add the shapetrackers for each reduce
    # we use this to track which axes are reduced in each reduce
    for x in self.reduceops:
      self.sts.append(lazyop_sts_map[x])
      self.sts.append(lazyop_sts_map[x.src[0]])

    # move all reduce axes to the end
    reduce = list(enumerate(zip(self.full_shape, self.output_shape)))
    permute = tuple([i for i,(s,n) in reduce if s == n] + [i for i,(s,n) in reduce if s != n])
    self.reshape_and_permute(None, permute)

    # parameters for optimization
    self.applied_opts: List[Opt] = []
    self.group_for_reduces: int = 0
    self.upcasted: int = 0
    self.local_dims: int = 0
    self.tensor_core: Optional[TensorCore] = None
    self.tensor_core_opts: Optional[TensorCoreOptions] = None
    # the local aliased buffers for A and B
    self.bufs_for_tensor_core: Dict[LazyOp, Tuple[int, int]] = {}
    self.dont_use_locals: bool = False

    # group simplifies
    self.simplify_ones()
    self.simplify_merge_adjacent()

    # cache
    self.applied_opts_cache: Optional[List[Opt]] = None

  def copy(self):
    ret = type(self).__new__(type(self))

    # base linearizer params
    ret.opts, ret.ast = self.opts, self.ast

    # things downstream of the AST
    ret.reduceops, ret.vars, ret.bufs, ret.full_buf_index = \
      self.reduceops, self.vars, self.bufs, self.full_buf_index
    ret.sts = self.sts[:len(ret.bufs)+len(ret.reduceops)*2] # NOTE: must redo the local buffers with TC in beam

    # parameters for optimizations
    ret.applied_opts, ret.group_for_reduces, ret.upcasted, ret.local_dims, ret.dont_use_locals = \
      self.applied_opts[:], self.group_for_reduces, self.upcasted, self.local_dims, self.dont_use_locals
    ret.tensor_core, ret.tensor_core_opts, ret.bufs_for_tensor_core = self.tensor_core, self.tensor_core_opts, self.bufs_for_tensor_core

    # uncached since linearize didn't run
    ret.applied_opts_cache = None

    return ret

  @property
  def membufs(self) -> List[MemBuffer]: return [x for x in self.bufs if isinstance(x, MemBuffer)]

  # TODO: these need more tests or it might silently be no-op
  def shape_offsets(self, i:int): return itertools.product(*[list(range(cast(int, s))) for s in self.sts[i].shape[self.shape_len-self.upcasted:][::-1]]) if self.upcasted > 0 else [tuple()]  # noqa: E501
  def float4_axis(self, i:int): return [x-(self.shape_len-self.upcasted) for x in self.sts[i].unit_stride_axes() if x >= self.shape_len-self.upcasted and self.sts[i].shape[x]%4 == 0]  # noqa: E501

  def upcasted_axis(self, i:int) -> List[Tuple[int, Optional[sint], bool]]:
    upcasted_shape, upcasted_stride = self.sts[i].shape[self.shape_len-self.upcasted:], self.sts[i].real_strides()[self.shape_len-self.upcasted:]
    assert all_int(upcasted_shape), f"cannot upcast a symbolic amount {upcasted_shape=}"
    return list(zip(upcasted_shape, upcasted_stride,
                    [x!=y for x,y in zip(self.sts[0].shape[self.shape_len-self.upcasted:], self.full_shape[self.shape_len-self.upcasted:])]))

  # TODO: is there a better way to write this?
  def acc_offsets(self, i:int) -> List[int]:
    if self.upcasted == 0: return [0]
    upcasted_i = self.upcasted_axis(i)
    acc_strides = [x*(1-upcasted_i[::-1][i][2]) for i,x in enumerate(strides_for_shape(tuple(1 if r else s for s,_,r in upcasted_i[::-1])))]
    return [sum(t) for t in itertools.product(*[[y*acc_strides[i] for y in range(x[0])] for i,x in enumerate(upcasted_i[::-1])])]

  def get_float4_upcast_dim(self, i:int) -> List[int]:
    should_upcast = self.opts.supports_float4 and (self.bufs[i].dtype in (dtypes.float, dtypes.half) or isinstance(self.bufs[i].dtype, ImageDType))
    return [x for x in self.sts[i].unit_stride_axes() if x >= self.shape_len-self.upcasted and self.sts[i].shape[x] > 1] if should_upcast else []

  @property
  def first_reduce(self) -> int:
    return [x!=y for x,y in zip(self.sts[0].shape[:self.shape_len-self.upcasted]+(0,), self.full_shape[:self.shape_len-self.upcasted]+(1,))].index(True)  # noqa: E501

  @property
  def reduceop(self) -> Optional[LazyOp]: return self.reduceops[0] if len(self.reduceops) > 0 else None

  @property
  def output_shape(self) -> Tuple[sint, ...]: return self.sts[0].shape

  @property
  def full_shape(self) -> Tuple[sint, ...]: return self.sts[self.full_buf_index].shape

  @property
  def full_unupcasted_shape(self) -> Tuple[sint, ...]: return self.full_shape[:self.shape_len-self.upcasted]

  @property
  def shape_len(self) -> int: return len(self.sts[0].shape)

  @property
  def upcast_in_mid_reduce_axes(self) -> List[int]:
    return [j for j in range(self.first_reduce, self.first_reduce+self.group_for_reduces) if self.full_shape[j] == self.sts[0].shape[j]]

  @property
  def global_dims(self) -> int: return self.first_reduce-self.local_dims

  # there's eight chunks of the shape
  # blue   -- global dims
  # cyan   -- local dims (warp ones first)
  #  *** self.first_reduce
  # green  -- reduce-local dims
  # white  -- reduce-late upcasted dim (self.upcast_in_mid_reduce_axes)
  # red    -- reduce loops
  #  *** self.upcasted
  # purple -- reduce upcasted
  # yellow -- normal upcasted dimensions
  def colors(self) -> List[str]:
    # first non local non reduce dims are global (blue)
    colors = ["blue"] * self.global_dims if not self.dont_use_locals else ["BLUE"] * self.global_dims
    # after global are local_dims; warp ones used in tensor cores must be closest to first_reduce (cyan)
    colors += ["cyan"] * self.local_dims
    # between first_reduce and first_reduce + group_for_reduces, they are either upcast mid reduce (white), or late upcasted (green)
    colors += ["white" if i in self.upcast_in_mid_reduce_axes else "green" for i in range(self.first_reduce, self.first_reduce + self.group_for_reduces)]  # noqa: E501
    # between first_reduce + group_for_reduces and upcasted, they are reduce (red)
    colors += ["red"] * ((self.shape_len-self.upcasted) - (self.first_reduce + self.group_for_reduces))
    # upcasted dimensions are reduce (magenta) or normal (yellow)
    colors += ["magenta" if self.full_shape[i] != self.sts[0].shape[i] else "yellow" for i in range(self.shape_len-self.upcasted, self.shape_len)]
    assert len(colors) == self.shape_len, "colors size mismatch"
    return colors

  def colored_shape(self, pad:Optional[int]=None, dense=False) -> str:
    ret = ' '.join(colored(s, color) for s,color in zip([f"{s:4d}" if isinstance(s, int) and not dense else s for s in self.full_shape], self.colors()))  # noqa: E501
    if pad: ret += ' '*(pad-ansilen(ret))
    return ret

  # ******************** base simplifiers ********************

  # apply reshape and permute to all shapetrackers
  def reshape_and_permute(self, new_shape_fxn, axis):
    new_sts = []
    for st in self.sts:
      if new_shape_fxn is not None: st = st.reshape(tuple(new_shape_fxn(st.shape)))
      if axis is not None: st = st.permute(tuple(axis))
      new_sts.append(st)
    self.sts = new_sts

  # drops the final dimension
  def upcast(self):
    check(self.full_shape[-1] != 1, "can't upcast a dimension with size 1")
    self.upcasted += 1

  # axis : the axis to pull from
  # amount : the amount to take
  # top : if you want to pull that amount from the top
  # insert_before : place to insert the new stuff
  def shift_to(self, axis, amount, top=False, insert_before=None):
    if insert_before is None: insert_before = self.shape_len
    move_axis = axis if top else axis+1
    if move_axis < insert_before: insert_before += 1
    self.reshape_and_permute(
      lambda x: x[0:axis] + (((amount, x[axis]//amount) if top else (x[axis]//amount, amount)) if x[axis] > 1 else (1,1)) + x[axis+1:],
      [i for i in range(insert_before) if i != move_axis] + [move_axis] + [i for i in range(insert_before, self.shape_len+1) if i != move_axis])

  # ******************** complex simplifiers ********************

  def simplify_ones(self) -> bool:
    # remove places where the shape is all ones
    # TODO: this should be factored in to multi shape stride
    if self.shape_len == 0: return False
    all_ones = [s==1 for s in self.full_shape]
    self.local_dims -= sum(all_ones[self.first_reduce-self.local_dims:self.first_reduce])
    self.upcasted -= sum(all_ones[self.shape_len-self.upcasted:]) # TODO: no necessary since upcasted axis can't be un-upcasted
    self.reshape_and_permute(lambda shape: [x for i,x in enumerate(shape) if not all_ones[i]], None)
    return any(all_ones)

  def simplify_merge_adjacent(self):
    if self.shape_len == 0: return
    shapes, strides = [x.shape for x in self.sts], [x.real_strides() for x in self.sts]

    # if it's an image, insert fake strides such that this fusion doesn't happen across image axes
    if isinstance(self.bufs[0].dtype, ImageDType):
      base_shape = self.bufs[0].dtype.shape
      if shape_idx_groups := get_contraction(self.output_shape, base_shape):
        special_strides: Tuple[sint, ...] = tuple()
        for i,g in enumerate(shape_idx_groups):
          shape_piece = tuple(self.output_shape[x] for x in g)
          assert prod(shape_piece) == base_shape[i], f"get_contraction was wrong? {shape_piece} != {base_shape[i]}"
          special_strides += strides_for_shape(shape_piece)
        # adding the fake image shape
        shapes.append(self.output_shape)
        strides.append(special_strides)

    # merge dimensions if we can, multi _merge_dims
    # NOTE: this does not always preserve the reduce dimension
    # TODO: move this into shapetracker, with tests!
    # TODO: how does this work with multi-reduce?
    rets = [[(s[0], st[0])] for s,st in zip(shapes, strides)]
    for i in range(1, len(shapes[0])):
      can_merge = []
      for s,st,ret in zip(shapes, strides, rets):
        # TODO: added the always mergeability of 1s, is this right? if so, add to shapetracker in the 1 case
        si, sti, last_st = s[i], st[i], ret[-1][1]
        can_merge.append((sti is not None) and ((sti != 0 and last_st == si*sti) or (sti == 0 and last_st == 0)))
      # more can merge than this
      mergeable = all(can_merge) and i != self.first_reduce
      for j,(s,st) in enumerate(zip(shapes, strides)):
        if mergeable: rets[j][-1] = (rets[j][-1][0] * s[i], st[i])
        else: rets[j].append((s[i], st[i]))

    # do the reshapes
    for i,x in enumerate(rets[:len(self.sts)]): self.sts[i] = self.sts[i].reshape(tuple([y[0] for y in x]))

  # ******************** high level optimizers ********************

  def _create_tc_opts(self, reduceop:LazyOp, tc:TensorCore, axis:int, opt_level:int) -> Optional[TensorCoreOptions]:
    has_cast = tc.dtype_in != tc.dtype_out
    if has_cast and not(reduceop.src[0].op is UnaryOps.CAST and reduceop.src[0].arg == tc.dtype_out): return None

    mul_op = reduceop.src[0].src[0] if has_cast else reduceop.src[0]
    if mul_op.op is not BinaryOps.MUL: return None

    def buf_index(src: LazyOp) -> Optional[int]:
      # TODO: apply tc even if the sources are not from LOAD
      if src.op is BufferOps.LOAD and src.arg.dtype == tc.dtype_in: return self.bufs.index(cast(MemBuffer, src.arg))
      try:
        if opt_level >= 1 and src.op is UnaryOps.CAST and src.arg == tc.dtype_in: return self.bufs.index(cast(MemBuffer, src.src[0].arg))
      except ValueError: return None
      return None
    if (buf0:=buf_index(mul_op.src[0])) is None or (buf1:=buf_index(mul_op.src[1])) is None: return None

    buf0_strides, buf1_strides = self.sts[buf0].real_strides(), self.sts[buf1].real_strides()
    axis_buf0 = [(i,self.full_shape[i],buf1_strides[i]) for i,s in enumerate(buf0_strides[:self.first_reduce]) if s == 0]
    axis_buf1 = [(i,self.full_shape[i],buf0_strides[i]) for i,s in enumerate(buf1_strides[:self.first_reduce]) if s == 0]
    if not(axis_buf0 and axis_buf1 and ((self.shape_len-self.first_reduce) == 1 or (opt_level >= 1))): return None

    axis_choices = list(itertools.product(axis_buf0, axis_buf1, range(self.first_reduce, self.shape_len)))
    if not(axis < len(axis_choices)): return None

    s0, s1, s2 = axis_choices[-(axis+1)][0][0], axis_choices[-(axis+1)][1][0], axis_choices[-(axis+1)][2]  # s0 is n, s1 is m, s2 is k
    axis_pads = tuple((x, tc.dims[i]) for i, x in enumerate([s0, s1, s2]) if self.full_shape[x]%tc.dims[i] != 0)
    if axis_pads and (opt_level < 2): return None
    self.bufs_for_tensor_core[reduceop] = (buf0, buf1)
    if DEBUG >= 3: print("TENSOR CORES", axis_buf0, axis_buf1, tc)
    return TensorCoreOptions(axes=(s0, s1, s2), axes_exist=(True, True), axis_pads=axis_pads)

  def _apply_tc_opt(self, use_tensor_cores:int, axis:int, opt_level:int) -> bool:
    if use_tensor_cores and self.opts.has_local and self.reduceop is not None and self.reduceop.op is ReduceOps.SUM:
      for tc in self.opts.tensor_cores:
        tensor_core_opts = [self._create_tc_opts(reduceop, tc, axis, opt_level) for reduceop in self.reduceops]
        # can only fuse reduces with the same tc options
        assert all_same(tensor_core_opts)
        if tensor_core_opts[0] is None: continue
        # tensor core -- unroll the reduce dim, upcast input, then create the correct thread pattern
        self.tensor_core_opts = tc_opts = tensor_core_opts[0]

        # attempt to pad the tensor axes that require it
        try:
          for axis, dim in tc_opts.axis_pads: self.apply_opt(Opt(OptOps.PADTO, axis, dim), append_opt=False) # PADTO might fail
        except KernelOptError: continue
        if self.opts.device == "AMD":
          # NOTE: AMD requires locals first
          self.apply_opt(Opt(OptOps.UNROLL, tc_opts.axes[2]-self.first_reduce, tc.dims[2]), append_opt=False)
          for (tc_dim, tc_amt) in tc.threads:
            self.apply_opt(Opt(OptOps.LOCAL, tc_opts.axes[tc_dim], tc_amt), append_opt=False)
          for i, sz in enumerate([prod(x) for x in [[x[1] for x in tc.threads if x[0]==dim] for dim in range(2)]]): # upcast non-local'd N, M
            if tc.dims[i] > sz: self.apply_opt(Opt(OptOps.UPCAST, tc_opts.axes[i], tc.dims[i]//sz), append_opt=False)
        elif self.opts.device == "METAL":
          self.apply_opt(Opt(OptOps.UNROLL, tc_opts.axes[2]-self.first_reduce, tc.dims[2]), append_opt=False)
          for i, sz in enumerate([prod(x) for x in [[x[1] for x in tc.threads if x[0]==dim] for dim in range(2)]]): # upcast non-local'd N, M
            if tc.dims[i] > sz: self.apply_opt(Opt(OptOps.UPCAST, tc_opts.axes[i], tc.dims[i]//sz), append_opt=False)
          for (tc_dim, tc_amt) in tc.threads:
            self.apply_opt(Opt(OptOps.LOCAL, tc_opts.axes[tc_dim], tc_amt), append_opt=False)
        elif self.opts.device in {"CUDA", "NV"}:
          self.apply_opt(Opt(OptOps.UNROLL, tc_opts.axes[2]-self.first_reduce, 8), append_opt=False)
          self.apply_opt(Opt(OptOps.UNROLL, tc_opts.axes[2]-self.first_reduce, 2), append_opt=False)
          self.apply_opt(Opt(OptOps.UPCAST, tc_opts.axes[0], 2), append_opt=False)
          self.apply_opt(Opt(OptOps.LOCAL, tc_opts.axes[0], 2), append_opt=False)
          self.apply_opt(Opt(OptOps.LOCAL, tc_opts.axes[0], 2), append_opt=False)
          self.apply_opt(Opt(OptOps.LOCAL, tc_opts.axes[1], 2), append_opt=False)
          self.apply_opt(Opt(OptOps.LOCAL, tc_opts.axes[1], 2), append_opt=False)
          self.apply_opt(Opt(OptOps.LOCAL, tc_opts.axes[1], 2), append_opt=False)
          self.apply_opt(Opt(OptOps.UPCAST, tc_opts.axes[1], 2), append_opt=False)
          # NOTE: MERGE is needed because we can't deal with two upcasted dimensions
          self.apply_opt(Opt(OptOps.MERGE, self.shape_len-2), append_opt=False)
        # assert tensor core
        if use_tensor_cores == 1: self.tensor_core = tc # TC=2 will do the shape ops without the WMMA
        return True
    return False

  def apply_tensor_cores(self, use_tensor_cores=1, extra_opts:Optional[List[Opt]]=None, axis:int=0, tc_opt:Optional[int]=None) -> bool:
    """ Attempts to apply a tensor core optimization to the kernel.  If one exists and applies properly, return true, otherwise return false.
    Tensor cores are optimized instructions that matrix multiply-accumulate across a wave of threads: D(M, N) = A(M, K) * B(K, N) + C(M, N).

    Keyword arguments:
    use_tensor_cores -- controls how tensor cores are applied (default 1)
      0: will disable any tensor core matching
      1: enable tensor cores
      2: apply tensor core shape but don't use UOp.WMMA
    extra_opts -- additional Opt's to apply after the tensor core instead of the hand-coded additional Opt's (default None)
    tc_opt -- controls which kinds of kernels may be eligible for tensor cores application (default 2 during BEAM, 0 otherwise)
      0: applies to only kernels with a single reduce axis and direct BufferOps.LOAD into BinaryOps.MUL
      1: allows kernels with multiple reduce axes and also multiplication of UnaryOps.CAST'd buffers
      2: allows kernels with M, N, K axes that are not multiples of the tensor core dimensions by applying padding those axes as needed
    """
    if tc_opt is None: tc_opt = TC_OPT.value
    if not self.opts.tensor_cores and use_tensor_cores != 2: return False
    try: # check TC first and apply hand-coded opts if successful
      self.apply_opt(Opt(OptOps.TC, axis, tc_opt))

      if (tc_opts:=self.tensor_core_opts) is not None:
        if extra_opts is not None:
          for opt in extra_opts: self.apply_opt(opt)
        else:
          # hand-coded TC opts
          def late_upcast_tc(tc_dim: int):
            if tc_opts.axes_exist[tc_dim]:
              ax_div = [upc for upc in [5,4,3,2,1] if self.full_shape[tc_opts.axes[tc_dim]]%upc == 0][0]
              if ax_div != 1: self.apply_opt(Opt(OptOps.UPCAST, tc_opts.axes[tc_dim], ax_div))
          late_upcast_tc(1) # attempt to upcast M
          late_upcast_tc(0) # attempt to upcast N

          if self.tensor_core and tc_opts.axes_exist[0]: # attempt to local N
            for upc in [4,2]:
              if self.full_shape[tc_opts.axes[0]] % upc == 0:
                self.apply_opt(Opt(OptOps.LOCAL, tc_opts.axes[0], upc))
                break

      return True
    except KernelOptError:
      return False

  def apply_opt(self, opt:Opt, append_opt:bool=True):
    check(not self.dont_use_locals or opt.op not in {OptOps.LOCAL, OptOps.GROUP, OptOps.GROUPTOP, OptOps.UPCASTMID}, "not using locals")

    if opt.op is OptOps.TC:
      check(len(self.applied_opts) == 0, "tensor core opts must be first") # TODO: things like PADTO might be fine
      check(opt.axis is not None and opt.amt is not None, "tensor core opts must have an axis and amt")
      check((use_tensor_cores:=USE_TC.value) == 2 or len(self.opts.tensor_cores) > 0, "must have tensor cores or TC=2")
      check(self._apply_tc_opt(use_tensor_cores, cast(int, opt.axis), cast(int, opt.amt)), "no tensor core available")
      self.applied_opts.append(opt)
      return

    axis = opt.real_axis(self)
    check(axis < len(self.full_shape), "invalid axis")

    if opt.amt is not None:
      amt = opt.amt if opt.amt != 0 else self.full_shape[axis]
      check(isinstance(amt, int) and amt != 1, "shift/padto of amt 1 or Node is meaningless")
      if opt.op is not OptOps.PADTO: check(self.full_shape[axis] % amt == 0, "no longer valid shift")
    else: amt = -1

    if self.reduceop and (opt.op in {OptOps.GROUP, OptOps.GROUPTOP} or (self.group_for_reduces and opt.op not in {OptOps.NOLOCALS, OptOps.PADTO})):
      acc_sz, upcast_idx = dt.base.itemsize if isinstance((dt:=self.reduceop.dtype), ImageDType) else dt.itemsize, self.shape_len-self.upcasted
      upcast_sz = prod([a for a,b in zip(self.full_shape[upcast_idx:], self.sts[0].shape[upcast_idx:]) if a == b])
      local_sz = prod(self.full_shape[self.first_reduce-self.local_dims:self.first_reduce+self.group_for_reduces])
      smem_sz = amt*acc_sz*upcast_sz*local_sz
      check(smem_sz <= self.opts.shared_max, f"exceeds maximum shared memory size: needs {smem_sz}, max {self.opts.shared_max}")

    if opt.op is OptOps.LOCAL:    # cyan
      check(self.opts.has_local, "target does not support local")
      check(axis < self.global_dims, "local is for globals")
      self.shift_to(axis, amt, insert_before=self.first_reduce)
      self.local_dims += 1
    elif opt.op in {OptOps.GROUP, OptOps.GROUPTOP}:   # green
      check(self.opts.has_local and self.opts.has_shared, "target does not support local or shared mem")
      check(axis >= self.first_reduce + self.group_for_reduces and axis < self.shape_len-self.upcasted, "must be reduce axis to group")
      check(not self.tensor_core, "can't group with tensor cores")
      check(len(self.reduceops) == 1, "can't group with multiple reduces")
      self.shift_to(axis, amt, top=(opt.op is OptOps.GROUPTOP), insert_before=self.first_reduce + self.group_for_reduces)
      self.group_for_reduces += 1
    elif opt.op is OptOps.UNROLL:                     # purple
      check(axis < self.shape_len-self.upcasted, "can't upcasted already upcasted")
      check(amt <= 32, "don't unroll more than 32")
      # TODO: fix upcast_count to put purples before yellows. broken because of METAL tensor cores
      #upcast_count = sum(x == y for x,y in zip(self.full_shape[-self.upcasted:], self.output_shape[-self.upcasted:])) if self.upcasted else 0
      #self.shift_to(axis, amt, insert_before=None if upcast_count == 0 else self.shape_len-upcast_count)
      if self.full_shape[axis] == amt and axis == self.first_reduce: self.local_dims += 1 # first_reduce will ++, so offset loss in simplify_ones
      if self.full_shape[axis] == amt and axis < self.first_reduce+self.group_for_reduces: self.group_for_reduces -= 1 # fully unrolling a GROUP
      self.shift_to(axis, amt, insert_before=None)
      self.upcast()
    elif opt.op is OptOps.UPCAST:                     # yellow
      check(axis < self.first_reduce, "upcast is for non-reduce")
      check(not(self.tensor_core and self.global_dims <= axis < self.global_dims+len(self.tensor_core.threads)), "can't upcast TC locals")
      check(amt <= 8, "don't upcast more than 8")
      self.shift_to(axis, amt, insert_before=None)
      self.upcast()
    elif opt.op is OptOps.UPCASTMID:                  # white
      check(self.bufs[0].dtype.name.startswith('image') and not self.float4_axis(0) and self.group_for_reduces != 0 and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1, "invalid upcast mid reduce")  # noqa: E501
      axes = self.sts[0].unit_stride_axes()
      check(len(axes) == 1, f"wrong number of stride 1 axis : {axes}")
      check(axes[0] == axis, "wrong axis")
      check(amt == 4, "don't upcast mid anything but 4")
      self.shift_to(axis, amt, insert_before=self.first_reduce + self.group_for_reduces)
      self.group_for_reduces += 1
    elif opt.op is OptOps.NOLOCALS:
      check(self.opts.has_local and not self.dont_use_locals, "NOLOCALS is meaningless if target does not support local or already not using locals")
      check(self.local_dims == 0 and self.group_for_reduces == 0, "can't have no locals with locals")
      self.dont_use_locals = True
    elif opt.op is OptOps.MERGE:
      check(axis >= self.shape_len-self.upcasted, "only merge upcasted")
      check(self.full_shape[axis:axis+2] == self.output_shape[axis:axis+2], "can't merge reduces")
      self.reshape_and_permute(None, tuple(range(axis)) + (axis+1, axis) + tuple(range(axis+2, self.shape_len)))
      self.reshape_and_permute(lambda x: x[0:axis] + (x[axis] * x[axis+1],) + x[axis+2:], None)
      self.upcasted -= 1
    elif opt.op is OptOps.PADTO:
      check(not self.vars, "does not work with symbolic shape")
      check(axis < self.shape_len - self.upcasted, "cannot pad upcasted")
      # ok to pad SUM if all parent ops have f(0) = 0
      if self.first_reduce <= axis:
        check((r:=cast(LazyOp, self.reduceop)).op is ReduceOps.SUM and \
            all(op.op not in UNSAFE_PAD_OPS for sop in r.src for op in sop.lazyops), "cannot pad")
      padded = False
      for i,st in enumerate(self.sts):
        if self.sts[i].shape[axis] == 1: continue  # reduced
        check(self.sts[i].shape[axis] > amt//4, f"pad adds more than quadruple the work {self.sts[i].shape[axis]=} > {amt//4=}")
        if (ru := round_up(cast(int, self.sts[i].shape[axis]), cast(int, amt)) - self.sts[i].shape[axis]):
          # pad right seems to be faster
          self.sts[i] = st.pad(((0,0),) * axis + ((0,ru),) + ((0,0),) * (len(st.shape)-axis-1))
          padded = True
      check(padded, "nothing was padded")

    if append_opt: self.applied_opts.append(opt)
    if self.simplify_ones() and self.tensor_core_opts:
      self.tensor_core_opts.fix_axes(axis) # fix up axes in TC opts if required after simplify_ones()

  def required_optimizations(self):
    if self.bufs[0].dtype.__class__ is ImageDType:
      unit_stride_axes_mul_4 = [i for i in self.sts[0].unit_stride_axes(ignore_valid=True) if self.sts[0].shape[i]%4 == 0]
      assert len(unit_stride_axes_mul_4) >= 1, f"needs a unit stride axis in {self.bufs[0]}"
      if len(unit_stride_axes_mul_4) and all(x < (self.shape_len-self.upcasted) for x in unit_stride_axes_mul_4) and unit_stride_axes_mul_4[0] not in self.upcast_in_mid_reduce_axes:  # noqa: E501
        self.apply_opt(Opt(OptOps.UPCAST, unit_stride_axes_mul_4[0], 4))

  def hand_coded_optimizations(self):
    self.required_optimizations()

    # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
    MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
    if self.opts.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and  \
        self.reduceop is not None and self.reduceop.op is ReduceOps.SUM and len(self.full_shape) >= 2 and self.opts.has_shared and \
        (mulop:=self.reduceop.src[0]).op is BinaryOps.MUL and mulop.src[0].op is BufferOps.LOAD and mulop.src[1].op is BufferOps.LOAD:
      st0, st1 = self.sts[self.bufs.index(mulop.src[0].arg)], self.sts[self.bufs.index(mulop.src[1].arg)]
      strides0, strides1 = st0.real_strides(), st1.real_strides()
      def has_expanded_axis(shape, strides): return any(s > 1 and st == 0 for s,st in zip(shape,strides))
      if strides0[self.first_reduce] == 1 and not (has_expanded_axis(st0.shape, strides0) and has_expanded_axis(st1.shape, strides1)):
        for global_idx in range(self.global_dims):
          if self.full_shape[self.first_reduce]%MV_THREADS_PER_ROW == 0 and self.full_shape[global_idx]%(MV_BLOCKSIZE*MV_ROWS_PER_THREAD) == 0:
            if DEBUG >= 3:
              print(f"MATVEC: {self.full_shape=} {self.first_reduce=} {strides0=} {MV_BLOCKSIZE=} {MV_THREADS_PER_ROW=} {MV_ROWS_PER_THREAD=}")
            if MV_THREADS_PER_ROW > 1: self.apply_opt(Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
            if MV_BLOCKSIZE > 1: self.apply_opt(Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
            if MV_ROWS_PER_THREAD > 1: self.apply_opt(Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
            return

    if self.opts.has_local and self.opts.has_shared and all_int(self.sts[0].shape[:self.first_reduce]):
      # are we grouping? (requires local shape support)
      if not self.float4_axis(0) and self.first_reduce <= 2 and self.first_reduce + 1 <= self.shape_len and prod(self.sts[0].shape[:self.first_reduce]) <= 2048:  # noqa: E501
        # TODO: use 1024 if it's allowed in a smarter way
        for sz in (([256, 16]) if prod(self.sts[0].shape[:self.first_reduce]) <= 32 else [16]):
          if all(st.shape[self.first_reduce] % sz == 0 or st.shape[self.first_reduce] == 1 for st in self.sts):
            try: # may fail due to excessive smem usage
              self.apply_opt(Opt(OptOps.GROUPTOP, 0, sz))
              break
            except KernelOptError: pass

      # are we upcasting in mid reduce? (only for images)
      if self.bufs[0].dtype.name.startswith('image') and not self.float4_axis(0) and self.group_for_reduces and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1:  # noqa: E501
        axes = self.sts[0].unit_stride_axes()
        assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
        if self.sts[0].shape[axes[0]]%4 == 0:
          self.apply_opt(Opt(OptOps.UPCASTMID, axes[0], 4))

    # upcast float4 images
    for buf_index,buf in enumerate(self.bufs):
      unit_stride_axes_mul_4 = [i for i in self.sts[buf_index].unit_stride_axes(ignore_valid=True) if self.sts[buf_index].shape[i]%4 == 0]
      if buf.dtype.__class__ is ImageDType:
        #assert len(unit_stride_axes_mul_4) >= 1, f"needs a unit stride axis in {self.bufs[buf_index]}"
        if len(unit_stride_axes_mul_4) and all(x < (self.shape_len-self.upcasted) for x in unit_stride_axes_mul_4) and unit_stride_axes_mul_4[0] not in self.upcast_in_mid_reduce_axes:  # noqa: E501
          if unit_stride_axes_mul_4[0] < self.first_reduce:
            self.apply_opt(Opt(OptOps.UPCAST, unit_stride_axes_mul_4[0], 4))
          else:
            self.apply_opt(Opt(OptOps.UNROLL, unit_stride_axes_mul_4[0]-self.first_reduce, 4))

    # no more opt if we are grouping
    if self.group_for_reduces: return

    # **** below this line need to be optional and benchmarked ****

    # TODO: doing extra upcasts with images doesn't work for some reason (maybe has to do with to_image_idx)
    # to trigger the above bug, remove prod(self.full_shape[self.shape_len - self.upcasted:]) from the below
    # expression and run test/test_ops.py with IMAGE=2
    # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
    # this can be made much smarter
    to_upcast: List[int] = []
    # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
    for axis in range(self.first_reduce):
      # we might want to be able to split axes that are masked, or refuse to merge them in simplify_merge_adjacent
      # for now skip upcasting here if there is a symbolic axis
      if isinstance(self.full_shape[axis], int) and self.full_shape[axis] <= 7 and any(st.axis_is_masked(axis) for st in self.sts) and \
        prod(self.full_shape[self.shape_len - self.upcasted:]) * prod(self.full_shape[j] for j in to_upcast) * self.full_shape[axis] <= 7 * 7:
        if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
        to_upcast.append(axis)
    for axis in to_upcast[::-1]: self.apply_opt(Opt(OptOps.UPCAST, axis, 0))

    # potentially do more upcasts of non reduce axes based on a heuristic
    upcasted_axis = set()
    while prod(self.sts[0].shape[:self.first_reduce]) >= 1024:
      xb_choices = []
      for axis, upcast_amount in itertools.product(range(self.first_reduce), [3,4]):   # consider all the non reduce axes, and a 3 or 4 reduce
        # if we haven't upcasted it, it's not symbolic, it mods, and buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
        if axis not in upcasted_axis and isinstance(self.full_shape[axis], int) and self.full_shape[axis]%upcast_amount == 0 and any(st.views[-1].strides[axis] == 0 and not any(x[1] == 0 for x in self.upcasted_axis(buf_index)) for buf_index, st in enumerate(self.sts)):  # noqa: E501
          xb_choices.append((sum(st.views[-1].strides[axis]>0 for st in self.sts), sum(st.views[-1].strides[axis] for st in self.sts), axis, upcast_amount))  # noqa: E501
      if xb_choices:
        xb_choices = sorted(xb_choices)
        if DEBUG >= 4: print(f"float4 merging axis : {xb_choices}")
        self.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][2], xb_choices[0][3]))
        upcasted_axis.add(xb_choices[0][2])
      else: break

    # if last dim is small(ish) and it's a reduce dim, upcast the reduce (loop unrolling). no simplify needed since it's just an upcast.
    if self.first_reduce < (self.shape_len-self.upcasted) and (len(list(self.shape_offsets(self.full_buf_index))) <= 4 or not any(r for _,_,r in self.upcasted_axis(self.full_buf_index))) and (self.upcasted == 0 or prod(self.full_shape[-self.upcasted:]) < 64):  # noqa: E501
      if (s:=self.full_unupcasted_shape[-1]) <= 32 and isinstance(s, int):  # NOTE: cannot loop unroll symbolic axis
        self.apply_opt(Opt(OptOps.UNROLL, len(self.full_unupcasted_shape)-1-self.first_reduce, 0))
        # if it's small, upcast a second reduce dimension too
        if self.first_reduce < (self.shape_len-self.upcasted) and s <= 3 and (s2:=self.full_unupcasted_shape[-1]) <= 3 and isinstance(s2, int):
          self.apply_opt(Opt(OptOps.UNROLL, len(self.full_unupcasted_shape)-1-self.first_reduce, 0))
      else:
        for splits in [4]:
          if self.full_unupcasted_shape[-1]%splits == 0:
            self.apply_opt(Opt(OptOps.UNROLL, len(self.full_unupcasted_shape)-1-self.first_reduce, splits))
            break

    # if nothing at all is upcasted and it's easy to, do an upcast
    # TODO: this is breaking the tests
    for splits in [4]:
      if self.upcasted == 0 and self.full_unupcasted_shape and self.full_unupcasted_shape[-1] % splits == 0:
        self.apply_opt(Opt(OptOps.UPCAST, len(self.full_unupcasted_shape)-1, splits))

    # **** local groups ****

    if self.opts.has_local:
      if getenv("NOLOCALS") and self.local_dims == 0 and not self.group_for_reduces:
        self.apply_opt(Opt(OptOps.NOLOCALS))
      else:
        # prioritize making expand axes local
        local_axis_ranking = [(any(self.sts[buf_index].views[-1].strides[axis] == 0 for buf_index in range(len(self.sts))), axis) for axis in range(len(self.full_shape[:self.first_reduce]))]  # noqa: E501
        to_local: List[Tuple[int, int]] = []
        for _, axis in sorted(local_axis_ranking, key=lambda x: (-x[0], -x[1])):
          local_size = prod(sz for _, sz in to_local)
          local_sz: Optional[int] = next((x for x in ([32] * (axis == 0) + [16, 8, 4, 3, 2]) if self.full_shape[axis] % x == 0 and local_size * x <= 128), None)  # noqa: E501
          if local_sz is not None: to_local.append((axis, local_sz))
        deleted_shape = 0
        for axis, local_sz in sorted(to_local[:3]):
          axis = axis - deleted_shape
          will_delete_shape = local_sz == self.full_shape[axis]
          self.apply_opt(Opt(OptOps.LOCAL, axis, local_sz))
          if will_delete_shape: deleted_shape += 1

  # **** kernel outputs ****

  kernel_cnt: Final[DefaultDict[str, int]] = defaultdict(int)
  @functools.cached_property
  def name(self) -> str:
    # kernel name (before late upcast)
    name = ("r" if self.reduceop else ("C" if all(x.op in BufferOps for x in self.ast.lazyops) else "E")) + \
                 (f"{len(self.ast.src)}_" if len(self.ast.src) > 1 else "_") + \
                 colored('_', 'BLACK').join([colored(str(x), c) for x,c in zip(self.full_shape, self.colors())])

    # name the function something unique
    Kernel.kernel_cnt[(function_name := to_function_name(name))] += 1
    suffix = f"{'n'+str(Kernel.kernel_cnt[function_name]-1)}" if Kernel.kernel_cnt[function_name] > 1 else ""
    return name+colored(suffix, 'BLACK')

  def get_optimized_ast(self) -> LazyOp:
    # set the shapetrackers to the optimized ones, fixup reduceop
    # transformed to the final LazyOp
    @functools.lru_cache(None)
    def fixup_ast(op:LazyOp, apply_to_st=None) -> LazyOp:
      if op.op in BufferOps:
        idx = self.bufs.index(op.arg)
        arg = replace(op.arg, st=self.sts[idx] if apply_to_st is None else apply_to_st(self.sts[idx]))
      elif op.op in ReduceOps:
        reduce_idx = len(self.bufs) + self.reduceops.index(op)*2
        arg = tuple(i for i in range(self.first_reduce+self.group_for_reduces, self.shape_len)
                    if self.sts[reduce_idx].shape[i] != self.sts[reduce_idx+1].shape[i])
        if op in self.bufs_for_tensor_core and (tc := self.tensor_core):
          rsrc = op.src[0]
          if rsrc.op is UnaryOps.CAST: rsrc = rsrc.src[0]
          assert rsrc.op is BinaryOps.MUL

          def fix_st(warp_dims, tcd_dims, tcd_expand, pattern_1, pattern_2, st1):
            wd = self.global_dims
            tcd = self.shape_len-self.upcasted
            assert st1.shape[wd:wd+len(warp_dims)] == warp_dims, f"warp dims wrong: {st1.shape[wd:wd+len(warp_dims)]=} != {warp_dims=}"
            assert st1.shape[tcd:tcd+len(tcd_dims)] == tcd_dims, f"tcd dims wrong: {st1.shape[tcd:tcd+len(tcd_dims)]=} != {tcd_dims=}"
            new_shape = st1.shape[:tcd] + tcd_expand + st1.shape[tcd+len(tcd_dims):]  # expand the tcd
            permaxis = list(range(wd))
            permaxis += [y + (wd if x == 0 else tcd) for x,y in pattern_1]
            permaxis += list(range(wd+len(warp_dims), tcd))
            permaxis += [y + (wd if x == 0 else tcd) for x,y in pattern_2]
            permaxis += list(range(tcd+len(tcd_expand), len(new_shape)))
            return st1.reshape(new_shape).simplify().permute(tuple(permaxis)).reshape(st1.shape).simplify()

          if self.opts.device == "AMD":
            reduce_axes = [self.shape_len-self.upcasted]
            upcast_axis = (self.shape_len-self.upcasted, self.shape_len-self.upcasted, self.shape_len-self.upcasted+1)
            fix_st1 = functools.partial(fix_st, (8,2,2), (16,8), (16,2,4), ((1,2), (0,2), (1,1), (0,1)), ((1,0), (0,0)))
            fix_st2 = None
          elif self.opts.device == "METAL":
            reduce_axes = [self.shape_len-self.upcasted]
            upcast_axis = (self.shape_len-self.upcasted+1, self.shape_len-self.upcasted+1, self.shape_len-self.upcasted+1)
            fix_st1 = functools.partial(fix_st, (2,4,2,2), (8,2), (2,2,2,2), ((1,1), (0,1), (1,0), (0,3)), ((0,0), (0,2), (1,3), (1,2)))
            fix_st2 = functools.partial(fix_st, (2,4,2,2), (8,2), (2,2,2,2), ((0,0), (1,1), (1,2), (0,2), (1,0)), ((0,1), (0,3), (1,3)))
          elif self.opts.device in {"CUDA", "NV"}:
            reduce_axes = [self.shape_len-self.upcasted, self.shape_len-self.upcasted+1]
            upcast_axis = (self.shape_len-self.upcasted, self.shape_len-self.upcasted+2, self.shape_len-self.upcasted+2)
            # https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
            fix_st1 = functools.partial(fix_st, (2,2,2,2,2), (8,2,4), (2,2,2,2,2,2),
              ((1,1), (1,0), (0,2), (0,3), (0,4)), ((1,3), (1,4), (1,2), (0,0), (0,1), (1,5)))
            fix_st2 = functools.partial(fix_st, (2,2,2,2,2), (8,2,4), (2,2,2,2,2,2),
              ((1,1), (1,0), (1,5), (0,0), (0,1)), ((0,4), (0,2), (1,4), (0,3), (1,3), (1,2)))
          else:
            raise RuntimeError("unsupported device for tensor cores")

          assert apply_to_st is None, "double tensor core? not supported"
          wmma_sz = [prod(l) for l in tc.thread_local_sizes]
          wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, tuple(wmma_sz), self.opts.device, upcast_axis, tuple(reduce_axes))
          ret = LazyOp(ReduceOps.WMMA, (fixup_ast(rsrc.src[0], fix_st1), fixup_ast(rsrc.src[1], fix_st2)), wmma_arg)
          new_reduce_axes = tuple(i for i in arg if i not in reduce_axes)
          return LazyOp(op.op, (ret,), new_reduce_axes) if new_reduce_axes else ret
        if self.group_for_reduces:
          start = LazyOp(op.op, tuple(fixup_ast(x, apply_to_st) for x in op.src), arg)
          local_shape = (1,) * self.global_dims + self.full_shape[self.global_dims:self.global_dims+self.local_dims+self.group_for_reduces] + \
            (1,) * (self.shape_len - self.upcasted - self.group_for_reduces - self.first_reduce) + tuple([x[0] for x in self.upcasted_axis(0)])
          local_buffer = MemBuffer(-1, start.dtype, ShapeTracker.from_shape(local_shape))
          local_store = LazyOp(BufferOps.STORE, (start,), local_buffer)
          local_load = LazyOp(BufferOps.LOAD, (local_store,), local_buffer)
          return LazyOp(op.op, (local_load,), tuple(range(self.first_reduce, self.first_reduce+self.group_for_reduces)))
      elif op.op is MetaOps.KERNEL:
        arg = KernelInfo(self.local_dims, self.upcasted)
      else:
        arg = op.arg
      return LazyOp(op.op, tuple(fixup_ast(x, apply_to_st) for x in op.src), arg)
    return fixup_ast(self.ast)

  # **** this is the lowerer ****

  def linearize(self) -> Kernel:
    modified_ast = self.get_optimized_ast()

    if DEBUG >= 3:
      print(self.name)
      print_tree(modified_ast)
    verify_lazyop(modified_ast)

    uop_sink = lazyop_to_uop(modified_ast, self.opts)

    # extract global/local sizes
    if self.opts.has_local:
      self.global_size: Optional[List[int]] = [1,1,1]
      self.local_size: Optional[List[int]] = [1,1,1]
      for u in uop_sink.parents:
        if u.op is UOps.SPECIAL:
          if u.arg[1][0] == 'l': self.local_size[u.arg[0]] = u.arg[2]
          else: self.global_size[u.arg[0]] = u.arg[2]
    else:
      self.global_size, self.local_size = None, None

    # generate the UOpGraph
    self.uops:UOpGraph = UOpGraph(uop_sink, self.opts)
    if DEBUG >= 5: self.uops.print()
    if getenv("GRAPHUOPS"):
      self.uops.graph()
      if getenv("GRAPHUOPS") == 2: exit(0)
    return self

  def to_program(self) -> Program:
    self.linearize()
    src = self.opts.render(name:=to_function_name(self.name), self.uops)
    if getenv("RUN_PROCESS_REPLAY"):
      table_name = f"process_replay_{getenv('GITHUB_RUN_ID', 'HEAD')}"
      diskcache_put(table_name, id(self), (self.ast, self.opts, self.applied_opts, name, src, {k:v.value for k,v in ContextVar._cache.items()}))
    info = get_lazyop_info(self.ast.src[0])   # TODO: this should be removed
    ops, mem = flops_mem(self.uops.uops)
    run_count = prod((self.global_size or []) + (self.local_size or []))
    return Program(self.name, src, self.opts.device, self.global_size, self.local_size,
                   self.uops, min(info.flops, ops * run_count), min(info.mem_estimate, mem * run_count))
