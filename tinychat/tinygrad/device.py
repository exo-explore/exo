from __future__ import annotations
import multiprocessing
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Optional, Dict, Tuple, Any, cast, Protocol, Type
import importlib, inspect, functools, pathlib, os, ctypes, atexit, time, contextlib, array
from tinygrad.helpers import getenv, diskcache_get, diskcache_put, DEBUG, GlobalCounters, flat_mv, from_mv, ProfileLogger, PROFILE
from tinygrad.dtype import DType, ImageDType
from tinygrad.renderer import Renderer

# **************** Device ****************

class _Device:
  def __init__(self) -> None: self._devices: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]  # noqa: E501
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def _canonicalize(self, device:str) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "")   # noqa: E501
  # NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  def canonicalize(self, device:Optional[str]) -> str: return self._canonicalize(device) if device is not None else Device.DEFAULT
  def __getitem__(self, ix:str) -> Compiled: return self.__get_canonicalized_item(self.canonicalize(ix))
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __get_canonicalized_item(self, ix:str) -> Compiled:
    assert ((cpn:=multiprocessing.current_process().name) == "MainProcess") or ix.split(":")[0] in ["DISK", "NPY"], \
      f"can only open device {ix} from parent, not {cpn}"
    x = ix.split(":")[0].upper()
    ret = [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "device") and x in self._devices][0](ix)  # noqa: E501
    if DEBUG >= 1: print(f"opened device {ix} from pid:{os.getpid()}")
    return ret
  @functools.cached_property
  def DEFAULT(self) -> str:
    device_from_env: Optional[str] = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._devices, None)   # type: ignore
    if device_from_env: return device_from_env
    for device in ["METAL", "AMD", "NV", "CUDA", "GPU", "CLANG", "LLVM"]:
      try:
        if self[device]:
          os.environ[device] = "1"   # we set this in environment for spawned children
          return device
      except Exception: pass
    raise RuntimeError("no usable devices")
Device = _Device()

# **************** Buffer + Allocators ****************

@dataclass(frozen=True, eq=True)
class BufferOptions:
  image: Optional[ImageDType] = None
  uncached: bool = False
  cpu_access: bool = False
  host: bool = False
  nolru: bool = False

class Buffer:
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:Optional[BufferOptions]=None,
               initial_value:Optional[bytes]=None, lb_refcount=0, base:Optional[Buffer]=None, offset:int=0, preallocate=False):
    assert isinstance(dtype, DType)
    if isinstance(dtype, ImageDType): options = BufferOptions(image=dtype) # TODO: image hack shouldn't be here. where should it be?
    self.device, self.size, self.dtype, self.options, self.offset = device, size, dtype, options, offset
    if base is None:
      assert offset == 0, "base buffers can't have offset"
      self._base = None
      self._lb_refcount = lb_refcount
      if opaque is not None: self.allocate(opaque)
      if initial_value is not None:
        self.allocate()
        self.copyin(memoryview(initial_value))
    else:
      assert base._base is None, "base can't have a base"
      assert device == base.device, "base must have the same device"
      self._base = base
    if preallocate: self.allocate()
  @property
  def base(self) -> Buffer: return self._base if self._base is not None else self
  @property
  def lb_refcount(self): return self.base._lb_refcount
  def ref(self, cnt): self.base._lb_refcount += cnt
  def is_allocated(self) -> bool: return hasattr(self, '_buf')
  def ensure_allocated(self) -> Buffer: return self.allocate() if not hasattr(self, '_buf') else self
  def allocate(self, opaque=None) -> Buffer:
    assert not hasattr(self, '_buf'), "can't allocate already allocated buffer"
    self.allocator = Device[self.device].allocator
    if self._base is not None:
      self._base.ensure_allocated()
      assert hasattr(self.allocator, "offset"), "offset function required for view"
      self._buf: Any = self.allocator.offset(self.base._buf, self.nbytes, self.offset)
    else:
      self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
      if not self.device.startswith("DISK"): GlobalCounters.mem_used += self.nbytes
    return self
  def __reduce__(self):
    buf = None
    if self._base is not None:
      return self.__class__, (self.device, self.size, self.dtype, None, None, None, 0, self.base, self.offset, hasattr(self, '_buf'))
    if self.device == "NPY": return self.__class__, (self.device, self.size, self.dtype, self._buf, self.options, None, self.lb_refcount)
    if self.is_allocated():
      buf = bytearray(self.nbytes)
      self.copyout(memoryview(buf))
    return self.__class__, (self.device, self.size, self.dtype, None, self.options, buf, self.lb_refcount)
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  def __del__(self):
    if not hasattr(self, '_buf'): return
    if self._base is None:
      if not self.device.startswith("DISK"): GlobalCounters.mem_used -= self.nbytes
      self.allocator.free(self._buf, self.nbytes, self.options)
  def __repr__(self):
    return f"<buf real:{hasattr(self, '_buf')} device:{self.device} size:{self.size} dtype:{self.dtype}" + \
           (f" offset:{self.offset}" if hasattr(self, "base") else "") + \
           (">" if self.options is None else f" {self.options=}>")
  def as_buffer(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_buffer (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, 'as_buffer'): return self.allocator.as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.nbytes)))
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_allocated(), "can't copyin to unallocated buffer"
    self.allocator.copyin(self._buf, mv)
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_allocated(), "can't copyout unallocated buffer"
    self.allocator.copyout(mv, self._buf)
    return mv
  def view(self, size:int, dtype:DType, offset:int) -> Buffer:
    assert offset < self.nbytes, "offset must be less than nbytes"
    if self._base is not None: return Buffer(self.device, size, dtype, base=self._base, offset=self.offset+offset)
    return Buffer(self.device, size, dtype, base=self, offset=offset)

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator:
  def alloc(self, size:int, options:Optional[BufferOptions]=None):
    assert not isinstance(size, int) or size > 0, f"alloc size must be positve, getting {size}"
    return self._alloc(size, options if options is not None else BufferOptions())
  def _alloc(self, size:int, options:BufferOptions): raise NotImplementedError("need alloc")
  def free(self, opaque, size:int, options:Optional[BufferOptions]=None):
    self._free(opaque, options if options is not None else BufferOptions())
  def _free(self, opaque, options:BufferOptions): pass  # if opaque is a Python object, you don't need a free
  def copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")

class LRUAllocator(Allocator):  # pylint: disable=abstract-method
  """
  The LRU Allocator is responsible for caching buffers.
  It ensures that buffers are not freed until it is absolutely necessary, optimizing performance.
  """
  def __init__(self): self.cache: Dict[Tuple[int, Optional[BufferOptions]], Any] = defaultdict(list)
  def alloc(self, size:int, options:Optional[BufferOptions]=None):
    if len(c := self.cache[(size, options)]): return c.pop()
    try: return super().alloc(size, options)
    except (RuntimeError, MemoryError):
      self.free_cache()
      return super().alloc(size, options)
  def free_cache(self):
    for (sz,options),opaques in self.cache.items():
      for opaque in opaques: super().free(opaque, sz, options)
      opaques.clear()
  def free(self, opaque:Any, size:int, options:Optional[BufferOptions]=None):
    if getenv("LRU", 1) and (options is None or not options.nolru): self.cache[(size, options)].append(opaque)
    else: super().free(opaque, size, options)

class _MallocAllocator(LRUAllocator):
  def _alloc(self, size:int, options:BufferOptions): return (ctypes.c_uint8 * size)()
  def as_buffer(self, src) -> memoryview: return flat_mv(memoryview(src))
  def copyin(self, dest, src:memoryview): ctypes.memmove(dest, from_mv(src), len(src))
  def copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src, len(dest))
  def offset(self, buf, size:int, offset:int): return from_mv(self.as_buffer(buf)[offset:offset+size])

MallocAllocator = _MallocAllocator()

# **************** for Compiled Devices ****************

class CompileError(Exception): pass

class Compiler:
  def __init__(self, cachekey:Optional[str]=None): self.cachekey = None if getenv("DISABLE_COMPILER_CACHE") else cachekey
  def compile(self, src:str) -> bytes: raise NotImplementedError("need a compile function")
  def compile_cached(self, src:str) -> bytes:
    if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
      assert not getenv("ASSERT_COMPILE"), f"tried to compile with ASSERT_COMPILE set\n{src}"
      lib = self.compile(src)
      if self.cachekey is not None: diskcache_put(self.cachekey, src, lib)
    return lib

class Compiled:
  def __init__(self, device:str, allocator:Allocator, renderer:Optional[Renderer], compiler:Optional[Compiler], runtime, graph=None):
    self.dname, self.allocator, self.compiler, self.runtime, self.graph = device, allocator, compiler or Compiler(), runtime, graph
    self.renderer = renderer or Renderer()
  def synchronize(self):
    """
    Synchronize all pending operations on the device.

    This method ensures that all previously queued operations on the device have been completed before proceeding.
    """
    # override this in your device implementation

# **************** for HCQ Compatible Devices ****************

def hcq_command(func):
  """
  Decorator for HWCommandQueue commands. Enables command indexing and stores metadata for command updates.

  For example:
    ```python
      @hcq_command
      def command_method(self, ...): ...
    ```
  """
  def __wrapper(self, *args, **kwargs):
    self.cmds_offset.append(len(self.q))
    func(self, *args, **kwargs)
    self.cmds_len.append(len(self.q) - self.cmds_offset[-1])
    self.cmds_meta.append(func.__name__)
    return self
  return __wrapper

class HWCommandQueue:
  """
  A base class for hardware command queues in the HCQ (Hardware Command Queue) API.
  Both compute and copy queues should have the following commands implemented.
  """

  def __init__(self): self.q, self.binded_device, self.cmds_offset, self.cmds_len, self.cmds_meta = [], None, [], [], []
  def __len__(self): return len(self.cmds_offset)
  def _patch(self, cmd_idx, offset, data): self.q[(st:=self.cmds_offset[cmd_idx]+offset):st+len(data)] = array.array('I', data)

  @hcq_command
  def signal(self, signal:Any, value:int):
    """
    Enqueues a signal command which sets the signal to the given value, ensuring all previous operations are completed.

    Args:
      signal: The signal to set
      value: The value to set the signal to
    """
    self._signal(signal, value)
  def _signal(self, signal:Any, value:int): raise NotImplementedError("backend should overload this function")

  @hcq_command
  def wait(self, signal:Any, value:int):
    """
    Enqueues a wait command which halts execution until the signal is greater than or equal to a specific value.

    Args:
      signal: The signal to wait on
      value: The value to wait for
    """
    self._wait(signal, value)
  def _wait(self, signal, value): raise NotImplementedError("backend should overload this function")

  @hcq_command
  def timestamp(self, signal:Any):
    """
    Enqueues a timestamp command which records the current time in a signal after all previously enqueued commands are completed.

    Args:
      signal: The signal to store the timestamp
    """
    self._timestamp(signal)
  def _timestamp(self, signal): raise NotImplementedError("backend should overload this function")

  def update_signal(self, cmd_idx:int, signal:Optional[Any]=None, value:Optional[int]=None):
    """
    Updates a previously queued signal command.

    Args:
      cmd_idx: Index of the signal command to update
      signal: New signal to set (if None, keeps the original)
      value: New value to set (if None, keeps the original)
    """
    if self.cmds_meta[cmd_idx] != "signal": raise RuntimeError("called update_signal not on a signal command")
    self._update_signal(cmd_idx, signal, value)
    return self
  def _update_signal(self, cmd_idx:int, signal:Optional[Any], value:Optional[int]): raise NotImplementedError("backend should overload this function")

  def update_wait(self, cmd_idx:int, signal:Optional[Any]=None, value:Optional[int]=None):
    """
    Updates a previously queued wait command.

    Args:
      cmd_idx: Index of the wait command to update
      signal: New signal to wait on (if None, keeps the original)
      value: New value to wait for (if None, keeps the original)
    """
    if self.cmds_meta[cmd_idx] != "wait": raise RuntimeError("called update_wait not on a wait command")
    self._update_wait(cmd_idx, signal, value)
    return self
  def _update_wait(self, cmd_idx:int, signal:Optional[Any], value:Optional[int]): raise NotImplementedError("backend should overload this function")

  def submit(self, device:HCQCompatCompiled):
    """
    Submits the command queue to a specific device for execution.

    Args:
      device: The device to submit the queue to
    """
    self._submit(device)
    return self
  def _submit(self, device:HCQCompatCompiled): raise NotImplementedError("backend should overload this function")

class HWComputeQueue(HWCommandQueue):
  @hcq_command
  def memory_barrier(self):
    """
    Enqueues a memory barrier command to ensure memory coherence between agents.
    """
    self._memory_barrier()
  def _memory_barrier(self): pass

  @hcq_command
  def exec(self, prg:HCQCompatProgram, kernargs:int, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int]):
    """
    Enqueues an execution command for a kernel program.

    Args:
      prg: The program to execute
      kernargs: The pointer to kernel arguments
      global_size: The global work size
      local_size: The local work size
    """
    self._exec(prg, kernargs, global_size, local_size)
  def _exec(self, prg, kernargs, global_size, local_size): raise NotImplementedError("backend should overload this function")

  def update_exec(self, cmd_idx:int, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int]):
    """
    Updates a previously queued execution command.

    Args:
      cmd_idx: Index of the execution command to update
      global_size: New global work size
      local_size: New local work size
    """
    if self.cmds_meta[cmd_idx] != "exec": raise RuntimeError("called update_exec not on an exec command")
    self._update_exec(cmd_idx, global_size, local_size)
    return self
  def _update_exec(self, cmd_idx, global_size, local_size): raise NotImplementedError("backend should overload this function")

class HWCopyQueue(HWCommandQueue):
  @hcq_command
  def copy(self, dest:HCQCompatAllocRes, src:HCQCompatAllocRes, copy_size:int):
    """
    Enqueues a copy command to transfer data.

    Args:
      dest: The destination of the copy
      src: The source of the copy
      copy_size: The size of data to copy
    """
    self._copy(dest, src, copy_size)
  def _copy(self, dest:HCQCompatAllocRes, src:HCQCompatAllocRes, copy_size:int): raise NotImplementedError("backend should overload this function")

  def update_copy(self, cmd_idx:int, dest:Optional[HCQCompatAllocRes]=None, src:Optional[HCQCompatAllocRes]=None):
    """
    Updates a previously queued copy command.

    Args:
      cmd_idx: Index of the copy command to update
      dest: New destination of the copy (if None, keeps the original)
      src: New source of the copy (if None, keeps the original)
    """
    if self.cmds_meta[cmd_idx] != "copy": raise RuntimeError("called update_copy not on an copy command")
    self._update_copy(cmd_idx, dest, src)
    return self
  def _update_copy(self, cmd_idx, dest, src): raise NotImplementedError("backend should overload this function")

@contextlib.contextmanager
def hcq_profile(dev, enabled, desc, queue_type=None, queue=None):
  st, en = (dev._alloc_signal(), dev._alloc_signal()) if enabled else (None, None)

  if enabled and queue is not None: queue.timestamp(st)
  elif enabled: queue_type().timestamp(st).submit(dev)

  try: yield (st, en)
  finally:
    if enabled and queue is not None: queue.timestamp(en)
    elif enabled: queue_type().timestamp(en).submit(dev)

    if enabled and PROFILE: dev.sig_prof_records.append((st, en, desc, queue_type is dev.hw_copy_queue_t))

class HCQCompatProgram:
  def __init__(self, kernargs_alloc_size:int, kernargs_args_offset:int=0):
    self.kernargs_alloc_size, self.kernargs_args_offset = kernargs_alloc_size, kernargs_args_offset
  def fill_kernargs(self, kernargs_ptr:int, bufs:Tuple[Any, ...], vals:Tuple[int, ...]=()): raise NotImplementedError("need fill_kernargs")

class HCQCompatCompiled(Compiled):
  """
  A base class for devices compatible with the HCQ (Hardware Command Queue) API.
  """

  def __init__(self, device:str, allocator:Allocator, renderer:Renderer, compiler:Compiler, runtime,
               comp_queue_t:Type[HWComputeQueue], copy_queue_t:Type[HWCopyQueue], timeline_signals:Tuple[Any, Any]):
    self.hw_compute_queue_t, self.hw_copy_queue_t = comp_queue_t, copy_queue_t
    self.timeline_value:int = 1
    self.timeline_signal, self._shadow_timeline_signal = timeline_signals
    self.sig_prof_records:List[Tuple[Any, Any, str, bool]] = []
    self.raw_prof_records:List[Tuple[int, int, str, bool]] = []
    if PROFILE: self._prof_setup()

    from tinygrad.runtime.graph.hcq import HCQGraph
    super().__init__(device, allocator, renderer, compiler, runtime, HCQGraph)

  @classmethod
  def _read_signal(cls, signal:Any) -> int:
    """
    Read a value for a signal.
    """
    raise NotImplementedError("_read_signal needs to be implemented")

  @classmethod
  def _read_timestamp(cls, signal:Any) -> int:
    """
    Read a timestamp for a signal.
    """
    raise NotImplementedError("_read_timestamp needs to be implemented")

  @classmethod
  def _set_signal(cls, signal:Any, value:int):
    """
    Set a value for a signal.
    """
    raise NotImplementedError("_set_signal needs to be implemented")

  @classmethod
  def _alloc_signal(cls, value:int = 0, **kwargs) -> Any:
    """
    Allocate a new signal.
    """
    raise NotImplementedError("_alloc_signal needs to be implemented")

  @classmethod
  def _free_signal(cls, signal:Any):
    """
    Free a signal.
    """
    raise NotImplementedError("_free_signal needs to be implemented")

  @classmethod
  def _wait_signal(cls, signal:Any, value:int = 0, timeout:int = 10000):
    """
    Wait for a signal to reach a specific value. Signals
    """
    raise NotImplementedError("_wait_signal needs to be implemented")

  def _gpu2cpu_time(self, gpu_time:int, is_copy:bool) -> float:
    """
    Convert GPU time to CPU time. `is_copy` flag indicating if this is a copy queue.
    """
    raise NotImplementedError("_gpu2cpu_time needs to be implemented")

  def _prof_setup(self):
    if not hasattr(self, 'profile_logger'): atexit.register(self._prof_finalize)
    self.profile_logger = ProfileLogger()

    def _sync_queue(q_t):
      q_t().timestamp(self.timeline_signal).signal(self.timeline_signal, self.timeline_value).submit(self)
      self.timeline_value += 1
      cpu_start_time = time.perf_counter_ns() / 1e3
      self._wait_signal(self.timeline_signal, self.timeline_value - 1)
      return cpu_start_time, self._read_timestamp(self.timeline_signal)
    self.cpu_start_time, self.gpu_start_time = _sync_queue(self.hw_compute_queue_t)
    self.copy_cpu_start_time, self.copy_gpu_start_time = _sync_queue(self.hw_copy_queue_t)

  def _prof_process_events(self):
    self.raw_prof_records += [(self._read_timestamp(st), self._read_timestamp(en), name, is_cp) for st, en, name, is_cp in self.sig_prof_records]
    for st, en, _, _ in self.sig_prof_records: map(self._alloc_signal, [st, en])
    self.sig_prof_records = []

  def _prof_finalize(self):
    for st, en, name, is_cp in self.raw_prof_records:
      self.profile_logger.events += [(name, self._gpu2cpu_time(st, is_cp), self._gpu2cpu_time(en, is_cp), self.dname, ["COMPUTE", "DMA"][is_cp])]
    del self.profile_logger

  def _wrap_timeline_signal(self):
    self.timeline_signal, self._shadow_timeline_signal, self.timeline_value = self._shadow_timeline_signal, self.timeline_signal, 1
    self._set_signal(self.timeline_signal, 0)
    cast(HCQCompatAllocator, self.allocator).b_timeline = [0] * len(cast(HCQCompatAllocator, self.allocator).b)

# Protocol for hcq compatible allocators for allocated buffers to contain VA address and it's size.
class HCQCompatAllocRes(Protocol): va_addr:int; size:int # noqa: E702

class HCQCompatAllocator(LRUAllocator): # pylint: disable=abstract-method
  """
  A base allocator class compatible with the HCQ (Hardware Command Queue) API.

  This class implements basic copy operations following the HCQ API, utilizing both `HWComputeQueue` and `HWCopyQueue`.
  """

  def __init__(self, device:HCQCompatCompiled, batch_size:int=(2 << 20), batch_cnt:int=32):
    self.device:Any = device
    self.b = [self._alloc(batch_size, BufferOptions(host=True)) for _ in range(batch_cnt)]
    self.b_timeline, self.b_next = [0] * len(self.b), 0
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions) -> HCQCompatAllocRes: raise NotImplementedError("need hcq compat alloc")

  def copyin(self, dest:HCQCompatAllocRes, src:memoryview):
    with hcq_profile(self.device, queue_type=self.device.hw_copy_queue_t, desc=f"CPU -> {self.device.dname}", enabled=PROFILE):
      for i in range(0, src.nbytes, self.b[0].size):
        self.b_next = (self.b_next + 1) % len(self.b)
        self.device._wait_signal(self.device.timeline_signal, self.b_timeline[self.b_next])
        ctypes.memmove(self.b[self.b_next].va_addr, from_mv(src[i:]), lsize:=min(self.b[self.b_next].size, src.nbytes-i))
        self.device.hw_copy_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                                     .copy(dest.va_addr+i, self.b[self.b_next].va_addr, lsize) \
                                     .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
        self.b_timeline[self.b_next] = self.device.timeline_value
        self.device.timeline_value += 1

  def copy_from_disk(self, dest:HCQCompatAllocRes, src, size):
    def _get_temp_buf():
      # Check if the next buffer is safe to be used (its signal has passed) and reserve it.
      if self.b_timeline[(self.b_next + 1) % len(self.b)] <= self.device._read_signal(self.device.timeline_signal):
        self.b_timeline[(self.b_next + 1) % len(self.b)], self.b_next = (1 << 64), (self.b_next + 1) % len(self.b)
        return (self.b[self.b_next].va_addr, self.b_next)
      return None

    with hcq_profile(self.device, queue_type=self.device.hw_copy_queue_t, desc=f"DISK -> {self.device.dname}", enabled=PROFILE):
      for (batch_info, dst_off, src_off, copy_size) in src.device.allocator._copyout_sharded(src, size, _get_temp_buf, seg_len=self.b[0].size):
        self.device.hw_copy_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                                     .copy(dest.va_addr + dst_off, batch_info[0] + src_off, copy_size) \
                                     .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
        self.b_timeline[batch_info[1]] = self.device.timeline_value
        self.device.timeline_value += 1

  def copyout(self, dest:memoryview, src:HCQCompatAllocRes):
    self.device.synchronize()

    with hcq_profile(self.device, queue_type=self.device.hw_copy_queue_t, desc=f"{self.device.dname} -> CPU", enabled=PROFILE):
      for i in range(0, dest.nbytes, self.b[0].size):
        self.device.hw_copy_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
                                     .copy(self.b[0].va_addr, src.va_addr+i, lsize:=min(self.b[0].size, dest.nbytes-i)) \
                                     .signal(self.device.timeline_signal, self.device.timeline_value).submit(self.device)
        self.device._wait_signal(self.device.timeline_signal, self.device.timeline_value)
        self.device.timeline_value += 1

        ctypes.memmove(from_mv(dest[i:]), self.b[0].va_addr, lsize)

  def transfer(self, dest:HCQCompatAllocRes, src:HCQCompatAllocRes, sz:int, src_dev, dest_dev):
    src_dev._gpu_map(dest)

    with hcq_profile(self.device, queue_type=self.device.hw_copy_queue_t, desc=f"{src_dev.dname} -> {dest_dev.dname}", enabled=PROFILE):
      src_dev.hw_copy_queue_t().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                               .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                               .copy(dest.va_addr, src.va_addr, sz) \
                               .signal(src_dev.timeline_signal, src_dev.timeline_value).submit(src_dev)
      src_dev.timeline_value += 1

    if src_dev != dest_dev:
      dest_dev.hw_compute_queue_t().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                                   .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                                   .signal(dest_dev.timeline_signal, dest_dev.timeline_value).submit(dest_dev)
      dest_dev.timeline_value += 1

  def offset(self, buf, size:int, offset:int) -> HCQCompatAllocRes:
    return type(buf)(va_addr=buf.va_addr + offset, size=size, **{k:v for k,v in buf.__dict__.items() if k not in ['va_addr', 'size']},
                     **{x[0]:getattr(buf, x[0]) for x in getattr(buf, '_fields_', []) if x[0] not in ['va_addr', 'size']}, _base=buf)
