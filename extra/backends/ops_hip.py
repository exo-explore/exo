from __future__ import annotations
import ctypes, functools, subprocess, io
from typing import Tuple, TypeVar, List, Any, cast, Set
import tinygrad.runtime.autogen.hip as hip
from tinygrad.helpers import DEBUG, getenv, init_c_var
from tinygrad.helpers import from_mv, round_up, to_mv, colored, init_c_struct_t
from tinygrad.device import Compiled, LRUAllocator, BufferOptions, Runner, Device, Buffer, MallocAllocator, update_stats, Compiler, CompilerOptions
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.support.hip_comgr import compile_hip
from tinygrad.renderer.rdna import uops_to_rdna

class RDNACompiler(Compiler):
  linearizer_opts = LinearizerOptions("HIP", has_tensor_cores=True)
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_rdna_{self.arch}")
  def render(self, name:str, uops) -> str: return uops_to_rdna(name, uops)
  def compile(self, src:str) -> bytes:
    ret = compile_hip(src, self.arch, True)
    #with open("/tmp/out.so", "wb") as f: f.write(ret)
    return ret

class HIPCompiler(Compiler):
  compiler_opts = CompilerOptions("HIP", has_tensor_cores=True, shared_max=65536)
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def render(self, name:str, uops) -> str: return HIPRenderer(name, uops)
  def compile(self, src:str) -> bytes: return compile_hip(src, self.arch)

hip_current_device = None
def hip_set_device(d:int):
  global hip_current_device
  if d == hip_current_device: return
  check(hip.hipSetDevice(d))
  hip_current_device = d

def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")

class HIPProgram:
  def __init__(self, device:int, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib

    if DEBUG >= 6:
      asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    hip_set_device(self.device)
    self.module = init_c_var(hip.hipModule_t(), lambda x: check(hip.hipModuleLoadData(ctypes.byref(x), lib)))
    self.prg = init_c_var(hip.hipFunction_t(), lambda x: check(hip.hipModuleGetFunction(ctypes.byref(x), self.module, name.encode("utf-8"))))

  def __del__(self):
    if hasattr(self, 'module'): check(hip.hipModuleUnload(self.module))

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    hip_set_device(self.device)
    if not hasattr(self, "vargs"):
      self.c_args = init_c_struct_t(tuple([(f'f{i}', hip.hipDeviceptr_t) for i in range(len(args))] +
                                          [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))(*args, *vals)
      self.vargs = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(ctypes.byref(self.c_args), ctypes.c_void_p),
                                         ctypes.c_void_p(2), ctypes.cast(ctypes.byref(ctypes.c_size_t(ctypes.sizeof(self.c_args))), ctypes.c_void_p),
                                         ctypes.c_void_p(3))
    else:
      for i in range(len(args)): self.c_args.__setattr__(f'f{i}', args[i])
      for i in range(len(vals)): self.c_args.__setattr__(f'v{i}', vals[i])
    if wait:
      evs = [init_c_var(hip.hipEvent_t(), lambda x: hip.hipEventCreate(ctypes.byref(x), 0)) for _ in range(2)]
      check(hip.hipEventRecord(evs[0], None))
    check(hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, self.vargs))
    if wait:
      check(hip.hipEventRecord(evs[1], None))
      check(hip.hipEventSynchronize(evs[1]))
      check(hip.hipEventElapsedTime(ctypes.byref(ret := ctypes.c_float()), evs[0], evs[1]))
      for ev in evs: check(hip.hipEventDestroy(ev))
      return ret.value * 1e-3
    return None

T = TypeVar("T")
CHUNK_SIZE, PAGE_SIZE = 256*1024*1024, 0x1000
class HIPAllocator(LRUAllocator):
  def __init__(self, device:HIPDevice):
    self.device = device
    self.track_cross_device: Set[HIPDevice] = set()
    super().__init__()
  def full_synchronize(self):
    self.device.synchronize()
    for x in self.track_cross_device: x.synchronize()
    self.track_cross_device.clear()
  def free_cache(self):
    self.full_synchronize()
    return super().free_cache()
  def _alloc(self, size:int):
    hip_set_device(self.device.device)
    return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipMalloc(ctypes.byref(x), size)))
  def _alloc_with_options(self, size:int, options:BufferOptions):
    hip_set_device(self.device.device)
    if options.uncached:
      return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipExtMallocWithFlags(ctypes.byref(x), size, 3)))  # hipDeviceMallocUncached = 3
    elif options.host:
      return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipHostMalloc(ctypes.byref(x), size, 2 if options.signal else 0)))
    else:
      raise Exception("no options")
  def _free(self, opaque:T): check(hip.hipFree(opaque))
  def copy_from_fd(self, dest, fd, offset, size):
    hip_set_device(self.device.device)
    if not hasattr(self, 'hb'):
      self.hb = [self._alloc_with_options(CHUNK_SIZE, BufferOptions(host=True)) for _ in range(2)]
      self.hb_events = [None, None]
      self.hb_polarity = 0
    fo = io.FileIO(fd, "a+b", closefd=False)
    fo.seek(offset - (minor_offset:=offset % PAGE_SIZE))
    copied_in = 0
    for local_offset in range(0, size+minor_offset, CHUNK_SIZE):
      local_size = min(round_up(size+minor_offset, PAGE_SIZE)-local_offset, CHUNK_SIZE)
      if self.hb_events[self.hb_polarity] is not None:
        # NOTE: block doesn't work here because we modify the CPU memory
        check(hip.hipEventSynchronize(self.hb_events[self.hb_polarity]))
        check(hip.hipEventDestroy(self.hb_events[self.hb_polarity]))
        self.hb_events[self.hb_polarity] = None
      fo.readinto(to_mv(self.hb[self.hb_polarity], local_size))
      check(hip.hipMemcpyAsync(ctypes.c_void_p(dest.value + copied_in), ctypes.c_void_p(self.hb[self.hb_polarity].value + minor_offset),
                               copy_size:=min(local_size-minor_offset, size-copied_in), hip.hipMemcpyHostToDevice, None))
      self.hb_events[self.hb_polarity] = init_c_var(hip.hipEvent_t(), lambda x: check(hip.hipEventCreate(ctypes.byref(x))))
      check(hip.hipEventRecord(self.hb_events[self.hb_polarity], None))
      copied_in += copy_size
      self.hb_polarity = (self.hb_polarity+1) % len(self.hb)
      minor_offset = 0 # only on the first
  def copyin(self, dest:T, src: memoryview):
    hip_set_device(self.device.device)
    host_mem = self._alloc_with_options(len(src), BufferOptions(host=True))
    self.device.pending_copyin.append(host_mem)
    ctypes.memmove(host_mem, from_mv(src), len(src))
    check(hip.hipMemcpyAsync(dest, host_mem, len(src), hip.hipMemcpyHostToDevice, None))
  def copyout(self, dest:memoryview, src:T):
    self.full_synchronize()
    hip_set_device(self.device.device)
    check(hip.hipMemcpy(from_mv(dest), src, len(dest), hip.hipMemcpyDeviceToHost))
  def transfer(self, dest:T, src:T, sz:int, **kwargs):
    hip_set_device(self.device.device)
    check(hip.hipMemcpyAsync(dest, src, sz, hip.hipMemcpyDeviceToDevice, None))

class HIPSyncEvent(Runner):
  def __init__(self, lb):
    self.lb, self.device, self.dname = lb, cast(HIPDevice, Device[lb.device]), lb.device
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals, wait=False, jit=False):
    to_mv(rawbufs[0]._buf, 4).cast("I")[0] = 0
    hip_set_device(self.device.device)
    check(hip.hipStreamWriteValue32(None, rawbufs[0]._buf, 1, 0))
    update_stats(colored("sync", "red"), 0, 0, {}, None, 1, jit, device=self.dname)

class HIPWaitEvent(Runner):
  def __init__(self, device):
    self.device, self.dname = cast(HIPDevice, Device[device]), device
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals, wait=False, jit=False):
    hip_set_device(self.device.device)
    check(hip.hipStreamWaitValue32(None, rawbufs[0]._buf, 1, 1, 0xFFFFFFFF))
    update_stats(colored("wait", "RED"), 0, 0, {}, None, 1, jit, device=self.dname)

if getenv("HIPCPU"):
  rhip = ctypes.CDLL("/usr/local/lib/libremu.so")
  class RHIPProgram:
    def __init__(self, name:str, lib:bytes):
      self.name, self.lib = name, lib
    def __call__(self, *args, global_size, local_size, vals=(), wait=False):
      args = (*args, *vals)
      rhip.hipModuleLaunchKernel(self.lib, len(self.lib), *global_size, *local_size, 0, None, None,
                                len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]))

class HIPDevice(Compiled):
  def __init__(self, device:str=""):
    self.device = int(device.split(":")[1]) if ":" in device else 0
    self.pending_copyin: List[ctypes.c_void_p] = []
    self.track_cross_buffer: List[Any] = []
    self.peers: Set[int] = set()

    if getenv("HIPCPU"):
      super().__init__(device, MallocAllocator, HIPCompiler("gfx1100"), RHIPProgram)
    else:
      self.arch = init_c_var(hip.hipDeviceProp_t(), lambda x: check(hip.hipGetDeviceProperties(x, self.device))).gcnArchName.decode()
      from tinygrad.runtime.graph.hip import HIPGraph
      super().__init__(device, HIPAllocator(self), RDNACompiler(self.arch) if getenv("RDNA") else HIPCompiler(self.arch),
                       functools.partial(HIPProgram, self.device), HIPGraph)
  def synchronize(self):
    if getenv("HIPCPU"): return
    hip_set_device(self.device)
    check(hip.hipDeviceSynchronize())
    for opaque in self.pending_copyin: check(hip.hipFree(opaque))
    self.track_cross_buffer.clear()
    self.pending_copyin.clear()
  def enable_peer(self, dnum):
    if self.device == dnum or dnum in self.peers: return
    hip_set_device(self.device)
    check(hip.hipDeviceEnablePeerAccess(dnum, 0))
    self.peers.add(dnum)
