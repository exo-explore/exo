from __future__ import annotations
import ctypes, functools
from typing import Tuple
import tinygrad.runtime.autogen.hip as hip
from tinygrad.helpers import DEBUG, init_c_var, from_mv, init_c_struct_t
from tinygrad.device import Compiled, LRUAllocator, BufferOptions
from tinygrad.runtime.ops_amd import AMDCompiler, disasm
from tinygrad.renderer.cstyle import HIPRenderer

def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")

class HIPProgram:
  def __init__(self, device:HIPDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib

    if DEBUG >= 6: print(disasm(lib))

    check(hip.hipSetDevice(self.device.device_id))
    self.module = init_c_var(hip.hipModule_t(), lambda x: check(hip.hipModuleLoadData(ctypes.byref(x), lib)))
    self.prg = init_c_var(hip.hipFunction_t(), lambda x: check(hip.hipModuleGetFunction(ctypes.byref(x), self.module, name.encode("utf-8"))))

  def __del__(self):
    if hasattr(self, 'module'): check(hip.hipModuleUnload(self.module))

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    check(hip.hipSetDevice(self.device.device_id))
    if not hasattr(self, "vargs"):
      self.c_args = init_c_struct_t(tuple([(f'f{i}', hip.hipDeviceptr_t) for i in range(len(args))] +
                                          [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))(*args, *vals)
      self.vargs = (ctypes.c_void_p * 5)(1, ctypes.cast(ctypes.byref(self.c_args), ctypes.c_void_p), 2,
                                         ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(self.c_args))), ctypes.c_void_p), 3)

    for i in range(len(args)): self.c_args.__setattr__(f'f{i}', args[i])
    for i in range(len(vals)): self.c_args.__setattr__(f'v{i}', vals[i])

    if wait: check(hip.hipEventRecord(self.device.time_event_st, None))

    check(hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, self.vargs))

    if wait:
      check(hip.hipEventRecord(self.device.time_event_en, None))
      check(hip.hipEventSynchronize(self.device.time_event_en))
      check(hip.hipEventElapsedTime(ctypes.byref(ret := ctypes.c_float()), self.device.time_event_st, self.device.time_event_en))
      return ret.value * 1e-3

class HIPAllocator(LRUAllocator):
  def __init__(self, device:HIPDevice):
    self.device = device
    super().__init__()
  def _alloc(self, size:int, options:BufferOptions):
    check(hip.hipSetDevice(self.device.device_id))
    return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipMalloc(ctypes.byref(x), size)))
  def _free(self, opaque): check(hip.hipFree(opaque))
  def copyin(self, dest, src: memoryview):
    check(hip.hipSetDevice(self.device.device_id))
    check(hip.hipMemcpy(dest, from_mv(src), len(src), hip.hipMemcpyHostToDevice))
  def copyout(self, dest:memoryview, src):
    self.device.synchronize()
    check(hip.hipMemcpy(from_mv(dest), src, len(dest), hip.hipMemcpyDeviceToHost))

class HIPDevice(Compiled):
  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.arch = init_c_var(hip.hipDeviceProp_t(), lambda x: check(hip.hipGetDeviceProperties(x, self.device_id))).gcnArchName.decode()
    self.time_event_st, self.time_event_en = [init_c_var(hip.hipEvent_t(), lambda x: hip.hipEventCreate(ctypes.byref(x), 0)) for _ in range(2)]
    super().__init__(device, HIPAllocator(self), HIPRenderer(), AMDCompiler(self.arch), functools.partial(HIPProgram, self))
  def synchronize(self):
    check(hip.hipSetDevice(self.device_id))
    check(hip.hipDeviceSynchronize())
