from __future__ import annotations
import os, subprocess, pathlib, ctypes, tempfile, functools
import Metal, libdispatch
from typing import List, Set, Any, Tuple, Optional
from tinygrad.helpers import prod, getenv, DEBUG, unwrap2
from tinygrad.device import Compiled, Compiler, CompileError, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer

def wait_check(cbuf: Any):
  cbuf.waitUntilCompleted()
  if (error := cbuf.error()) is not None:
    raise RuntimeError(error)

class MetalCompiler(Compiler):
  def __init__(self, device:Optional[MetalDevice]):
    self.device = device
    super().__init__("compile_metal")
  def compile(self, src:str) -> bytes:
    if self.device is None:
      # NOTE: if you run llvm-dis on "air" you can see the llvm bytecode
      air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=src.encode('utf-8'))
      return subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
    options = Metal.MTLCompileOptions.new()
    options.setFastMathEnabled_(getenv("METAL_FAST_MATH"))
    try: library = unwrap2(self.device.device.newLibraryWithSource_options_error_(src, options, None))
    except AssertionError as e: raise CompileError(e) from e
    return library.libraryDataContents().bytes().tobytes()

class MetalProgram:
  def __init__(self, device:MetalDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    if DEBUG >= 6:
      with tempfile.NamedTemporaryFile(delete=True) as shader:
        shader.write(lib)
        shader.flush()
        ret = os.system(f"cd {pathlib.Path(__file__).parents[2]}/extra/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}")
        if ret:
          print("Error running disassembler: Make sure you have https://github.com/dougallj/applegpu cloned to tinygrad/extra/disassemblers/applegpu")
    assert lib[:4] == b"MTLB", "Invalid Metal library. Could be due to using conda. Try system python or METAL_XCODE=1 DISABLE_COMPILER_CACHE=1."
    data = libdispatch.dispatch_data_create(lib, len(lib), None, None)
    self.library = unwrap2(self.device.device.newLibraryWithData_error_(data, None))
    self.fxn = self.library.newFunctionWithName_(name)
    self.pipeline_state = unwrap2(self.device.device.newComputePipelineStateWithFunction_error_(self.fxn, None))

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if prod(local_size) > self.pipeline_state.maxTotalThreadsPerThreadgroup(): raise RuntimeError(f"local size {local_size} bigger than {self.pipeline_state.maxTotalThreadsPerThreadgroup()} with exec width {self.pipeline_state.threadExecutionWidth()} memory length {self.pipeline_state.staticThreadgroupMemoryLength()}")  # noqa: E501
    command_buffer = self.device.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(self.pipeline_state)
    for i,a in enumerate(bufs): encoder.setBuffer_offset_atIndex_(a.buf, a.offset, i)
    for i,a in enumerate(vals,start=len(bufs)): encoder.setBytes_length_atIndex_(ctypes.c_int32(a), 4, i)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    if wait:
      wait_check(command_buffer)
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    self.device.mtl_buffers_in_flight.append(command_buffer)

class MetalBuffer:
  def __init__(self, buf:Any, size:int, offset=0): self.buf, self.size, self.offset = buf, size, offset

class MetalAllocator(LRUAllocator):
  def __init__(self, device:MetalDevice):
    self.device:MetalDevice = device
    self.track_cross_device: Set[MetalDevice] = set()
    super().__init__()
  def free_cache(self):
    self.device.synchronize()
    for x in self.track_cross_device: x.synchronize()
    self.track_cross_device.clear()
    return super().free_cache()
  def _alloc(self, size:int, options) -> MetalBuffer:
    ret = self.device.device.newBufferWithLength_options_(size, Metal.MTLResourceStorageModeShared)
    if ret is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    return MetalBuffer(ret, size)
  def _free(self, opaque:MetalBuffer, options): opaque.buf.release()
  def transfer(self, dest:MetalBuffer, src:MetalBuffer, sz:int, src_dev: MetalDevice, **kwargs):
    src_dev.synchronize()
    command_buffer = self.device.mtl_queue.commandBuffer()
    encoder = command_buffer.blitCommandEncoder()
    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(src.buf, src.offset, dest.buf, dest.offset, sz)
    encoder.endEncoding()
    command_buffer.commit()
    self.device.mtl_buffers_in_flight.append(command_buffer)
  def from_buffer(self, src:memoryview) -> Optional[Any]:
    ret = self.device.device.newBufferWithBytesNoCopy_length_options_deallocator_(src, src.nbytes, Metal.MTLResourceStorageModeShared, None)
    if ret: self.device.mv_in_metal.append(src)
    return MetalBuffer(ret, src.nbytes)
  def as_buffer(self, src:MetalBuffer) -> memoryview:
    self.device.synchronize()
    return src.buf.contents().as_buffer(src.offset+src.size)[src.offset:]
  def copyin(self, dest:MetalBuffer, src:memoryview): self.as_buffer(dest)[:] = src
  def copyout(self, dest:memoryview, src:MetalBuffer): dest[:] = self.as_buffer(src)
  def offset(self, buf:MetalBuffer, size:int, offset:int): return MetalBuffer(buf.buf, size, offset)

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.device = Metal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = self.device.newCommandQueueWithMaxCommandBufferCount_(1024)
    self.mtl_buffers_in_flight: List[Any] = []
    self.mv_in_metal: List[memoryview] = []
    self.track_cross_buffer: List[Any] = []
    from tinygrad.runtime.graph.metal import MetalGraph
    super().__init__(device, MetalAllocator(self), MetalRenderer(), MetalCompiler(None if getenv("METAL_XCODE") else self),
                     functools.partial(MetalProgram, self), MetalGraph)
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: wait_check(cbuf)
    self.mv_in_metal.clear()
    self.mtl_buffers_in_flight.clear()
    self.track_cross_buffer.clear()
