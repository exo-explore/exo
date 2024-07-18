from typing import List, Any, Dict, cast, Optional
import Metal
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, unwrap2, GraphException
from tinygrad.device import Buffer
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.engine.jit import GraphRunner
from tinygrad.shape.symbolic import Variable
from tinygrad.runtime.ops_metal import wait_check

class MetalGraph(GraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    # create metal batch exec
    icb_descriptor = Metal.MTLIndirectCommandBufferDescriptor.new()
    icb_descriptor.setCommandTypes_(Metal.MTLIndirectCommandType(Metal.MTLIndirectCommandTypeConcurrentDispatch))
    icb_descriptor.setInheritBuffers_(False)
    icb_descriptor.setInheritPipelineState_(False)
    icb_descriptor.setMaxKernelBufferBindCount_(31)
    self.icb = self.device.device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(icb_descriptor, len(self.jit_cache),
                                                                                                  Metal.MTLResourceOptions(0))
    if self.icb is None: raise GraphException("create indirect command buffer failed, does your system support this?")

    if len(self.vars): self.int_buf = self.device.allocator.alloc(len(self.vars)*dtypes.int32.itemsize)
    all_resources = [self.int_buf.buf] if len(self.vars) else []

    for j,ji in enumerate(self.jit_cache):
      prg: CompiledRunner = cast(CompiledRunner, ji.prg)
      descriptor = Metal.MTLComputePipelineDescriptor.new()
      descriptor.setComputeFunction_(prg.clprg.fxn)
      descriptor.setSupportIndirectCommandBuffers_(True)
      icb_command = self.icb.indirectComputeCommandAtIndex_(j)
      icb_command.setComputePipelineState_(unwrap2(
        self.device.device.newComputePipelineStateWithDescriptor_options_reflection_error_(descriptor, Metal.MTLPipelineOption(0), None, None)))
      for i,b in enumerate(ji.bufs):
        if b is not None:
          icb_command.setKernelBuffer_offset_atIndex_(b._buf.buf, b._buf.offset, i)
          all_resources.append(b._buf.buf)
      for i,v in enumerate(prg.p.vars): icb_command.setKernelBuffer_offset_atIndex_(self.int_buf.buf, self.vars.index(v)*4, len(ji.bufs)+i)
      if j not in self.jc_idx_with_updatable_launch_dims:
        global_size, local_size = prg.p.launch_dims(var_vals)
        icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
      icb_command.setBarrier()

    self.all_resources = dedup(all_resources)
    self.command_buffer: Any = None
    if len(self.vars): self.int_buf_view = self.int_buf.buf.contents().as_buffer(self.int_buf.buf.length()).cast('i')

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    if self.command_buffer is not None and self.command_buffer in self.device.mtl_buffers_in_flight: wait_check(self.command_buffer)
    all_resources = dedup(self.all_resources + [x._buf.buf for x in input_rawbuffers])

    for (j,i),input_idx in self.input_replace.items():
      self.icb.indirectComputeCommandAtIndex_(j).setKernelBuffer_offset_atIndex_(input_rawbuffers[input_idx]._buf.buf,
                                                                                 input_rawbuffers[input_idx]._buf.offset, i)
    for j in self.jc_idx_with_updatable_launch_dims:
      global_size, local_size = cast(CompiledRunner, self.jit_cache[j].prg).p.launch_dims(var_vals)
      self.icb.indirectComputeCommandAtIndex_(j).concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size),
                                                                                                       Metal.MTLSize(*local_size))
    for j, var in enumerate(self.vars): self.int_buf_view[j] = var_vals[var]

    command_buffer = self.device.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.useResources_count_usage_(all_resources, len(all_resources), Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite)
    encoder.executeCommandsInBuffer_withRange_(self.icb, Metal.MTLIndirectCommandBufferExecutionRangeMake(0, len(self.jit_cache)))
    encoder.endEncoding()
    command_buffer.commit()
    self.command_buffer = command_buffer

    if wait:
      wait_check(command_buffer)
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    self.device.mtl_buffers_in_flight.append(command_buffer)
    return None
