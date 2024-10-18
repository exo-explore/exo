from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from tinygrad import dtypes
from tinygrad.codegen.kernel import Kernel
from .metal_model_shard import MetalKernelMetadata

class SwiftCodeGenerator:
    def generate_swift_wrapper(self, kernel_metadata: Dict[str, MetalKernelMetadata]) -> str:
        """Generate Swift wrapper code for Metal kernels"""
        return f"""
import Metal
import MetalKit

enum MLXMetalError: Error {{
    case deviceNotFound
    case commandQueueCreationFailed
    case libraryCompilationFailed
    case kernelCreationFailed
    case bufferCreationFailed
    case encoderCreationFailed
    case invalidInputSize
}}

@available(iOS 11.0, macOS 10.13, *)
class MLXMetalEngine {{
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var kernels: [String: MTLComputePipelineState] = [:]
    private let kernelMetadata: [String: KernelMetadata]
    
    struct KernelMetadata {{
        let inputShapes: [[Int]]
        let outputShape: [Int]
        let workGroupSize: (Int, Int, Int)
        let globalSize: (Int, Int, Int)
        let bufferSizes: [Int]
    }}
    
    init() throws {{
        guard let device = MTLCreateSystemDefaultDevice() else {{
            throw MLXMetalError.deviceNotFound
        }}
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {{
            throw MLXMetalError.commandQueueCreationFailed
        }}
        self.commandQueue = commandQueue
        
        self.kernelMetadata = [
            {self._generate_kernel_metadata_initialization(kernel_metadata)}
        ]
        
        try compileKernels()
    }}
    
    private func compileKernels() throws {{
        let source = """
        {self._generate_metal_kernel_source(kernel_metadata)}
        """
        
        let library: MTLLibrary
        do {{
            library = try device.makeLibrary(source: source, options: nil)
        }} catch {{
            throw MLXMetalError.libraryCompilationFailed
        }}
        
        for kernelName in kernelMetadata.keys {{
            guard let function = library.makeFunction(name: kernelName) else {{
                throw MLXMetalError.kernelCreationFailed
            }}
            
            do {{
                kernels[kernelName] = try device.makeComputePipelineState(function: function)
            }} catch {{
                throw MLXMetalError.kernelCreationFailed
            }}
        }}
    }}
    
    func createBuffer<T>(_ data: [T]) throws -> MTLBuffer {{
        let size = MemoryLayout<T>.stride * data.count
        guard let buffer = device.makeBuffer(bytes: data, length: size, options: .storageModeShared) else {{
            throw MLXMetalError.bufferCreationFailed
        }}
        return buffer
    }}
    
    func execute(kernelName: String, inputs: [[Float]]) throws -> [Float] {{
        guard let kernel = kernels[kernelName],
              let metadata = kernelMetadata[kernelName] else {{
            throw MLXMetalError.kernelCreationFailed
        }}
        
        for (input, expectedShape) in zip(inputs, metadata.inputShapes) {{
            guard input.count == expectedShape.reduce(1, *) else {{
                throw MLXMetalError.invalidInputSize
            }}
        }}
        
        let inputBuffers = try inputs.map {{ try createBuffer($0) }}
        
        let outputSize = metadata.outputShape.reduce(1, *)
        let outputBuffer = device.makeBuffer(length: outputSize * MemoryLayout<Float>.size,
                                             options: .storageModeShared)!
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {{
            throw MLXMetalError.encoderCreationFailed
        }}
        
        encoder.setComputePipelineState(kernel)
        
        for (index, buffer) in inputBuffers.enumerated() {{
            encoder.setBuffer(buffer, offset: 0, index: index)
        }}
        
        encoder.setBuffer(outputBuffer, offset: 0, index: inputBuffers.count)
        
        let (w, h, d) = metadata.globalSize
        let threadsPerThreadgroup = MTLSize(width: 32, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (w + 31) / 32,
            height: h,
            depth: d
        )
        
        encoder.dispatchThreadgroups(threadgroupsPerGrid,
                                     threadsPerThreadgroup: threadsPerThreadgroup)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let outputPtr = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: outputSize
        )
        return Array(UnsafeBufferPointer(start: outputPtr, count: outputSize))
    }}
}}
"""
        def _generate_kernel_metadata_initialization(self, kernel_metadata: Dict[str, MetalKernelMetadata]) -> str:
        initializations = []
        for name, metadata in kernel_metadata.items():
            init = f'"{name}": KernelMetadata('
            init += f'inputShapes: {metadata.input_shapes}, '
            init += f'outputShape: {metadata.output_shape}, '
            init += f'workGroupSize: {metadata.work_group_size}, '
            init += f'globalSize: {metadata.global_size}, '
            init += f'bufferSizes: {metadata.buffer_sizes})'
            initializations.append(init)
        return ",\n            ".join(initializations)
