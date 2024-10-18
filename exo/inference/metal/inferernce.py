from typing import Dict, List, Any , Tuple
import numpy as np
from tinygrad import dtypes
from .metal_model_shard import MetalModelShard, MetalKernelMetadata
from .utils import KernelOperation , Linearizer,Kernel
from typing import Optional, List, Dict
from exo.download.shard_download import ShardDownloader
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from metal_kernel_compiler import MetalKernelCompiler
from swift_code_generator import SwiftCodeGenerator
from .sharded_utils import load_shard, get_image_from_str

class MetalDynamicShardInferenceEngine:
    def __init__(self, shard_downloader):
        self.shard = None
        self.shard_downloader = shard_downloader
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.metal_compiler = MetalKernelCompiler()
        self.swift_generator = SwiftCodeGenerator()
        self.metal_engine = None
        self.tokenizer = None

    async def initialize_metal_engine(self, model_shard: MetalModelShard):
        """Initialize the Metal engine with compiled kernels"""
        # Compile all kernels in the model
        compiled_kernels = {}
        for kernel_name, kernel in model_shard.kernels.items():
            metal_code, metadata = self.metal_compiler.compile_kernel_to_metal(kernel)
            compiled_kernels[kernel_name] = (metal_code, metadata)
            
        # Generate Swift wrapper code
        swift_code = self.swift_generator.generate_swift_wrapper(
            {name: meta for name, (_, meta) in compiled_kernels.items()}
        )
        
        # Initialize Metal engine through Swift bridge
        self.metal_engine = await self._initialize_swift_metal_engine(swift_code)
