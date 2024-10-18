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

    async def infer_prompt(self, request_id: str, shard: MetalModelShard, 
                          prompt: str, image_str: Optional[str] = None,
                          inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        """Run inference on text or image-text input"""
        await self.ensure_shard(shard)
        loop = asyncio.get_running_loop()
        
        if image_str:
            # Handle image-text input
            image = await get_image_from_str(image_str)
            inputs = await loop.run_in_executor(
                self.executor,
                self._prepare_image_text_input,
                prompt,
                image
            )
            
            # Run through vision encoder
            vision_features = await self._run_vision_encoder(inputs["pixel_values"])
            
            # Run through text encoder
            output_data = await self._run_text_decoder(
                request_id,
                inputs["input_ids"],
                vision_features
            )
        else:
            # Handle text-only input
            input_ids = await loop.run_in_executor(
                self.executor,
                self._tokenize_text,
                prompt
            )
            output_data = await self._run_text_decoder(request_id, input_ids)
            
        is_finished = (output_data.size == 1 and 
                      output_data.item() == self.tokenizer.eos_token_id)
        return output_data, "", is_finished

    async def infer_tensor(self, request_id: str, shard: MetalModelShard,
                          input_data: np.ndarray,
                          inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        """Run inference on raw tensor input"""
        await self.ensure_shard(shard)
        
        # Convert numpy array to Metal buffer
        input_buffer = await self._numpy_to_metal_buffer(input_data)
        
        # Run inference kernels
        output_data = await self._run_inference_kernels(request_id, input_buffer)
        
        is_finished = (output_data.size == 1 and 
                      output_data.item() == self.tokenizer.eos_token_id)
        return output_data, "", is_finished

    async def ensure_shard(self, shard: MetalModelShard):
        """Load model shard if needed"""
        if self.shard == shard:
            return

        model_path = await self.shard_downloader.ensure_shard(shard)

        if self.shard != shard:
            loop = asyncio.get_running_loop()
            
            # Load shard weights and config
            def load_shard_wrapper():
                return asyncio.run(self._load_metal_shard(model_path, shard))
            
            model_shard, self.tokenizer = await loop.run_in_executor(
                self.executor,
                load_shard_wrapper
            )
            
            # Initialize Metal engine with new shard
            await self.initialize_metal_engine(model_shard)
            self.shard = shard
