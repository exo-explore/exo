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
