from typing import Dict, List, Any , Tuple
import numpy as np
from tinygrad import dtypes
from .metal_model_shard import MetalModelShard, MetalKernelMetadata
from .utils import KernelOperation , Linearizer,Kernel

class MetalKernelCompiler:
    def __init__(self):
        self.kernels: Dict[str, str] = {}
        self.kernel_order: List[str] = []
        self.kernel_metadata: Dict[str, MetalKernelMetadata] = {}
