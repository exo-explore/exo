from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from tinygrad import dtypes
from tinygrad.codegen.kernel import Kernel
from .utils import KernelOperation , Linearizer,Kernel

@dataclass
class MetalKernelMetadata:
    name: str
    input_shapes: List[List[int]]
    output_shape: List[int]
    work_group_size: Tuple[int, int, int]
    global_size: Tuple[int, int, int]
    buffer_sizes: List[int]
    op_sequence: List[KernelOperation]
