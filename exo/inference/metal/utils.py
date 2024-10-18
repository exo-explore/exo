from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from tinygrad import dtypes
from tinygrad.codegen.kernel import Kernel
# from tinygrad import Linearizer

@dataclass
class UOp:
    op: str
    dtype: Any
    args: List[Any]

@dataclass
class Kernel:
    name: str
    uops: List[UOp]
