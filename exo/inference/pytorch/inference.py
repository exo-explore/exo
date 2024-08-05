# experimental, based off of tinygrad/inference.py
import os
import numpy as np
import asyncio
import json
import torch
from functools import partial
from pathlib import Path
from typing import List, Optional, Union, Callable, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
# from exo.inference.pytorch.helpers import 

# default settings
TEMPERATURE = 0  # 0.85
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0


# don't think prefill is needed
# think that is used for stats but will look into

class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self):
        self.shard = None

    # async def infer_prompt

    # async def infer_tensor

    # async def ensure_shard

    # def set_on_download_progess [is this needed?]
