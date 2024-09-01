import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader
from typing import Tuple, Optional, Union, List

class ShardedHuggingFaceModel(InferenceEngine):
    def __init__(self, shard: Shard):
        self.shard = shard

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.torch_dtype = torch.float32
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.torch_dtype = torch.float32
        else:
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float16

        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                shard.model_id,
                torch_dtype=torch.float32,
                device_map="auto"
            )
        except Exception as err:
            print(f"error loading model: {err}")
            raise

    def forward(
        self,
        input_ids: torch.tensor
    ) -> Tuple[np.ndarray, any]:
        """
        Forward through layers using the base model

        Args:
            input_ids: tensor input

        Returns:
            generator_ids: token ids from generation
        """

        torch_dtype = 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.shard.model_id,
            torch_dtype=torch.float32,
            device_map="auto",
        )


