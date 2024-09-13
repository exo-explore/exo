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
    def __init__(self, shard: Shard, ):
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
                torch_dtype=self.torch_dtype,
                device_map="auto"
            )

            # build layers from shard
            layers = self.base_model.model.layers
            copy_layers = nn.ModuleList(
                [layers[i] for i in range(self.shard.start_layer, self.shard.end_layer + 1)]
            )

            # apply layers back to model
            self.base_model.model.layers.load_state_dict(
                copy_layers.state_dict(),
                strict=False
            )
        except Exception as err:
            print(f"error loading and splitting model: {err}")
            raise

    def run(
        self,
        input_ids: torch.tensor
    ) -> Tuple[np.ndarray, any]:
        """
        Run through a set of model layers

        Args:
            input_ids: tensor input
                this could be tokens or hidden states from other layers

        Returns:
            layer_outputs: dict
                layer output including hidden states, key values or logits
        """
