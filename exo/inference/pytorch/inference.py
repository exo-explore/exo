# experimental, based off of tinygrad/inference.py
import os
import numpy as np
import torch
import numpy as np
import json
from typing import Optional, Callable, Tuple
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.inference.pytorch.model.hf import ShardedHuggingFaceModel
from exo.api.chatgpt_api import resolve_tokenizer
from exo.helpers import DEBUG
from transformers import DynamicCache

from exo.inference.pytorch.model.utils import sample_logits

class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    """
    PyTorch Dynamic Shard Inference Engine for performing model inference with sharded models.
    """

    def __init__(self, shard):
        """
        Initialize the inference engine.

        Args:
            debug (bool): If True, enables debug logging. Defaults to False.
        """
        self.shard = shard
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.getenv("TORCH_CACHED") == "True":
            self.use_cache = True
        else:
            self.use_cache = False

    async def infer_prompt(
        self, 
        request_id: str, 
        shard: Optional[Shard] = None, 
        prompt: str = "", 
        image_str: Optional[str] = None, 
        inference_state: Optional[str] = None
    ) -> Tuple[np.ndarray, str, bool]:
        
        await self.ensure_shard(shard)

        # need to make this so inference_state is not a string
        # cant use it with dynamic cache
           
        tokens = self.tokenizer.encode(prompt, return_tensors="pt")

        if self.use_cache:
            # convert inference_state or cache from json to DynamicCache
            past_kv = DynamicCache()
            if inference_state != None:
                cache_dict = json.loads(inference_state)
                past_kv.key_cache = [torch.tensor(data) for data in cache_dict['key_cache']]
                past_kv.value_cache = [torch.tensor(data) for data in cache_dict['value_cache']]
                
            output_data, current_kvs = self.model.forward(
                tokens,
                past_kv,
                use_cache=True
            )
        else:
            output_data = self.model.forward(
                tokens,
                use_cache=False
            )

        is_finished = output_data.size == 1 and output_data.item() in [self.tokenizer.eos_token_id]

        if DEBUG >= 4:
            print("infer_prompt called")
            print(f"tokens: {tokens}\n")
            print(f"output_data: {output_data}\n")
            print(f"output_data.size {output_data.size}\n")
            
            print(f"finished: {is_finished}")
            print(f"self.tokenizer.eos_token_id {self.tokenizer.eos_token_id}")
            print(f"output_data[-1] {output_data[-1]}")

            if output_data.size == 1:
                print(f"size 1 output_data.item() {output_data.item()}")
                print(f"output_data.item() in [self.tokenizer.eos_token_id]: {output_data.item() in [self.tokenizer.eos_token_id]}")

        if self.use_cache:
            # legacy_cache = current_kvs.to_legacy_cache()
            print(current_kvs.key_cache)
            cache_dict = {
                'key_cache': [tensor.tolist() for tensor in current_kvs.key_cache],
                'value_cache': [tensor.tolist() for tensor in current_kvs.value_cache]
            }

        return (
            output_data,
            json.dumps(cache_dict) if self.use_cache else "",
            is_finished
        )

    async def infer_tensor(
        self, 
        request_id: str, 
        shard: Shard, 
        input_data: np.ndarray, 
        inference_state: Optional[str] = None
    ) -> Tuple[np.ndarray, str, bool]:

        await self.ensure_shard(shard)

        in_tensor = torch.tensor(input_data)

        if self.use_cache:
            # convert inference_state or cache from json to DynamicCache
            past_kv = DynamicCache()
            if inference_state != None:
                cache_dict = json.loads(inference_state)
                past_kv.key_cache = [torch.tensor(data) for data in cache_dict['key_cache']]
                past_kv.value_cache = [torch.tensor(data) for data in cache_dict['value_cache']]

            output_data, current_kvs = self.model.forward(
                in_tensor,
                past_kv,
                use_cache=True
            )
        else:
            output_data = self.model.forward(
                in_tensor,
                use_cache=False
            )

        is_finished = output_data.size == 1 and output_data.item() in [self.tokenizer.eos_token_id]

        if DEBUG >= 4:
            print("infer_tensor called")
            print(f"input_data: {input_data}\n")
            print(f"in_tensor: {in_tensor}\n")
            print(f"output_data: {output_data}\n")
            print(f"output_data.size {output_data.size}\n")
            print(f"finished: {is_finished}")
            print(f"self.tokenizer.eos_token_id {self.tokenizer.eos_token_id}")
            print(f"output_data[-1] {output_data[-1]}")

            if output_data.size == 1:
                print(f"size 1 output_data.item() {output_data.item()}")
                print(f"output_data.item() in [self.tokenizer.eos_token_id]: {output_data.item() in [self.tokenizer.eos_token_id]}")

        if self.use_cache:
            legacy_cache = current_kvs.to_legacy_cache()
            cache_dict = {
                'key_cache': [tensor.tolist() for tensor in legacy_cache.key_cache],
                'value_cache': [tensor.tolist() for tensor in legacy_cache.value_cache]
            }

        return (
            output_data,
            json.dumps(cache_dict) if self.use_cache else "",
            is_finished
        )

    async def ensure_shard(self, shard: Optional[Shard]):
        """
        Ensure the model shard is loaded and ready for inference.

        Args:
            shard (Optional[Shard]): Shard information for the model.
        """
        if self.shard == shard:
            return

        if DEBUG >= 4:
            print(f"Loading new shard: {shard}")

        # if self.model:
        #     if DEBUG >= 2:
        #         print(f"\nCLEARING MODEL {self.shard.model_id}\n")
                
        #     # delete model and free up memory to reload
        #     self.model.cpu()
        #     del self.model
        #     torch.cuda.empty_cache()

        self.shard = shard
        self.tokenizer = await resolve_tokenizer(shard.model_id)
        self.model = ShardedHuggingFaceModel(shard, self.tokenizer)

        if DEBUG >= 4:
            print(f"Shard loaded successfully: {shard}")