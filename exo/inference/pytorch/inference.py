# experimental, based off of tinygrad/inference.py
import numpy as np
import torch
import json
import gc
from typing import Optional, Tuple
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.inference.pytorch.model.hf import ShardedHuggingFaceModel
from exo.api.chatgpt_api import resolve_tokenizer
from exo.helpers import DEBUG
from transformers import DynamicCache
from accelerate import disk_offload
from exo.download.shard_download import ShardDownloader

# model value options 
TOP_K = 20
TEMP = 0.6
TOP_P = 0.9
MAX_LENGTH = 125
MAX_TIME = 60.0

class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    """
    PyTorch Dynamic Shard Inference Engine for performing model inference with sharded Pytorch/HF based models.
    """

    def __init__(self, shard_downloader: ShardDownloader):
        """
        Initialize the inference engine.

        Args:
            debug (bool): If True, enables debug logging. Defaults to False.
        """
        self.shard = None
        self.shard_downloader = shard_downloader
        self.stateful_sharded_model = None
        self.tokenizer = None

        # the whole history with new logits need to 
        # be passed to the model to reach the end token 
        # even with caching
        self.past_input_ids = None 

        # setup cuda device 
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.torch_dtype = torch.float32
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.torch_dtype = torch.float32
        else:
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float16

        # setup unfinished sequence
        self.unfinished_sequences = torch.ones(1, dtype=torch.long, device=self.device) 

    async def infer_prompt(
        self, 
        request_id: str, 
        shard: Optional[Shard] = None, 
        prompt: str = "", 
        image_str: Optional[str] = None, 
        inference_state: Optional[str] = None
    ) -> Tuple[np.ndarray, str, bool]:
        if DEBUG >= 4:
            print("infer_prompt called")
        
        await self.ensure_shard(shard)

        # setup prompt input 
        messages = [{"role": "user", "content": prompt}]
        txt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([txt], return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        input_attention_mask = inputs.attention_mask.to(self.device) 
        batch_size, seq_length = input_ids.shape[:2]


        if inference_state is not None:
            past_kvs = DynamicCache.from_legacy_cache(json.loads(inference_state))
        else:
            past_kvs = None


        if DEBUG >= 4:
            print(f"input_ids: {input_ids}\n")
        
        shard_hidden_states, shard_past_kvs, shard_logits = self.stateful_sharded_model.forward(
            input_ids=input_ids,
            attention_mask=input_attention_mask,
            past_key_values=past_kvs
        )

        if DEBUG >= 4:
            print(f"\nshard_hidden_states: {shard_hidden_states}\n")
            print(f"\nshard_past_kvs {shard_past_kvs}\n")
            print(f"\nshard_logits: {shard_logits}")

        if shard_logits is not None:
            next_token = self.stateful_sharded_model.logits_sample(shard_logits)
            self.past_input_ids = torch.cat([input_ids, next_token[:, None].squeeze(-1)], dim=-1)
            input_ids = next_token

        if shard_past_kvs is not None:
            cache_dict = {
                'key_cache': [tensor.tolist() for tensor in shard_past_kvs.key_cache],
                'value_cache': [tensor.tolist() for tensor in shard_past_kvs_kvs.value_cache]
            }
        else:
            cache_dict = None

        stopping_critera = self.stateful_sharded_model.stopping_critera
        print("set stopping critera")
        self.unfinished_sequences = self.unfinished_sequences & ~stopping_critera(input_ids, None)
        is_finished = self.unfinished_sequences.max() == 0 or input_ids.item() == self.tokenizer.eos_token_id

        if is_finished:
            self.past_input_ids = None

        return_values = (
            input_ids.numpy(force=True) if shard_logits is not None else shard_hidden_states.numpy(force=True),
            json.dumps(cache_dict),
            is_finished
        )

        if DEBUG >= 4:
            print(f"return_values: {return_values}")

        return return_values

    async def infer_tensor(
        self, 
        request_id: str, 
        shard: Shard, 
        input_data: np.ndarray, 
        inference_state: Optional[str] = None
    ) -> Tuple[np.ndarray, str, bool]:
        if DEBUG >= 4:
            print("infer_tensor called")
            print(f"input_data: {input_data}")
            print(f"input_data.size: {input_data.size}")
            print(f"input_data.shape: {input_data.shape}")
            print(f"shard: {self.shard}")

        await self.ensure_shard(shard)
        
        input_ids = torch.tensor(input_data).long().to(self.device)

        if self.past_input_ids is not None:
            self.past_input_ids = torch.cat([self.past_input_ids, input_ids], dim=-1)
        else:
            self.past_input_ids = input_ids

        if inference_state is not None:
            past_kvs = DynamicCache.from_legacy_cache(json.loads(inference_state))
        else:
            past_kvs = None

        if DEBUG >= 4:
            print(f"input_ids: {input_ids}")
            print(f"inference_state: {inference_state}")

        shard_hidden_states, shard_past_kvs, shard_logits = self.stateful_sharded_model.forward(
            input_ids=self.past_input_ids,
            past_key_values=past_kvs
        )

        if shard_logits is not None:
            input_ids = self.stateful_sharded_model.logits_sample(shard_logits)
            
        if shard_past_kvs is not None:
            cache_dict = {
                'key_cache': [tensor.tolist() for tensor in shard_past_kvs.key_cache],
                'value_cache': [tensor.tolist() for tensor in shard_past_kvs_kvs.value_cache]
            }
        else:
            cache_dict = None

        stopping_critera = self.stateful_sharded_model.stopping_critera
        self.unfinished_sequences = self.unfinished_sequences & ~stopping_critera(input_ids, None)
        is_finished = self.unfinished_sequences.max() == 0 or input_ids.item() == self.tokenizer.eos_token_id

        if DEBUG >= 4:
            print(f"\nshard_hidden_states: {shard_hidden_states}\n")
            print(f"\nshard_past_kvs {shard_past_kvs}\n")
            print(f"\nshard_logits: {shard_logits}")

        return_values = (
            input_ids.numpy(force=True) if shard_logits is not None else shard_hidden_states.numpy(force=True),
            json.dumps(cache_dict),
            is_finished
        )

        if DEBUG >= 4:
            print(f"return_values: {return_values}")

        return return_values
                
        
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

        # -- TO DO --
        # Build in shard downloader but requires pulling 
        # apart how TrainedModel loads weight in its __init__ 
        # function in the transformer library
        # model_path = await self.shard_downloader.ensure_shard(shard)
        
        self.tokenizer = await resolve_tokenizer(shard.model_id)
        self.stateful_sharded_model = ShardedHuggingFaceModel(
            shard=shard,
            device=self.device,
            dtype=self.torch_dtype,
            top_k=TOP_K,
            temp=TEMP,
            top_p=TOP_P,
            max_length=MAX_LENGTH,
            max_time=MAX_TIME
        )

        self.shard = shard

        if DEBUG >= 4:
            print(f"Shard loaded successfully: {shard}")
