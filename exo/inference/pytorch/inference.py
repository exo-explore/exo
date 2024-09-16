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
TOP_K = 35
TEMP = 0.6
TOP_P = 0.8

class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    """
    PyTorch Dynamic Shard Inference Engine for performing model inference with sharded models.
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
            input_ids = self.stateful_sharded_model.logits_sample(input_ids, shard_logits)
            print(input_ids)
            
        if shard_past_kvs is not None:
            cache_dict = {
                'key_cache': [tensor.tolist() for tensor in shard_past_kvs.key_cache],
                'value_cache': [tensor.tolist() for tensor in shard_past_kvs_kvs.value_cache]
            }
        else:
            cache_dict = None

        stopping_critera = self.stateful_sharded_model.stopping_critera
        self.unfinished_sequences = self.unfinished_sequences & ~stopping_critera(input_ids, None)
        is_finished = self.unfinished_sequences.max() == 0

        return (
            input_ids.numpy(force=True) if shard_logits is not None else shard_hidden_states.numpy(force=True),
            json.dumps(cache_dict),
            is_finished
        )

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

        if input_data.size == 1:
            hidden_states = torch.tensor(input_data).to(self.device)
            hidden_states = hidden_states.unsqueeze(0)
        else:
            hidden_states = torch.tensor(input_data).long().to(self.device)

        if inference_state is not None:
            past_kvs = DynamicCache.from_legacy_cache(json.loads(inference_state))
        else:
            past_kvs = None

        if DEBUG >= 4:
            print(f"hidden_states: {hidden_states}")
            print(f"inference_state: {inference_state}")

        shard_hidden_states, shard_past_kvs, shard_logits = self.stateful_sharded_model.forward(
            input_ids=hidden_states,
            past_key_values=past_kvs,
            infer_tensor=True
        )

        if shard_logits is not None:
            input_ids = self.stateful_sharded_model.logits_sample(hidden_states, shard_logits)
            
        if shard_past_kvs is not None:
            cache_dict = {
                'key_cache': [tensor.tolist() for tensor in shard_past_kvs.key_cache],
                'value_cache': [tensor.tolist() for tensor in shard_past_kvs_kvs.value_cache]
            }
        else:
            cache_dict = None

        stopping_critera = self.stateful_sharded_model.stopping_critera
        self.unfinished_sequences = self.unfinished_sequences & ~stopping_critera(input_ids, None)
        is_finished = self.unfinished_sequences.max() == 0

        return (
            input_ids.numpy(force=True) if shard_logits is not None else shard_hidden_states.numpy(force=True),
            json.dumps(cache_dict),
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

        # need to build in shard downloader
        # model_path = await self.shard_downloader.ensure_shard(shard)
        
        if self.stateful_sharded_model:
            print("Deleting model")
            del self.stateful_sharded_model
        #    gc.collect()
        #    torch.cuda.empty_cache()
        
        self.tokenizer = await resolve_tokenizer(shard.model_id)
        self.stateful_sharded_model = ShardedHuggingFaceModel(shard, self.device, self.torch_dtype)
        self.shard = shard

        if DEBUG >= 4:
            print(f"Shard loaded successfully: {shard}")
