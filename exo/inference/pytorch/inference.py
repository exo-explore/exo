# experimental, based off of tinygrad/inference.py
import numpy as np
import torch
import numpy as np
import json
from typing import Optional, Tuple
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.inference.pytorch.model.archive.hf_manual import ShardedHuggingFaceModel
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
        self.unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device) 

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
        input_ids = inputs.input_ids.to("cuda")
        input_attention_mask = inputs.attention_mask.to("cuda") 
        batch_size, seq_length = input_ids.shape[:2]
       
        if DEBUG >= 4:
            print(f"input_ids: {input_ids}\n")
            print(f"layer_count: {self.shard.get_layer_count()}")
            print(f"is_first_layer: {self.shard.is_first_layer()}")
            print(f"is_last_layer: {self.shard.is_last_layer()}")
        
        output_data = self.stateful_sharded_model.forward(
            input_ids
        )

        is_finished = output_data.size == 1 and output_data.item() in [self.tokenizer.eos_token_id]

        if DEBUG >= 4:
            print(f"output_data: {output_data}\n")
            print(f"output_data.size {output_data.size}\n")
            
            print(f"finished: {is_finished}")
            print(f"self.tokenizer.eos_token_id {self.tokenizer.eos_token_id}")
            print(f"output_data[-1] {output_data[-1]}")

            if output_data.size == 1:
                print(f"size 1 output_data.item() {output_data.item()}")
                print(f"output_data.item() in [self.tokenizer.eos_token_id]: {output_data.item() in [self.tokenizer.eos_token_id]}")

        cache_dict = {
            'key_cache': [tensor.tolist() for tensor in current_kvs.key_cache],
            'value_cache': [tensor.tolist() for tensor in current_kvs.value_cache]
        }

        return (
            output_data,
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
        if DEBUG >= 3:
            print("infer_tensor called")
            print(f"input_data: {input_data}")
            print(f"input_data.size: {input_data.size}")
            print(f"input_data.shape: {input_data.shape}")
            print(f"shard: {self.shard}")

        await self.ensure_shard(shard)

        current_kvs = None


        if input_data.size == 1:
            in_tensor = torch.tensor([[input_data.item()]]).to(self.device)
        else:
            in_tensor = torch.tensor(input_data).to(self.device)

        # in_tensor = torch.tensor(input_data).to(self.device)
        
        # in_tensor = self.stateful_sharded_model.embed_tokens(in_tensor)
            
        # convert inference_state or cache from json to DynamicCache
        past_kv = DynamicCache()
        if inference_state != None:
            try:
                cache_dict = json.loads(inference_state)
                past_kv.key_cache = [torch.tensor(data).to(self.device) for data in cache_dict['key_cache']]
                past_kv.value_cache = [torch.tensor(data).to(self.device) for data in cache_dict['value_cache']]
                past_kv_length = past_kv[0][0].shape[2]
            except json.JSONDecodeError:
                print(f"ERROR DECODING INFERENCE STATE")

        if DEBUG >= 3:
            # print(f"input_tensor: {in_tensor}")
            print(f"layer_count: {self.shard.get_layer_count()}")
            print(f"is_first_layer: {self.shard.is_first_layer()}")
            print(f"is_last_layer: {self.shard.is_last_layer()}")
            print(f"input_data.shape: {input_data.shape}")

            print(f"in_tensor: {in_tensor}")
        output_data, current_kvs = self.stateful_sharded_model.forward(
            in_tensor,
            None,
            past_kv
        )

        is_finished = output_data.size == 1 and output_data.item() in [self.tokenizer.eos_token_id]

        if DEBUG >= 3:
            print(f"output_data: {output_data}\n")
            print(f"output_data.size {output_data.size}\n")
            print(f"finished: {is_finished}")
            print(f"self.tokenizer.eos_token_id {self.tokenizer.eos_token_id}")
            print(f"output_data[-1] {output_data[-1]}")
            print("====================================================")

            if output_data.size == 1:
                print(f"size 1 output_data.item() {output_data.item()}")
                print(f"output_data.item() in [self.tokenizer.eos_token_id]: {output_data.item() in [self.tokenizer.eos_token_id]}")

        
        cache_dict = {
            'key_cache': [tensor.tolist() for tensor in current_kvs.key_cache],
            'value_cache': [tensor.tolist() for tensor in current_kvs.value_cache]
        }

        return (
            output_data,
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
        
        self.tokenizer = await resolve_tokenizer(shard.model_id)
        self.stateful_sharded_model = ShardedHuggingFaceModel(shard)
        self.shard = shard

        if DEBUG >= 4:
            print(f"Shard loaded successfully: {shard}")
