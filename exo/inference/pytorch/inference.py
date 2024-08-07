# experimental, based off of tinygrad/inference.py

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, Tuple
from transformers import AutoTokenizer, Cache
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.inference.pytorch.model.hf import ShardedHuggingFaceModel
import logging

logging.basicConfig()
logging.getLogger("pytorch.inference").setLevel(logging.DEBUG)

# Default settings
TEMPERATURE = 0.7
TOP_K = 50

class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    """
    PyTorch Dynamic Shard Inference Engine for performing model inference with sharded models.
    """

    def __init__(self, model_name: str, debug: bool = True):
        """
        Initialize the inference engine.

        Args:
            debug (bool): If True, enables debug logging. Defaults to False.
        """
        self.shard = None
        self.model = None
        self.model_name = model_name if model_name else "meta-llama/Meta-Llama-3-8B"
        self.debug = debug
        self.log = logging.getLogger("pytorch.inference")
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def infer_prompt(
            self, 
            request_id: str, 
            shard: Optional[Shard], 
            prompt: str, 
            inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        """
        Perform inference based on a text prompt.

        Args:
            request_id (str): Unique identifier for the request.
            shard (Optional[Shard]): Shard information for the model.
            prompt (str): The input text prompt for inference.
            inference_state (Optional[str]): The previous inference state.

        Returns:
            Tuple[np.ndarray, str, bool]: The output data, new inference state, and end-of-sequence flag.
        """
        await self.ensure_shard(shard)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cpu")
        
        # Continue the sequence if inference state exists
        past_key_values = None
        if inference_state:
            past_key_values = self._load_kv_cache(json.loads(inference_state).get("past_key_values"))

        output, past_key_values = self.model.full_model(input_ids, past_key_values=past_key_values)

        if self.shard.is_last_layer():
            logits = self._apply_generation_settings(output, TEMPERATURE, TOP_K)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            output_data = np.array([next_token.item()])
            is_eos = next_token.item() == self.tokenizer.eos_token_id
        else:
            output_data = output.cpu().numpy()
            is_eos = False

        new_inference_state = json.dumps({"past_key_values": self._save_kv_cache(past_key_values)})

        if self.debug:
            self.log.info(
                f"Infer Prompt Debug - Request ID: {request_id}, Output: {output_data}, EOS: {is_eos}")

        return output_data, new_inference_state, is_eos

    async def infer_tensor(
            self, 
            request_id: str, 
            shard: Optional[Shard], 
            input_data: np.ndarray, 
            inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        """
        Perform inference based on an input tensor.

        Args:
            request_id (str): Unique identifier for the request.
            shard (Optional[Shard]): Shard information for the model.
            input_data (np.ndarray): The input tensor for inference.
            inference_state (Optional[str]): The previous inference state.

        Returns:
            Tuple[np.ndarray, str, bool]: The output data, new inference state, and end-of-sequence flag.
        """
        await self.ensure_shard(shard)

        input_tensor = torch.tensor(input_data).unsqueeze(0).to(self.device)
        
        # Continue the sequence if inference state exists
        past_key_values = None
        if inference_state:
            past_key_values = self._load_kv_cache(json.loads(inference_state).get("past_key_values"))

        output, past_key_values = self.model(input_tensor, past_key_values=past_key_values)

        if self.shard.is_last_layer():
            logits = self._apply_generation_settings(output, TEMPERATURE, TOP_K)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            output_data = np.array([next_token.item()])
            is_eos = next_token.item() == self.tokenizer.eos_token_id
        else:
            output_data = output.cpu().numpy()
            is_eos = False

        new_inference_state = json.dumps({"past_key_values": self._save_kv_cache(past_key_values)})

        if self.debug:
            self.log.info(f"Infer Tensor Debug - Request ID: {request_id}, Output: {output_data}, EOS: {is_eos}")

        return output_data, new_inference_state, is_eos

    def _apply_generation_settings(self, logits, temperature, top_k):
        """
        Apply temperature and top_k settings to logits.

        Args:
            logits (torch.Tensor): The logits to be adjusted.
            temperature (float): The temperature setting for generation.
            top_k (int): The top_k setting for generation.

        Returns:
            torch.Tensor: The adjusted logits.
        """
        logits = logits / temperature
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
            logits = logits.scatter(1, top_k_indices, top_k_values)
        return logits

    def _load_kv_cache(self, past_key_values_list):
        """
        Load key-value cache from the inference state.

        Args:
            past_key_values_list (list): List of past key-value tensors.

        Returns:
            Cache: Loaded past key-value cache.
        """
        if past_key_values_list is None:
            return Cache()
        cache = Cache()
        for kv in past_key_values_list:
            cache.append(torch.tensor(kv, device=self.device))
        return cache

    def _save_kv_cache(self, past_key_values):
        """
        Save key-value cache to the inference state.

        Args:
            past_key_values (list): List of past key-value tensors.

        Returns:
            list: List of key-value tensors in a format suitable for saving.
        """
        return [kv.cpu().tolist() for kv in past_key_values]

    async def ensure_shard(self, shard: Optional[Shard]):
        """
        Ensure the model shard is loaded and ready for inference.

        Args:
            shard (Optional[Shard]): Shard information for the model.
        """
        if self.shard == shard:
            return

        # Load model and tokenizer from the downloaded files
        # This is written for llama model but need to add in option for others
        if not self.model:
            self.model = ShardedHuggingFaceModel(self.model_name, shard)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.shard = shard

    def set_on_download_progress(self, on_download_progress: Callable[[int, int], None]):
        """
        Set a callback function to track download progress.

        Args:
            on_download_progress (Callable[[int, int], None]): Callback function to track progress.
        """
        # must have this function or inference engine breaks
        # This method can be implemented if progress tracking is needed
        pass
