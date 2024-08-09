# experimental, based off of tinygrad/inference.py

import numpy as np
import torch
import numpy as np
from typing import Optional, Callable, Tuple
from transformers import AutoTokenizer
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.inference.pytorch.model.hf import ShardedHuggingFaceModel
from exo.helpers import DEBUG

# Default settings
TEMPERATURE = 0.7
TOP_K = 50

class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    """
    PyTorch Dynamic Shard Inference Engine for performing model inference with sharded models.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the inference engine.

        Args:
            debug (bool): If True, enables debug logging. Defaults to False.
        """
        self.shard = None
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def infer_prompt(
        self, 
        request_id: str, 
        shard: Optional[Shard] = None, 
        prompt: str = "", 
        image_str: Optional[str] = None, 
        inference_state: Optional[str] = None
    ) -> Tuple[np.ndarray, str, bool]:

        # Ensure the shard is loaded
        await self.ensure_shard(shard)

        # Tokenize the prompt
        tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # Load the past key values from the inference state if available
        # past_key_values = self._load_kv_cache(inference_state)

        # Run the forward pass through the model layers
        # output_data, past_key_values
        if DEBUG >= 2:
            print(f"tokens: {tokens}\n")

        
        
        output_data = self.model.forward_layers(
            tokens,
            "prompt"
            # past_key_values=past_key_values
        )

        # Save the past key values to the inference state
        # self._save_kv_cache(past_key_values)

        is_finished = output_data.size() == 1

        if DEBUG >= 2:
            print("infer_prompt called")
            print(f"Output data: {output_data} output_data.size: {output_data.size()}")
            print(f"finished: {is_finished}")
            print(f"self.tokenizer.eos_token_id {self.tokenizer.eos_token_id}")

        with torch.no_grad():
            output_npa = np.array(output_data.cpu())

        return (
            output_npa,
            "",
            is_finished
        )

    async def infer_tensor(
        self, 
        request_id: str, 
        shard: Shard, 
        input_data: np.ndarray, 
        inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        
        in_tensor = torch.tensor(input_data)
        if DEBUG >= 2:
            print("infer_tensor called")
            print(f"input_data: {input_data}\n")
            print(f"in_tensor: {in_tensor}\n")

        # Ensure the shard is loaded
        await self.ensure_shard(shard)

        # Run the forward pass through the model layers
        # output_data, past_key_values
        
        output_data = self.model.forward_layers(
            in_tensor,
            "tensor"
            # past_key_values=past_key_values
        )

        is_finished = output_data.size == 1

        if DEBUG >= 2:
            print(f"Output data: {output_data} finished: {is_finished}")


        with torch.no_grad():
            output_npa = np.array(output_data.cpu())

        return (
            output_npa,
            "",
            is_finished
        )

    # def _load_kv_cache(self, past_key_values_list):
    #     """
    #     Load key-value cache from the inference state.

    #     Args:
    #         past_key_values_list (list): List of past key-value tensors.

    #     Returns:
    #         list: List of loaded past key-value tensors.
    #     """
    #     if past_key_values_list is None:
    #         return []
    #     return [torch.tensor(kv, device=self.device) for kv in past_key_values_list]

    # def _save_kv_cache(self, past_key_values):
    #     """
    #     Save key-value cache to the inference state.

    #     Args:
    #         past_key_values (list): List of past key-value tensors.

    #     Returns:
    #         list: List of key-value tensors in a format suitable for saving.
    #     """
    #     if past_key_values is None:
    #         return []
        
    #     new_cache = []
    #     for kv in past_key_values:
    #         if kv:
    #             new_cache.append(kv.cpu().tolist())

    #     return new_cache

    async def ensure_shard(self, shard: Optional[Shard]):
        """
        Ensure the model shard is loaded and ready for inference.

        Args:
            shard (Optional[Shard]): Shard information for the model.
        """
        if self.shard == shard:
            return

        if DEBUG >= 2:
            print(f"Loading new shard: {shard}")

        self.model = ShardedHuggingFaceModel(shard)
        self.tokenizer = AutoTokenizer.from_pretrained(
            shard.model_id,
            add_eos_token=True,
            use_fast=True
        )
        self.shard = shard

        if DEBUG >= 2:
            print(f"Shard loaded successfully: {shard}")

    def set_on_download_progress(self, on_download_progress: Callable[[int, int], None]):
        """
        Set a callback function to track download progress.

        Args:
            on_download_progress (Callable[[int, int], None]): Callback function to track progress.
        """
        # must have this function or inference engine breaks
        # This method can be implemented if progress tracking is needed
        pass
