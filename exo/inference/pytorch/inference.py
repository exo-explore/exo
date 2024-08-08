# experimental, based off of tinygrad/inference.py

import json
import torch
import numpy as np
from typing import Optional, Callable, Tuple
from transformers import AutoTokenizer
from exo.inference.shard import Shard
from exo.inference.inference_engine import InferenceEngine
from exo.inference.pytorch.model.hf import ShardedHuggingFaceModel

# Default settings
TEMPERATURE = 0.7
TOP_K = 50

class PyTorchDynamicShardInferenceEngine(InferenceEngine):
    """
    PyTorch Dynamic Shard Inference Engine for performing model inference with sharded models.
    """

    def __init__(self, debug: bool = True):
        """
        Initialize the inference engine.

        Args:
            debug (bool): If True, enables debug logging. Defaults to False.
        """
        self.shard = None
        self.model = None
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def infer_prompt(
            self, 
            request_id: str, 
            shard: Optional[Shard] = None, 
            prompt: str = "", 
            image_str: Optional[str] = None, 
            inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        """
        Perform inference based on a text prompt.

        Args:
            request_id (str): Unique identifier for the request.
            shard (Optional[Shard]): Shard information for the model.
            prompt (str): The input text prompt for inference.
            image_str (Optional[str]): Optional image string for multi-modal models.
            inference_state (Optional[str]): The previous inference state.

        Returns:
            Tuple[np.ndarray, str, bool]: The output data, new inference state, and end-of-sequence flag.
        """
        await self.ensure_shard(shard)

        toks = self.tokenizer.encode(prompt)
        start_pos = json.loads(inference_state).get("start_pos", 0) if inference_state else 0
        past_key_values_list = json.loads(inference_state).get("past_key_values", None) if inference_state else None
        past_key_values = self._load_kv_cache(past_key_values_list)

        start_pos = self.model.prefill(torch.tensor(toks[:-1], device=self.model.device), start_pos=start_pos)
        last_tok = torch.tensor([toks[-1]], device=self.model.device).unsqueeze(0)

        output_data, past_key_values = self.model.forward_layers(last_tok, past_key_values=past_key_values)
        output_data = output_data.detach().cpu().numpy()

        if output_data.size == 1:
            start_pos += 1

        return (
            output_data,
            json.dumps({"start_pos": start_pos, "past_key_values": self._save_kv_cache(past_key_values)}),
            output_data.size == 1 and output_data.item() in [self.tokenizer.eos_token_id],
        )

    async def infer_tensor(
            self, 
            request_id: str, 
            shard: Optional[Shard] = None, 
            input_data: np.ndarray = None, 
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

        input_tensor = torch.tensor(input_data).unsqueeze(0).to(self.model.device)

        start_pos = json.loads(inference_state).get("start_pos", 0) if inference_state else 0
        past_key_values_list = json.loads(inference_state).get("past_key_values", None) if inference_state else None
        past_key_values = self._load_kv_cache(past_key_values_list)

        output_data, past_key_values = self.model.forward_layers(input_tensor, past_key_values=past_key_values)
        output_data = output_data.detach().cpu().numpy()

        if output_data.size == 1:
            start_pos += 1

        return (
            output_data,
            json.dumps({"start_pos": start_pos, "past_key_values": self._save_kv_cache(past_key_values)}),
            output_data.size == 1 and output_data.item() in [self.tokenizer.eos_token_id],
        )

    def _load_kv_cache(self, past_key_values_list):
        """
        Load key-value cache from the inference state.

        Args:
            past_key_values_list (list): List of past key-value tensors.

        Returns:
            list: List of loaded past key-value tensors.
        """
        if past_key_values_list is None:
            return []
        return [torch.tensor(kv, device=self.device) for kv in past_key_values_list]

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

        self.model = ShardedHuggingFaceModel(shard)
        self.tokenizer = AutoTokenizer.from_pretrained(shard.model_id)
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
