import torch
import numpy as np
from transformers import AutoModelForCausalLM, DynamicCache, Cache
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from typing import Tuple, Optional, Union, List
from exo.inference.pytorch.model.utils import sample_logits

TOP_P = 0.75 #0.95
TOP_K = 20
TEMP = 0.8

class ShardedHuggingFaceModel(torch.nn.Module):
    def __init__(self, shard: Shard, tokenizer: any):
        super(ShardedHuggingFaceModel, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")

        self.shard = shard
        self.tokenizer = tokenizer

        # Load the model
        try:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                shard.model_id,
                torch_dtype=torch.float32,
                device_map="auto",
                # offload_buffers=True
            )

            # disk_offload(model=self.llm_model, offload_dir="./.offload")

            self.base_model = self.llm_model.model
        except Exception as err:
            print(f"Error loading model: {err}")
            raise

        if DEBUG >= 2:
            print(f"\nShardedHuggingFaceModel init with shard {shard}")
            print(f"self.llm_model: {self.llm_model}")
            print(f"self.base_model: {self.base_model}")

        if DEBUG >= 2:
            print(f"full_model.model layer: {len(self.base_model.layers)}")

        # Embeddings and final layer norm
        # used for doing what forward LlamaModel does in transformers
        self.norm = self.base_model.norm
        self.lm_head = self.llm_model.lm_head
        self.embed_tokens = self.base_model.embed_tokens
    
    def forward(
        self,
        input_ids: torch.tensor,
        past_kvs: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    ) -> Tuple[np.ndarray, any]:
        """
        Forward through layers using the base model

        Args:
            input_ids: tensor input
            past_kvs: past key value stores for cache
            use_cache: use cache
        
        Returns:
            hidden_states: numpy of states between layers
            or logits: numpy of normalization and linearization of last hidden state
            past_kvs: DynamicCache of past key values if use_cache is true

        Ref:
            https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py#L804
            https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L887
        """
        if DEBUG >= 4:
            print("forward called")
            print(f"input_ids: {input_ids}\n")
            print(f"layer_count: {self.shard.get_layer_count()}")
            print(f"is_first_layer: {self.shard.is_first_layer()}")
            print(f"is_last_layer: {self.shard.is_last_layer()}")

        past_kvs = DynamicCache.from_legacy_cache(past_kvs)
        past_seen_tokens = past_kvs.get_seq_length() if past_kvs is not None else 0

        cache_position = torch.arange(
            past_seen_tokens, 
            past_seen_tokens + input_ids.shape[1], 
            device=input_ids.device
        ).to(self.device)

        position_ids = cache_position.unsqueeze(0).to(self.device)

        try:
            position_embeddings = self.base_model.rotary_emb(
                input_ids,
                position_ids
            )
        except Exception as err:
            print(f"rotary_emb not found in base_model")
            position_embeddings = None

        # progress through layers
        for i in range(self.shard.start_layer, self.shard.end_layer + 1):
            decoder_layer = self.base_model.layers[i]

            if DEBUG >= 4:
                print("Going through layer")
                print(f"{decoder_layer}")

            layer_outputs = decoder_layer(
                input_ids,
                position_ids=position_ids if not position_embeddings else None,
                position_embeddings=position_embeddings,
                past_key_value=past_kvs,
                use_cache=True,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]
        next_kvs = layer_outputs[1]

        if DEBUG >= 3:
            print(f"layer_outputs {layer_outputs}")
        
        if self.shard.is_last_layer():
            hs_norm = self.norm(hidden_states)
            hs_lm_head = self.llm_model.lm_head(hs_norm).float()

            # Use the sampling function with default settings
            with torch.no_grad():
                output_token = sample_logits(
                    hs_lm_head[:, -1, :],
                    TEMP,
                    TOP_P,
                    TOP_K
                ).numpy(force=True).flatten()

            if DEBUG >= 2:
                print(f"hs_norm: {hs_norm}")
                print(f"hs_lm_head: {hs_lm_head}")
                print(f"output_token: {output_token}")

            return (output_token, next_kvs)
        
        with torch.no_grad():
            out_hidden_states = hidden_states.numpy(force=True)

        return (
            out_hidden_states,
            next_kvs
        )