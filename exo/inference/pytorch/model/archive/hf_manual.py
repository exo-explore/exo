# Attempted version to recreate manually using LlamaModel and others
# BROKEN
import torch
import numpy as np
from transformers import AutoModelForCausalLM, DynamicCache, Cache, AutoModel
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from typing import Tuple, Optional, Union, List
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from exo.inference.pytorch.model.archive.utils import sample_logits

TOP_P = 0.7 #0.95
TOP_K = 50
TEMP = 0.01


class ShardedHuggingFaceModel(torch.nn.Module):
    def __init__(self, shard: Shard):
        super(ShardedHuggingFaceModel, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")

        self.shard = shard

        # Load the model
        try:
            self.base_model = AutoModel.from_pretrained(
                shard.model_id,
                torch_dtype=torch.float32,
                device_map="auto",
                # offload_buffers=True
            )

            # disk_offload(model=self.base_model, offload_dir="./.offload")
        except Exception as err:
            print(f"Error loading model: {err}")
            raise

        if DEBUG >= 2:
            print(f"\nShardedHuggingFaceModel init with shard {shard}")
            print(f"self.base_model: {self.base_model}")

        # Embeddings and final layer norm
        # used for doing what forward LlamaModel does in transformers
        self.norm = self.base_model.norm
        self.lm_head = torch.nn.Linear(
            self.base_model.config.hidden_size,
            self.base_model.config.vocab_size,
            bias=False
        ).to(self.device)
        self.embed_tokens = self.base_model.embed_tokens
    
    def forward(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor = None,
        past_kvs: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    ) -> Tuple[np.ndarray, any]:
        """
        Forward through layers using the base model

        Args:
            input_ids: tensor input
            attention_mask: attention mask from tokenizer
            past_kvs: past key value stores for cache
        
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

        if self.shard.is_first_layer():
            if DEBUG >= 2:
                print("first layer, embed")
                print(f"input_ids: {input_ids}")
            input_ids = self.embed_tokens(input_ids)

            if DEBUG >= 2:
                print(f"embeded input_ids: {input_ids}")

        if attention_mask == None:
            # get attention mask
            past_kv_length = len(past_kvs)
            batch_size, seq_length = input_ids.shape[:2]
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (batch_size, seq_length), input_ids, past_kv_length
            )

        past_kvs = DynamicCache.from_legacy_cache(past_kvs)
        past_seen_tokens = past_kvs.get_seq_length() if past_kvs is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, 
            past_seen_tokens + input_ids.shape[1], 
            device=self.device
        )

        position_ids = cache_position.unsqueeze(0).to(self.device)

        try:
            position_embeddings = self.base_model.rotary_emb(
                input_ids,
                position_ids
            )
        except Exception as err:
            print(f"rotary_emb not found in base_model")
            position_embeddings = None

        causal_mask = self.base_model._update_causal_mask(
            attention_mask,
            input_ids,
            cache_position,
            past_kvs,
            self.base_model.config.output_attentions
        )

        # progress through layers
        for i in range(self.shard.start_layer, self.shard.end_layer + 1):
            decoder_layer = self.base_model.layers[i]

            if DEBUG >= 4:
                print("Going through layer")
                print(f"{decoder_layer}")
                print("input_ids")
                print(f"{input_ids}")
                print("causal_mask")
                print(f"{causal_mask}")

            try:
                layer_outputs = decoder_layer(
                    input_ids,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_value=past_kvs,
                    use_cache=True,
                    cache_position=cache_position,
                    output_logits=True
                )
            except Exception as err:
                print(f"Going through layer failed: {err}")
                print(err.__traceback__.tb_lineno)
                raise

        hidden_states = layer_outputs[0]
        next_kvs = layer_outputs[1]

        if DEBUG >= 3:
            print(f"layer_outputs {layer_outputs}")
            print(layer_outputs[1:])
        
        if self.shard.is_last_layer():
            hs_norm = self.norm(hidden_states).to(self.device)
            # hs_lm_head = self.base_model.lm_head(hs_norm).float()

            # Use the sampling function with default settings
            with torch.no_grad():
                logits = self.lm_head(
                    hs_norm[:, -1:, :]
                ).to(self.device).float()

                if DEBUG >= 2:
                    print(f"hs_norm: {hs_norm}")
                    # print(f"hs_lm_head: {hs_lm_head}")
                    print(f"logits: {logits}")
                    print(f"logits.shape: {logits.shape}")

                # output_token = sample_logits(
                #     logits,
                #     TEMP,
                #     TOP_P,
                #     TOP_K
                # ).unsqueeze(0).unsqueeze(0).long()

                output_token = torch.distributions.Categorical(
                    logits=logits
                ).sample(sample_shape=(1,))

            if DEBUG >= 2:
                print(f"output_token: {output_token}")

            return (output_token.numpy(force=True), next_kvs)
        
        with torch.no_grad():
            out_hidden_states = hidden_states.float().numpy(force=True)

        return (
            out_hidden_states,
            next_kvs
        )