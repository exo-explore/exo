import torch
import torch.nn as nn
import asyncio
import gc
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    Cache,
    LogitsProcessorList,
    #MinLengthLogitsProcessor,
    LogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
    MaxTimeCriteria
)

from transformers.generation.configuration_utils import (
    GenerationConfig,
    GenerationMode
)

# llama 
from transformers.models.llama.modeling_llama import LlamaModel

# qwen2
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

from exo.api.chatgpt_api import resolve_tokenizer
from typing import Tuple, Optional, Union, List
import re

TEMP = 0.6
TOP_K = 60

class OnionHuggingFaceLM():
    def __init__(self, layers, is_last=False):
        self.layers = layers
        self.is_last = is_last
        self.past_key_values = None
        self.cache_position = None 
        self.position_ids = None 
        self.input_embed = None 
        self.causal_mask = None 
        self.position_embeddings = None
        self.attention_mask = None 
        self.input_ids = None 
        self.hidden_states = None 
        self.next_decoder_cache = None 

    def forward(
            self,
            model,
            llm_model,
            input_ids: Optional[torch.tensor] = None,
            hidden_states: Optional[torch.tensor] = None,
            attention_mask: Optional[torch.tensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            **kwargs
        ) -> Tuple[Optional[torch.tensor], Optional[Union[Cache, List[torch.FloatTensor]]], Optional[torch.tensor]]:

        """
        Generate hidden states or logits via passing through set amount of layers of a model
        To be passed only input_ids OR hidden_state and not both. This is for connecting the model
        layer to generate a complete output

        Args:
            model: base llm model tramsformers class 
            llm_model: llm chat model class 
            input_ids: tensor Optional
            hidden_states: tensor Optional

        Returns:
            Tuple of 
                - hidden_states: tensor Optional
                - past_key_values
                - logits: tensor Optional

        """
        output_attentions = False # outputting attention not needed
        use_legacy_cache = False # some models still use legacy kv store 

        if input_ids is not None and hidden_states is not None:
            raise ValueError

        if hidden_states is not None:
            self.hidden_states = hidden_states

        if input_ids is not None:
            self.input_ids = input_ids

            # embed input_ids
            self.inputs_embeds = model.embed_tokens(self.input_ids)
        
            # cache
            if past_key_values and not isinstance(past_key_values, Cache):
                print("Using legacy cache")
                use_legacy_cache = True
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + self.inputs_embeds.shape[1],
                device=self.inputs_embeds.device
            )
        
            # position id 
            position_ids = cache_position.unsqueeze(0)

            # causal mask
            self.attention_mask = attention_mask
            self.causal_mask = model._update_causal_mask(
                None,
                self.inputs_embeds,
                cache_position,
                past_key_values,
                output_attentions
            )

            #print(f"causal_mask.dim(): {self.causal_mask.dim()}")

            print(f"\ncausal_mask:{self.causal_mask}\n\n")

            # embed positions, some models require and some dont
            if isinstance(model, LlamaModel):
                self.position_embeddings = model.rotary_emb(
                    self.inputs_embeds,
                    position_ids
                )
 
            model_inputs = llm_model.prepare_inputs_for_generation(
                self.input_ids,
                past_key_values=past_key_values,
                attention_mask=self.attention_mask,
                inputs_embeds=self.inputs_embeds,
                position_ids=position_ids,
                cache_position=cache_position
            )

            print(f"model_inputs\n{model_inputs}")

            self.hidden_states = self.inputs_embeds
            self.position_ids = model_inputs["position_ids"]
            self.cache_position = model_inputs["cache_position"]
            self.past_key_values = model_inputs["past_key_values"]


        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                self.hidden_states,
                attention_mask=self.causal_mask,
                position_ids=self.position_ids,
                past_key_values=self.past_key_values,
                use_cache=True,
                cache_position=self.cache_position

            )

            self.hidden_states = layer_outputs[0]
            self.next_decoder_cache = layer_outputs[1]

        if self.is_last:
            self.hidden_states = model.norm(self.hidden_states)

            if use_legacy_cache:
                self.past_key_values = self.next_decoder_cache.to_legacy_cache()
            else:
                self.past_key_values = self.next_decoder_cache
            
            # lm_head  
            logits = llm_model.lm_head(self.hidden_states).to("cuda")

            return (
                None,
                None,
                logits
            )
        
        return (
            self.hidden_states,
            self.past_key_values,
            None
        )

async def model_half_split_test(prompt: str, model_id: str, layers: int):
    """
    Test for splitting in half
    """

    half_layers = int(layers / 2)

    # inference
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    max_length = 512 #tokenizer.model_max_length

    # get llm model
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        use_cache=True
    )
    
    # get base model
    model = llm_model.model

    # add pad token if none, depending on model
    if tokenizer.pad_token == None:
        if re.match(r"Llama|llama", model_id):
            tokenizer.add_special_tokens({"pad_token":"<pad>"})
            model.resize_token_embeddings(len(tokenizer))

    
    # generate input_ids
    messages = [{"role": "user", "content": prompt}]
    txt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([txt], return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    input_attention_mask = inputs.attention_mask.to("cuda") 
    batch_size, seq_length = input_ids.shape[:2]
    
    is_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    logit_runs = 1

    raw_logits = None

    while not is_finished:
        print(f"\n\nLOGIT RUN {logit_runs}\n\n")

        print(f"input_ids:\n{input_ids}\n")
        print(input_ids.shape)

        print("\n first half of layers")
        shard_layers = nn.ModuleList(model.layers[:half_layers])#.to("cuda")
        #shard_layers = nn.ModuleList(model.layers)
        sharded_model = OnionHuggingFaceLM(layers=shard_layers)
        #sharded_model.is_last = True

        # generate first half
        # add if first layer of model check
        shard_hidden_states, shard_past_kvs, shard_logits = sharded_model.forward(
            model=model,
            llm_model=llm_model,
            attention_mask=input_attention_mask,
            input_ids=input_ids,
            hidden_states=None
        )

        # second half
        print(f"\n second half of layers")
        sharded_model.layers = nn.ModuleList(model.layers[half_layers:])
        sharded_model.is_last = True 

        shard_hidden_states, shard_past_kvs, shard_logits = sharded_model.forward(
            model=model,
            llm_model=llm_model,
            hidden_states=shard_hidden_states,
            past_key_values=shard_past_kvs
        )

        # this part of the generation and _sample functions for transformers GenerationMixin
        # ref: https://github.com/huggingface/transformers/blob/0a55d9f7376f72ad3ff296d4249840021b03bcc4/src/transformers/generation/utils.py#L1301
        
        # clone logit sample 
        logits = shard_logits[:, -1, :].clone().float()

        raw_logits = logits 

        # distribute
        logits_processor = LogitsProcessorList([
            TopKLogitsWarper(35),
            TemperatureLogitsWarper(0.6),
            TopPLogitsWarper(0.8)
        ])

        stopping_critera = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=50),
                MaxTimeCriteria(max_time=10.0),
            ]
        )
        
        next_token_scores = logits_processor(input_ids, logits)

        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        #next_tokens = torch.argmax(next_token_scores, dim=-1)

        # get inputs ready incase not finished 
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        unfinished_sequences = unfinished_sequences & ~stopping_critera(input_ids, None)
        is_finished = unfinished_sequences.max() == 0

        print(f"is_finished?:\n{is_finished}\n")

        logit_runs += 1

        del logits
        del shard_logits

    print(f"model.generation_config\n{llm_model.generation_config}")

    generated_text = tokenizer.batch_decode(
        input_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

    print(f"generated_text:\n{generated_text}\n")

    # free model from memory
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    #prompt = "In a single word only, what is the last name of the current president of the USA?"
    prompt = "What color is the sky? Explain why"
    #prompt = "In a single word only, what is the color of an apple?"

    #print("\n-------- Test TinyLlama/TinyLlama_v1.1 ----------\n")
    #model_id = "TinyLlama/TinyLlama_v1.1"
    #model_layers = 22

    #asyncio.run(
    #   model_half_split_test(
    #        prompt=prompt,
    #        model_id=model_id,
    #        layers=model_layers
    #    )
    #)

    #print("\n-------- Test meta-llama/Meta-Llama-3.1-8B ----------\n")
    #model_id = "meta-llama/Meta-Llama-3.1-8B"
    #model_layers = 32

    #asyncio.run(
    #    model_half_split_test(
    #        prompt=prompt,
    #        model_id=model_id,
    #        layers=model_layers
    #    )
    #)

    print("\n-------- Test Qwen/Qwen2-0.5B-Instruct ----------\n")
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    model_layers = 24

    asyncio.run(
        model_half_split_test(
            prompt=prompt,
            model_id=model_id,
            layers=model_layers
        )
    )

