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

from exo.api.chatgpt_api import resolve_tokenizer
from typing import Tuple, Optional
import re
from exo.inference.pytorch.utils import sample_logits, top_k_sampling

TEMP = 0.6
TOP_K = 60

class OnionHuggingFaceLM():
    def __init__(self, layers, is_last=False):
        self.layers = layers
        self.is_last = is_last

    def forward(
            self,
            model,
            llm_model,
            input_ids: Optional[torch.tensor],
            hidden_states: Optional[torch.tensor],
            attention_mask: torch.tensor=None,
            past_key_values: Cache=DynamicCache(),
            **kwargs
        ) -> Tuple[Optional[torch.tensor], Optional[torch.tensor], Optional[Cache]]:

        """
        Generate hidden states or logits via passing through set amount of layers of a model
        To be passed only input_ids OR hidden_state and not both. This is for connecting the model
        layer to generate a complete output

        Args:
            input_ids: tensor Optional
            hidden_states: tensor Optional

        Returns:
            Tuple of 
                - hidden_states: tensor Optional
                - logits: tensor Optional

        """
        is_first = False

        if input_ids is not None and hidden_states is not None:
            raise ValueError

        if input_ids is not None:
            # embed input_ids
            input_ids = model.embed_tokens(input_ids)
            # calculate position_ids
            batch_size, seq_length = input_ids.shape[:2]

            is_first = True

        if hidden_states is not None:
            batch_size, seq_length = hidden_states.shape[:2] 
        
        # cache
        past_key_values_length = len(past_key_values)
        cache_position = torch.arange(
            past_key_values_length, 
            seq_length + past_key_values_length, 
            dtype=torch.long,
            device=input_ids.device if input_ids is not None else hidden_states.device
        )

        position_ids = cache_position.unsqueeze(0)

        if is_first:
            model_inputs = llm_model.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_position=cache_position,
                attention_mask=attention_mask
            )

            print(f"model_inputs\n{model_inputs}")


        for layer in self.layers:
            layer_input = input_ids if input_ids is not None else hidden_states
            #print(f"INPUT: \n{layer_input}\n")
            #print(f"POSITION_IDS: \n{position_ids}\n")
            #print(f"LAYER: \n{layer}\n")
            layer_outputs = layer(
                model_inputs["input_ids"],
                position_ids=model_inputs["position_ids"],
                #attention_mask=model_inputs["attention_mask"],
                past_key_values=model_inputs["past_key_values"],
                return_dict=True,
                use_cache=True
            )

        hidden_states = layer_outputs[0]
        past_key_values = layer_outputs[1]

        if self.is_last:
            norm_states = model.norm(hidden_states)
            
            # lm_head  
            logits = llm_model.lm_head(norm_states).to("cuda")

            return (None, logits, past_key_values)
        
        return (hidden_states, None, past_key_values)

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

        #shard_layers = nn.ModuleList(model.layers[:half_layers])#.to("cuda")
        shard_layers = nn.ModuleList(model.layers)
        sharded_model = OnionHuggingFaceLM(layers=shard_layers)
        sharded_model.is_last = True

        # generate first half
        # add if first layer of model check
        shard_hidden_states, shard_logits, shard_past_kvs = sharded_model.forward(
            model=model,
            llm_model=llm_model,
            attention_mask=input_attention_mask,
            input_ids=input_ids,
            hidden_states=None
        )

        # second half
        #sharded_model.layers = nn.ModuleList(model.layers[half_layers:])
        #sharded_model.is_last = True 

        #shard_hidden_states, shard_logits, shard_past_kvs = sharded_model.forward(
        #    model=model,
        #    llm_model=llm_model,
        #    input_ids=None,
        #    hidden_states=shard_hidden_states,
        #    past_key_values=shard_past_kvs
        #)

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
    prompt = "In a single word only, what is the color of an apple?"

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

