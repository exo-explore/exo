import torch
import torch.nn as nn
import asyncio
import gc
import json
from transformers import AutoConfig, AutoModel
from safetensors import safe_open
from typing import Tuple, Optional
import re
from exo.inference.pytorch.utils import sample_logits, top_k_sampling
from exo.api.chatgpt_api import resolve_tokenizer

TEMP = 0.6
TOP_K = 60

class OnionHuggingFaceLM():
    def __init__(self, layers, safetensor_index_file, safetensor_directory, is_last=False):
        self.layers = layers
        self.is_last = is_last
        self.safetensor_index_file = safetensor_index_file
        self.safetensor_directory = safetensor_directory

        # Load the safetensor index JSON
        with open(safetensor_index_file, "r") as f:
            self.index_data = json.load(f)
        self.weight_map = self.index_data['weight_map']
        self.safetensors_metadata = self.index_data['safetensors_metadata']

    def load_layer_weights(self, model, layer_index):
        layer_tensors = {}
        for param_name, file_name in self.weight_map.items():
            if param_name.startswith(f"model.layers.{layer_index}"):
                file_path = f"{self.safetensor_directory}/{file_name}"
                print(f"loading safetensor\n{file_path}\nfor layer\n{layer_index}")
                offsets = self.safetensors_metadata[file_name]['offsets'][param_name]
                dtype = self.safetensors_metadata[file_name]['dtype']
                shape = self.safetensors_metadata[file_name]['shape']

                with safe_open(file_path, framework="pt", device="cuda") as f:
                    tensor = f.get_tensor_slice(offsets[0], offsets[1])
                    tensor = tensor.view(shape)  # Reshape to the correct shape
                
                layer_tensors[param_name] = tensor

        # Assign these tensors to the model's layer
        for param_name, tensor in layer_tensors.items():
            param_pointer = model
            param_parts = param_name.split('.')
            for attr in param_parts[:-1]:
                if attr.isdigit():
                    attr = int(attr)
                param_pointer = getattr(param_pointer, attr)
            setattr(param_pointer, param_parts[-1], tensor)

    def forward(
            self,
            model,
            input_ids: torch.tensor=None,
            hidden_states: torch.tensor=None,
            attention_mask: torch.tensor=None,
            **kwargs
        ) -> Tuple[Optional[torch.tensor], Optional[torch.tensor]]:

        base_model = model.model

        if input_ids is not None and hidden_states is not None:
            print("You must either pass a hidden_state or input_ids but not both")
            raise ValueError

        if input_ids is not None:
            hidden_states = base_model.embed_tokens(input_ids)
            position_ids = torch.arange(
                0,
                input_ids.size(1),
                device=input_ids.device
            ).unsqueeze(0)

        if hidden_states is not None:
            position_ids = torch.arange(
                0,
                hidden_states.size(1),
                device=hidden_states.device
            ).unsqueeze(0)

        for idx, layer in enumerate(self.layers):
            print(f"Loading weights for layer {idx}")
            self.load_layer_weights(model, idx)  # Load weights for the current layer
            print(f"Processing hidden state from layer {idx}\n")
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids
            )[0]

        if self.is_last:
            norm_states = base_model.norm(hidden_states).to("cuda")
            logits = model.lm_head(norm_states).to("cuda")

            return (None, logits)
        
        return (hidden_states, None)

async def model_half_split_test(
    prompt: str, 
    model_id: str, 
    layers: int,
    safetensor_index_file: str,
    safetensor_directory: str):

    half_layers = int(layers / 2)

    print("loading tokenizer")
    tokenizer = await resolve_tokenizer(model_id)
    max_length = 512

    print("loading config and model")
    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    model = AutoModel.from_config(config).to("cuda")

    print(model.hf_device_map)

    shard_layers = nn.ModuleList(model.model.layers[:half_layers])
    sharded_model = OnionHuggingFaceLM(
        layers=shard_layers,
        safetensor_index_file=safetensor_index_file,
        safetensor_directory=safetensor_directory
    )

    print(model)

    messages = [{"role": "user", "content": prompt}]
    txt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"Generating from chat template\n{txt}")

    inputs = tokenizer([txt], return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    input_attention_mask = inputs.attention_mask.to("cuda")

    shard_hidden_states, shard_logits = sharded_model.forward(
        model=model,
        input_ids=input_ids
    )

    print(f"shard_hidden_states\n{shard_hidden_states}")
    print(f"shard_logits\n{shard_logits}")

    print("Using first half hidden state for last half of model")
    shard_layers = nn.ModuleList(model.model.layers[half_layers:]).to("cuda")
    sharded_model.layers = shard_layers
    sharded_model.is_last = True 

    if shard_hidden_states is not None:
        shard_hidden_states, shard_logits = sharded_model.forward(
            model=model,
            hidden_states=shard_hidden_states
        )

        print(f"shard_hidden_states\n{shard_hidden_states}")
        print(f"shard_logits\n{shard_logits}")
    else:
        print("Sharded hidden states not found, error")
        raise ValueError

    print("generate from logits")
    if shard_logits is not None:
        generated_ids = sample_logits(shard_logits, TEMP, 0.95, TOP_K)
        print("generated_ids")
        print(generated_ids)

        generated_text = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        print("Generated text:")
        print(generated_text)
    else:
        print("Sharded logits missing from last layer run, error")
        raise ValueError

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
   prompt = "In a single word only, what is the last name of the current president of the USA?"

   print("\n-------- Test Qwen/Qwen2-7B-Instruct ----------\n")
   model_id = "Qwen/Qwen2-7B-Instruct"
   model_layers = 22

   asyncio.run(
       model_half_split_test(
           prompt=prompt,
           model_id=model_id,
           layers=model_layers,
           safetensor_index_file="./data/qwen2_7B_Instruct/model.safetensors.index.json",
           safetensor_directory="./data/qwen2_7B_Instruct/"
       )
   )

