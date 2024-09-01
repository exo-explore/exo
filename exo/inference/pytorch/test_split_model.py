import torch
import torch.nn as nn
import asyncio
from transformers import AutoModelForCausalLM, AutoConfig
from exo.api.chatgpt_api import resolve_tokenizer

async def model_split_test(prompt: str, model_id: str, layers: int):
    # inference
    tokenizer = await resolve_tokenizer(model_id)
    max_length = tokenizer.model_max_length

    # get full model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    half_layers = int(layers/2)

    # Create a copy of all the layers
    model_layers = model.model.layers
    copy_layers = []
    for i in range(half_layers):
        print(f"Copying layer {i}")
        layer_to_copy = model_layers[i]
        print(layer_to_copy)

        copy_layers.append(layer_to_copy)

    # load half layers back into model
    module_copy_list = nn.ModuleList(copy_layers).to("cuda")
    model.model.layers.load_state_dict(
        module_copy_list.state_dict(),
        strict=False
    )

    # generate first half
    inputs = tokenizer(prompt, return_tensors="pt")
    fhalf_generate_ids = model.generate(
        inputs.input_ids.to("cuda"),
        max_new_tokens=max_length/2
    ).to("cuda")

    print("fhalf_generate_ids")
    print(fhalf_generate_ids)

    # generate other half
    copy_layers = []
    for i in range(half_layers, layers):
        print(f"Copying layer {i}")
        layer_to_copy = model_layers[i]
        print(layer_to_copy)

        copy_layers.append(layer_to_copy)

    # load half layers back into model
    module_copy_list = nn.ModuleList(copy_layers).to("cuda")
    model.model.layers.load_state_dict(
        module_copy_list.state_dict(),
        strict=False
    )

    # generate second half with first half
    shalf_generate_ids = model.generate(
        fhalf_generate_ids
    ).to("cuda")
    
    print("generate_ids")
    print(shalf_generate_ids)
    print(tokenizer.eos_token_id)

    # decode second half
    decode = tokenizer.batch_decode(
        shalf_generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print("decode")
    print(decode)

if __name__ == "__main__":
    prompt = "In a single word only, what is the last name of the current president of the USA?"

    print("\n-------- Test TinyLlama/TinyLlama_v1.1 ----------\n")
    model_id = "TinyLlama/TinyLlama_v1.1"
    model_layers = 22

    asyncio.run(
        model_split_test(
            prompt=prompt,
            model_id=model_id,
            layers=model_layers
        )
    )

    print("\n-------- Test meta-llama/Meta-Llama-3.1-8B ----------\n")
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    model_layers = 32

    asyncio.run(
        model_split_test(
            prompt=prompt,
            model_id=model_id,
            layers=model_layers
        )
    )
