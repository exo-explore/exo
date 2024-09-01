import torch
import torch.nn as nn
import asyncio
import gc
from transformers import AutoModelForCausalLM, AutoConfig, Qwen2ForCausalLM
from exo.api.chatgpt_api import resolve_tokenizer
import re

async def model_split_test(prompt: str, model_id: str, layers: int):
    # inference
    tokenizer = await resolve_tokenizer(model_id)
    max_length = 512 #tokenizer.model_max_length

    # get full model
    if re.match(r"^Qwen|qwen", model_id):
        model = Qwen2ForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto",
            # attn_implementation="eager"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto",
        )

    # add pad token if none
    # this is for llama based models, will add a check
    if tokenizer.pad_token == None and re.match(r"Llama|llama", model_id):
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    half_layers = int(layers/2)

    # Create a copy of all the layers
    model_layers = model.model.layers
    copy_layers = []
    for i in range(half_layers):
        # print(f"Copying layer {i}")
        layer_to_copy = model_layers[i]
        # print(layer_to_copy)

        copy_layers.append(layer_to_copy)

    print(f"loading {len(copy_layers)} layers back to model")

    # load half layers back into model
    module_copy_list = nn.ModuleList(copy_layers).to("cuda")
    model.model.layers.load_state_dict(
        module_copy_list.state_dict(),
        strict=False
    )

    # generate first half
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
    fhalf_generate_ids = model.generate(
        input_ids,
        # attention_mask=input_attention_mask,
        max_length=int(max_length/2),
        output_hidden_states=True
        # output_attentions=True
    ).to("cuda")

    print("fhalf_generate_ids")
    print(fhalf_generate_ids)

    # nptest = fhalf_generate_ids.numpy(force=True)
    # print(f"nptest: {nptest}")

    # generate other half
    copy_layers = []
    for i in range(half_layers, layers):
        # print(f"Copying layer {i}")
        layer_to_copy = model_layers[i]
        # print(layer_to_copy)

        copy_layers.append(layer_to_copy)

    print(f"loading {len(copy_layers)} layers back to model")

    # load half layers back into model
    module_copy_list = nn.ModuleList(copy_layers).to("cuda")
    model.model.layers.load_state_dict(
        module_copy_list.state_dict(),
        strict=False
    )

    # generate second half with first half
    print(f"Generating from hidden layers output fhalf_generate_ids")
    shalf_generate_ids = model.generate(
        fhalf_generate_ids,
        # attention_mask=input_attention_mask,
        max_length=max_length
    ).to("cuda")
    
    print("shalf_generate_ids")
    print(shalf_generate_ids)
    # print(tokenizer.eos_token_id)

    # decode second half
    decode = tokenizer.batch_decode(
        shalf_generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print("decode")
    print(decode)

    # free model from memory
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    prompt = "In a single word only, what is the last name of the current president of the USA?"

    # print("\n-------- Test TinyLlama/TinyLlama_v1.1 ----------\n")
    # model_id = "TinyLlama/TinyLlama_v1.1"
    # model_layers = 22

    # asyncio.run(
    #     model_split_test(
    #         prompt=prompt,
    #         model_id=model_id,
    #         layers=model_layers
    #     )
    # )

    # print("\n-------- Test meta-llama/Meta-Llama-3.1-8B ----------\n")
    # model_id = "meta-llama/Meta-Llama-3.1-8B"
    # model_layers = 32

    # asyncio.run(
    #     model_split_test(
    #         prompt=prompt,
    #         model_id=model_id,
    #         layers=model_layers
    #     )
    # )

    print("\n-------- Test Qwen/Qwen2-0.5B-Instruct ----------\n")
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    model_layers = 24

    asyncio.run(
        model_split_test(
            prompt=prompt,
            model_id=model_id,
            layers=model_layers
        )
    )

