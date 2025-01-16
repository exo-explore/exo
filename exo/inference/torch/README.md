# PyTorch inference engine

## Devs
- [Vincent Castro](https://x.com/t0kenl1mit)

## ENV Vars
```bash
# Use the original max position embeddings amount, if present in the rope_scaling - default is False
TORCH_USE_ORG_SEQ = True or False

# Use cache - default is True
TORCH_USE_CACHE = True or False
```

## Notes/Issues
### 10/10/2024
- To select a pytorch device via environment variables, set the variable TORCH_DEVICE
  - XLA is currently not installed and will need to be added to inference.py, looking into doing this on a TPU VM
  - With pytorch, CUDA and ROCm are the same so specifying CUDA also enables ROCm support. See this [post](https://github.com/pytorch/pytorch/issues/55223#issuecomment-812587373)
  - Looking into adding mobile device support properly
- If device is not CPU the data type defaults to float32 else float16.

### 10/13/2024
Still working on split model development (see test_split_model.py). Right now, it seems to do it but still transformers is loading more in the RAM and GPU as it loads up a larger models (causing an OOM). Will research and add to next update. Right now, tests are added and are in development.

### 10/21/2024
Working on removing transformers due to inference and VRAM usage [issues](https://github.com/exo-explore/exo/pull/139#issuecomment-2424953962). Creating a pure pytorch implementation of llama3 as using transformers wont work for exo. Using some code from meta but also implementing the use of torchtune.

### 10/27/2024
Still working on llama3 model but wanted to note that a better KVCache needs to be investigated.

#### 11/17/2024
Llama sharded model now working and next step is inference engine. Still testing on small llama 3.2 1B but will try larger models.

### 01/16/2024
Torchtune has replaced huggingface transformers except for the tokenizer. Inferencing on Meta 3.2 1B seems okay but my GPU runs into a wall quickly. Will be trying on a bigger VM and LAN server to split up model.

## Tech
```bash
# Laptop/PC
Distributor ID:	Ubuntu
Description:	Ubuntu 24.04.1 LTS
Release:	24.04
Codename:	noble
CUDA Version: 12.4 
Nvidia Driver Version: 550.107.02

CPU: 11th Gen Intel® Core™ i7-11800H × 16
RAM: 16GB
GPU 1: Nvidia GeForce RTX 3060 6GB Laptop
```
```bash
# Server
Distributor ID: Pop
Description:    Pop!_OS 22.04 LTS
Release:        22.04
Codename:       jammy
CUDA Version:   12.4
Nvidia Driver Version: 550.90.07

GPU 1: NVIDIA T1000 8GB
GPU 2: NVIDIA Quadro M2000 4GB
GPU 3: NVIDIA Quadro M2000 4GB
GPU 4: NVIDIA Quadro P400 2GB
GPU 5: NVIDIA Quadro P400 2GB 
```

## Current Model

WIP pytorch llama model

```
# Llama-3.2-1B-Instruct #

ShardedLlamaModel(
  (model): ShardTransformerDecoder(
    (tok_embeddings): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-15): 16 x TransformerSelfAttentionLayer(
        (attn): MultiHeadAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (output_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (pos_embeddings): Llama3ScaledRoPE()
        )
        (mlp): FeedForward(
          (w1): Linear(in_features=2048, out_features=8192, bias=False)
          (w2): Linear(in_features=8192, out_features=2048, bias=False)
          (w3): Linear(in_features=2048, out_features=8192, bias=False)
          (activation): SiLU()
        )
        (sa_norm): RMSNorm()
        (mlp_norm): RMSNorm()
        (sa_scale): Identity()
        (mlp_scale): Identity()
      )
    )
    (norm): RMSNorm()
  )
)

```
