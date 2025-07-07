
Exo Community
===
> [!NOTE]  
> This is an unoffically maintained fork of [Exo](https://github.com/exo-explore/exo).
> The aim of this project is mainly maintenance upgrades and stability. 

> [!WARNING]
> No claim of ownership or future claim of stewardship is being made; No warranty is provided for this fork; use at your own risk.

# Contributing

I am looking for maintainers. Please open an issue if you are interested. 

<hr>

## Features

### Wide Model Support

exo supports different models including LLaMA ([MLX](exo/inference/mlx/models/llama.py) and [tinygrad](exo/inference/tinygrad/models/llama.py)), Mistral, LlaVA, Qwen, and Deepseek.

### Dynamic Model Partitioning

exo [optimally splits up models](exo/topology/ring_memory_weighted_partitioning_strategy.py) based on the current network topology and device resources available. This enables you to run larger models than you would be able to on any single device.

### Automatic Device Discovery

exo will [automatically discover](https://github.com/exo-explore/exo/blob/945f90f676182a751d2ad7bcf20987ab7fe0181e/exo/orchestration/node.py#L154) other devices using the best method available. Zero manual configuration.

### ChatGPT-compatible API

exo provides a [ChatGPT-compatible API](exo/api/chatgpt_api.py) for running models. It's a [one-line change](examples/chatgpt_api.sh) in your application to run models on your own hardware using exo.

### Device Equality

Unlike other distributed inference frameworks, exo does not use a master-worker architecture. Instead, exo devices [connect p2p](https://github.com/exo-explore/exo/blob/945f90f676182a751d2ad7bcf20987ab7fe0181e/exo/orchestration/node.py#L161). As long as a device is connected somewhere in the network, it can be used to run models.

Exo supports different [partitioning strategies](exo/topology/partitioning_strategy.py) to split up a model across devices. The default partitioning strategy is [ring memory weighted partitioning](exo/topology/ring_memory_weighted_partitioning_strategy.py). This runs an inference in a ring where each device runs a number of model layers proportional to the memory of the device.

!["A screenshot of exo running 5 nodes](docs/exo-screenshot.jpg)

## Installation

The current recommended way to install exo is from source.

### Prerequisites

- Python>=3.12.0 is required because of [issues with asyncio](https://github.com/exo-explore/exo/issues/5) in previous versions.
- For Linux with NVIDIA GPU support (Linux-only, skip if not using Linux or NVIDIA):
  - NVIDIA driver - verify with `nvidia-smi`
  - CUDA toolkit - install from [NVIDIA CUDA guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#cuda-cross-platform-installation), verify with `nvcc --version`
  - cuDNN library - download from [NVIDIA cuDNN page](https://developer.nvidia.com/cudnn-downloads), verify installation by following [these steps](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html#verifying-the-install-on-linux:~:text=at%20a%20time.-,Verifying%20the%20Install%20on%20Linux,Test%20passed!,-Upgrading%20From%20Older)

### Hardware Requirements

- The only requirement to run exo is to have enough memory across all your devices to fit the entire model into memory. For example, if you are running llama 3.1 8B (fp16), you need 16GB of memory across all devices. Any of the following configurations would work since they each have more than 16GB of memory in total:
  - 2 x 8GB M3 MacBook Airs
  - 1 x 16GB NVIDIA RTX 4070 Ti Laptop
  - 2 x Raspberry Pi 400 with 4GB of RAM each (running on CPU) + 1 x 8GB Mac Mini
- exo is designed to run on devices with heterogeneous capabilities. For example, you can have some devices with powerful GPUs and others with integrated GPUs or even CPUs. Adding less capable devices will slow down individual inference latency but will increase the overall throughput of the cluster.

### From source


```sh
git clone https://github.com/ai-at-home-community/exo.git
cd exo
pip install -e .
# alternatively, with venv
source install.sh
```

### Troubleshooting

- If running on Mac, MLX has an [install guide](https://ml-explore.github.io/mlx/build/html/install.html) with troubleshooting steps.

### Performance

- There are a number of things users have empirically found to improve performance on Apple Silicon Macs:

1. Upgrade to the latest version of macOS Sequoia.
2. Run `./configure_mlx.sh`. This runs commands to optimize GPU memory allocation on Apple Silicon Macs.


## Documentation

### Example Usage on Multiple macOS Devices

#### Device 1:

```sh
exo
```

#### Device 2:
```sh
exo
```

That's it! No configuration required - exo will automatically discover the other device(s).

exo starts a ChatGPT-like WebUI (powered by [tinygrad tinychat](https://github.com/tinygrad/tinygrad/tree/master/examples/tinychat)) on http://localhost:52415

For developers, exo also starts a ChatGPT-compatible API endpoint on http://localhost:52415/v1/chat/completions. Examples with curl:

#### Llama 3.2 3B:

```sh
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "llama-3.2-3b",
     "messages": [{"role": "user", "content": "What is the meaning of exo?"}],
     "temperature": 0.7
   }'
```

#### Llama 3.1 405B:

```sh
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "llama-3.1-405b",
     "messages": [{"role": "user", "content": "What is the meaning of exo?"}],
     "temperature": 0.7
   }'
```

#### DeepSeek R1 (full 671B):

```sh
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "deepseek-r1",
     "messages": [{"role": "user", "content": "What is the meaning of exo?"}],
     "temperature": 0.7
   }'
```

#### Llava 1.5 7B (Vision Language Model):

```sh
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "llava-1.5-7b-hf",
     "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What are these?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://images.cocodataset.org/val2017/000000039769.jpg"
            }
          }
        ]
      }
    ],
     "temperature": 0.0
   }'
```

### Example Usage on Multiple Heterogenous Devices (macOS + Linux)

#### Device 1 (macOS):

```sh
exo
```

Note: We don't need to explicitly tell exo to use the **tinygrad** inference engine. **MLX** and **tinygrad** are interoperable!

#### Device 2 (Linux):
```sh
exo
```

Linux devices will automatically default to using the **tinygrad** inference engine.

You can read about tinygrad-specific env vars [here](https://docs.tinygrad.org/env_vars/). For example, you can configure tinygrad to use the cpu by specifying `CLANG=1`.

### Example Usage on a single device with "exo run" command

```sh
exo run llama-3.2-3b
```

With a custom prompt:

```sh
exo run llama-3.2-3b --prompt "What is the meaning of exo?"
```

### Model Storage

Models by default are stored in `~/.cache/exo/downloads`.

You can set a different model storage location by setting the `EXO_HOME` env var.

## Model Downloading

Models are downloaded from Hugging Face. If you are running exo in a country with strict internet censorship, you may need to download the models manually and put them in the `~/.cache/exo/downloads` directory.

To download models from a proxy endpoint, set the `HF_ENDPOINT` environment variable. For example, to run exo with the huggingface mirror endpoint:

```sh
HF_ENDPOINT=https://hf-mirror.com exo
```

## Debugging

Enable debug logs with the DEBUG environment variable (0-9).

```sh
DEBUG=9 exo
```

For the **tinygrad** inference engine specifically, there is a separate DEBUG flag `TINYGRAD_DEBUG` that can be used to enable debug logs (1-6).

```sh
TINYGRAD_DEBUG=2 exo
```

## Formatting

We use [yapf](https://github.com/google/yapf) to format the code. To format the code, first install the formatting requirements:

```sh
pip3 install -e '.[formatting]'
```

Then run the formatting script:

```sh
python3 format.py ./exo
```

## Known Issues

- On certain versions of Python on macOS, certificates may not installed correctly, potentially causing SSL errors (e.g., when accessing huggingface.co). To resolve this, run the `Install Certificates` command, typicall as follows:

```sh
/Applications/Python 3.x/Install Certificates.command
```

## Inference Engines

exo supports the following inference engines:

- ✅ [MLX](exo/inference/mlx/sharded_inference_engine.py)
- ✅ [tinygrad](exo/inference/tinygrad/inference.py)
- 🚧 [PyTorch](https://github.com/exo-explore/exo/pull/139)
- 🚧 [llama.cpp](https://github.com/exo-explore/exo/issues/167)

## Discovery Modules

- ✅ [UDP](exo/networking/udp)
- ✅ [Manual](exo/networking/manual)
- ✅ [Tailscale](exo/networking/tailscale)
- 🚧 Radio
- 🚧 Bluetooth

# Peer Networking Modules

- ✅ [GRPC](exo/networking/grpc)
- 🚧 NCCL
