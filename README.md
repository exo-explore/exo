<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/exo-logo-black-bg.jpg">
  <img alt="exo logo" src="/docs/exo-logo-transparent.png" width="50%" height="50%">
</picture>

exo: Run your own AI cluster at home with everyday devices. Maintained by [exo labs](https://x.com/exolabs).


<h3>

[Discord](https://discord.gg/EUnjGpsmWw) | [Telegram](https://t.me/+Kh-KqHTzFYg3MGNk) | [X](https://x.com/exolabs)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/exo-explore/exo)](https://github.com/exo-explore/exo/stargazers)
[![Tests](https://dl.circleci.com/status-badge/img/circleci/TrkofJDoGzdQAeL6yVHKsg/4i5hJuafuwZYZQxbRAWS71/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/TrkofJDoGzdQAeL6yVHKsg/4i5hJuafuwZYZQxbRAWS71/tree/main)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

</div>

---

Forget expensive NVIDIA GPUs, unify your existing devices into one powerful GPU: iPhone, iPad, Android, Mac, Linux, pretty much any device!

<div align="center">
  <h2>Update: exo is hiring. See <a href="https://exolabs.net">here</a> for more details.</h2>
</div>

## Get Involved

exo is **experimental** software. Expect bugs early on. Create issues so they can be fixed. The [exo labs](https://x.com/exolabs) team will strive to resolve issues quickly.

We also welcome contributions from the community. We have a list of bounties in [this sheet](https://docs.google.com/spreadsheets/d/1cTCpTIp48UnnIvHeLEUNg1iMy_Q6lRybgECSFCoVJpE/edit?usp=sharing).

## Features

### Wide Model Support

exo supports different models including LLaMA ([MLX](exo/inference/mlx/models/llama.py) and [tinygrad](exo/inference/tinygrad/models/llama.py)), Mistral, LlaVA, Qwen and Deepseek.

### Dynamic Model Partitioning

exo [optimally splits up models](exo/topology/ring_memory_weighted_partitioning_strategy.py) based on the current network topology and device resources available. This enables you to run larger models than you would be able to on any single device.

### Automatic Device Discovery

exo will [automatically discover](https://github.com/exo-explore/exo/blob/945f90f676182a751d2ad7bcf20987ab7fe0181e/exo/orchestration/standard_node.py#L154) other devices using the best method available. Zero manual configuration.

### ChatGPT-compatible API

exo provides a [ChatGPT-compatible API](exo/api/chatgpt_api.py) for running models. It's a [one-line change](examples/chatgpt_api.sh) in your application to run models on your own hardware using exo.

### Device Equality

Unlike other distributed inference frameworks, exo does not use a master-worker architecture. Instead, exo devices [connect p2p](https://github.com/exo-explore/exo/blob/945f90f676182a751d2ad7bcf20987ab7fe0181e/exo/orchestration/standard_node.py#L161). As long as a device is connected somewhere in the network, it can be used to run models.

Exo supports different [partitioning strategies](exo/topology/partitioning_strategy.py) to split up a model across devices. The default partitioning strategy is [ring memory weighted partitioning](exo/topology/ring_memory_weighted_partitioning_strategy.py). This runs an inference in a ring where each device runs a number of model layers proportional to the memory of the device.

<p>
    <picture>
        <img alt="ring topology" src="docs/ring-topology.png" width="30%" height="30%">
    </picture>
</p>


## Installation

The current recommended way to install exo is from source.

### Prerequisites

- Python>=3.12.0 is required because of [issues with asyncio](https://github.com/exo-explore/exo/issues/5) in previous versions.
- Linux (with NVIDIA card):
  - NVIDIA driver (test with `nvidia-smi`)
  - CUDA (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#cuda-cross-platform-installation) (test with `nvcc --version`)
  - cuDNN (https://developer.nvidia.com/cudnn-downloads) (test with [link](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html#verifying-the-install-on-linux:~:text=at%20a%20time.-,Verifying%20the%20Install%20on%20Linux,Test%20passed!,-Upgrading%20From%20Older))

### From source


```sh
git clone https://github.com/exo-explore/exo.git
cd exo
pip install .
# alternatively, with venv
source install.sh
```


### Troubleshooting

- If running on Mac, MLX has an [install guide](https://ml-explore.github.io/mlx/build/html/install.html) with troubleshooting steps.

### Performance

- There are a number of things users have empirically found to improve performance on Apple Silicon Macs:

1. Upgrade to the latest version of MacOS 15.
2. Run `./configure_mlx.sh`. This runs commands to optimize GPU memory allocation on Apple Silicon Macs.


## Documentation

### Example Usage on Multiple MacOS Devices

#### Device 1:

```sh
exo
```

#### Device 2:
```sh
exo
```

That's it! No configuration required - exo will automatically discover the other device(s).

exo starts a ChatGPT-like WebUI (powered by [tinygrad tinychat](https://github.com/tinygrad/tinygrad/tree/master/examples/tinychat)) on http://localhost:8000

For developers, exo also starts a ChatGPT-compatible API endpoint on http://localhost:8000/v1/chat/completions. Examples with curl:

#### Llama 3.1 8B:

```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "llama-3.1-8b",
     "messages": [{"role": "user", "content": "What is the meaning of exo?"}],
     "temperature": 0.7
   }'
```

#### Llama 3.1 405B:

```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "llama-3.1-405b",
     "messages": [{"role": "user", "content": "What is the meaning of exo?"}],
     "temperature": 0.7
   }'
```

#### Llava 1.5 7B (Vision Language Model):

```sh
curl http://localhost:8000/v1/chat/completions \
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

### Example Usage on Multiple Heterogenous Devices (MacOS + Linux)

#### Device 1 (MacOS):

```sh
exo --inference-engine tinygrad
```

Here we explicitly tell exo to use the **tinygrad** inference engine.

#### Device 2 (Linux):
```sh
exo
```

Linux devices will automatically default to using the **tinygrad** inference engine.

You can read about tinygrad-specific env vars [here](https://docs.tinygrad.org/env_vars/). For example, you can configure tinygrad to use the cpu by specifying `CLANG=1`.


## Debugging

Enable debug logs with the DEBUG environment variable (0-9).

```sh
DEBUG=9 exo
```

For the **tinygrad** inference engine specifically, there is a separate DEBUG flag `TINYGRAD_DEBUG` that can be used to enable debug logs (1-6).

```sh
TINYGRAD_DEBUG=2 exo
```

## Known Issues

- On some versions of MacOS/Python, certificates are not installed properly which can lead to SSL errors (e.g. SSL error with huggingface.co). To fix this, run the Install Certificates command, usually:

```sh
/Applications/Python 3.x/Install Certificates.command
```

- 🚧 As the library is evolving so quickly, the iOS implementation has fallen behind Python. We have decided for now not to put out the buggy iOS version and receive a bunch of GitHub issues for outdated code. We are working on solving this properly and will make an announcement when it's ready. If you would like access to the iOS implementation now, please email alex@exolabs.net with your GitHub username explaining your use-case and you will be granted access on GitHub.

## Inference Engines

exo supports the following inference engines:

- ✅ [MLX](exo/inference/mlx/sharded_inference_engine.py)
- ✅ [tinygrad](exo/inference/tinygrad/inference.py)
- 🚧 [PyTorch](https://github.com/exo-explore/exo/pull/139)
- 🚧 [llama.cpp](https://github.com/exo-explore/exo/issues/167)

## Networking Modules

- ✅ [GRPC](exo/networking/grpc)
- 🚧 [Radio](TODO)
- 🚧 [Bluetooth](TODO)
