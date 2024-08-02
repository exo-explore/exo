<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/exo-logo-black-bg.jpg">
  <img alt="exo logo" src="/docs/exo-logo-transparent.png" width="50%" height="50%">
</picture>

exo: Run your own AI cluster at home with everyday devices. Maintained by [exo labs](https://x.com/exolabs_).


<h3>

[Discord](https://discord.gg/EUnjGpsmWw) | [Telegram](https://t.me/+Kh-KqHTzFYg3MGNk) | [X](https://x.com/exolabs_)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/exo-explore/exo)](https://github.com/exo-explore/exo/stargazers)
[![Tests](https://circleci.com/gh/exo-explore/exo.svg?style=svg)](https://circleci.com/gh/exo-explore/exo)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

</div>

---

Forget expensive NVIDIA GPUs, unify your existing devices into one powerful GPU: iPhone, iPad, Android, Mac, Linux, pretty much any device!

<div align="center">
  <h2>Update: Exo Supports Llama 3.1</h2>
  <p>Now the default models, run 8B, 70B and 405B parameter models on your own devices</p>
  <p><a href="https://github.com/exo-explore/exo/blob/main/exo/inference/mlx/models/llama.py">See the code</a></p>
</div>

## Get Involved

exo is **experimental** software. Expect bugs early on. Create issues so they can be fixed. The [exo labs](https://x.com/exolabs_) team will strive to resolve issues quickly.

We also welcome contributions from the community. We have a list of bounties in [this sheet](https://docs.google.com/spreadsheets/d/1cTCpTIp48UnnIvHeLEUNg1iMy_Q6lRybgECSFCoVJpE/edit?usp=sharing).

## Features

### Wide Model Support

exo supports LLaMA ([MLX](exo/inference/mlx/models/llama.py) and [tinygrad](exo/inference/tinygrad/models/llama.py)) and other popular models.

### Dynamic Model Partitioning

exo [optimally splits up models](exo/topology/ring_memory_weighted_partitioning_strategy.py) based on the current network topology and device resources available. This enables you to run larger models than you would be able to on any single device.

### Automatic Device Discovery

exo will [automatically discover](https://github.com/exo-explore/exo/blob/945f90f676182a751d2ad7bcf20987ab7fe0181e/exo/orchestration/standard_node.py#L154) other devices using the best method available. Zero manual configuration.

### ChatGPT-compatible API

exo provides a [ChatGPT-compatible API](exo/api/chatgpt_api.py) for running models. It's a [one-line change](examples/chatgpt_api.py) in your application to run models on your own hardware using exo.

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

### From source


```sh
git clone https://github.com/exo-explore/exo.git
cd exo
pip install .
# alternatively, with venv
source install.sh
```

### Troubleshooting

- If running on Mac, MLX has an [install guide](https://ml-explore.github.io/mlx/build/html/install.html) with troubleshooting steps

## Documentation

### Example Usage on Multiple MacOS Devices

#### Device 1:

```sh
python3 main.py
```

#### Device 2:
```sh
python3 main.py
```

That's it! No configuration required - exo will automatically discover the other device(s).

The native way to access models running on exo is using the exo library with peer handles. See how in [this example for Llama 3](examples/llama3_distributed.py).

exo starts a ChatGPT-like WebUI (powered by [tinygrad tinychat](https://github.com/tinygrad/tinygrad/tree/master/examples/tinychat)) on http://localhost:8000

For developers, exo also starts a ChatGPT-compatible API endpoint on http://localhost:8000/v1/chat/completions. Example with curls:

```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "llama-3.1-8b",
     "messages": [{"role": "user", "content": "What is the meaning of exo?"}],
     "temperature": 0.7
   }'
```

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

## Debugging

Enable debug logs with the DEBUG environment variable (0-9).

```sh
DEBUG=9 python3 main.py
```

## Known Issues

- 🚧 As the library is evolving so quickly, the iOS implementation has fallen behind Python. We have decided for now not to put out the buggy iOS version and receive a bunch of GitHub issues for outdated code. We are working on solving this properly and will make an announcement when it's ready. If you would like access to the iOS implementation now, please email alex@exolabs.net with your GitHub username explaining your use-case and you will be granted access on GitHub.

## Inference Engines

exo supports the following inference engines:

- ✅ [MLX](exo/inference/mlx/sharded_inference_engine.py)
- ✅ [tinygrad](exo/inference/tinygrad/inference.py)
- 🚧 [llama.cpp](TODO)

## Networking Modules

- ✅ [GRPC](exo/networking/grpc)
- 🚧 [Radio](TODO)
- 🚧 [Bluetooth](TODO)

