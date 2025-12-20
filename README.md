<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/exo-logo-black-bg.jpg">
  <img alt="exo logo" src="/docs/exo-logo-transparent.png" width="50%" height="50%">
</picture>

exo: Run your own AI cluster at home with everyday devices. Maintained by [exo labs](https://x.com/exolabs).


[![GitHub Repo stars](https://img.shields.io/github/stars/exo-explore/exo)](https://github.com/exo-explore/exo/stargazers)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)

<a href="https://trendshift.io/repositories/11849" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11849" alt="exo-explore%2Fexo | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

---

EXO connects all your devices into an AI cluster. It pools together the resources of all your devices in order to run large models. Not only does EXO enable running models larger than would fit on a single device, but with [day-0 support for RDMA over Thunderbolt](https://x.com/exolabs/status/2001817749744476256?s=20), makes models run faster as you add more devices.

## Features

- **Automatic Device Discovery**: Devices running EXO automatically discover each other - no manual configuration.
- **RDMA over Thunderbolt**: EXO ships with [day-0 support for RDMA over Thunderbolt 5](https://x.com/exolabs/status/2001817749744476256?s=20), enabling 99% reduction in latency between devices.
- **Topology-Aware Auto Parallel**: EXO figures out the best way to split your model across all available devices based on a realtime view of your device topology. It takes into account device resources and network latency/bandwidth between each link.
- **Tensor Parallelism**: EXO supports sharding models, for up to 1.8x speedup on 2 devices and 3.2x speedup on 4 devices.
- **MLX Support**: EXO uses [MLX](https://github.com/ml-explore/mlx) as an inference backend and [MLX distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) for distributed communication.

## Benchmarks

<details>
  <summary>Qwen3-235B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-1-qwen3-235b.jpeg" alt="Benchmark - Qwen3-235B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

<details>
  <summary>DeepSeek v3.1 671B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-2-deepseek-3.1-671b.jpeg" alt="Benchmark - DeepSeek v3.1 671B (8-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

<details>
  <summary>Kimi K2 Thinking (native 4-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA</summary>
  <img src="docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-3-kimi-k2-thinking.jpeg" alt="Benchmark - Kimi K2 Thinking (native 4-bit) on 4 × M3 Ultra Mac Studio with Tensor Parallel RDMA" width="80%" />
  <p>
    <strong>Source:</strong> <a href="https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5">Jeff Geerling: 15 TB VRAM on Mac Studio – RDMA over Thunderbolt 5</a>
  </p>
</details>

---

## Quick Start

Devices running EXO automatically discover each other, without needing any manual configuration. Each device provides an API and a dashboard for interacting with your cluster (runs at `http://localhost:52415`).

There are two ways to run EXO:

### Run from Source (Mac & Linux)

**Prerequisites:**
- [uv](https://github.com/astral-sh/uv) (for Python dependency management)
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Or with brew:
  ```bash
  brew install uv
  ```
- [macmon](https://github.com/vladkens/macmon) (for hardware monitoring on Apple Silicon)
  ```bash
  brew install macmon
  ```

Clone the repo, build the dashboard, and run EXO:

```bash
cd dashboard && npm install && npm run build && cd..
uv run exo
```

**One-liner:**

```bash
git clone https://github.com/exo-explore/exo && cd exo/dashboard && npm i && npm run build && cd .. && uv run exo
```

---

### macOS App

EXO ships a macOS app that runs in the background on your Mac.

<img src="docs/macos-app-one-macbook.png" alt="EXO macOS App - running on a MacBook" width="35%" />

The macOS app requires macOS Tahoe 26.2 or later.

Download the latest build here: [EXO-latest.dmg](https://assets.exolabs.net/EXO-latest.dmg).

The app will ask for permission to modify system settings and install a new Network profile. Improvements to this are being worked on.

---

## Hardware Accelerator Support

On macOS, EXO uses the GPU. On Linux, EXO currently runs on CPU. We are working on extending hardware accelerator support. If you'd like support for a new hardware platform, please search for an existing feature request and add a thumbs up so we know what hardware is important to the community.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to EXO.
