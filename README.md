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

EXO lets you run your own AI cluster at home with everyday devices. We take advantage of Apple's M-series hardware and unified memory to run large language models, building a cluster to enable even more memory.

EXO underwent a full rewrite for v1. For legacy exo, see this repo's history or [exo-explore/ex-exo](https://github.com/exo-explore/ex-exo) for a snapshot.

---

## Features

- **Automatic discovery**: Devices running EXO automatically find each other on your local network - no manual configuration.
- **RDMA over Thunderbolt**: Ultra-low latency communication between macOS devices using RDMA over Thunderbolt.
- **Super-linear scaling**: Get up to 3.2x performance running large models across 4 machines with Tensor parallelism and RDMA.
- **MLX Support**: Uses the [mlx-explore/mlx](https://github.com/ml-explore/mlx) library for compute, enabling efficient and flexible machine learning on Apple silicon.

---

## Quick Start

You need at least one Mac device running macOS Tahoe 26.2 (released December 12th 2025).

You can download the latest build here: [EXO-latest.dmg](https://assets.exolabs.net/EXO-latest.dmg). It will ask for permission to modify system settings and install a new Network profile. We hope to make this smoother in the future!

To run from source, clone the repo, build the dashboard with `cd dashboard && npm install && npm run build` and run `uv run exo`.

After starting with either of these methods go to `http://localhost:8000` in your browser, and you'll have EXO.

---

## Requirements

- Mac devices with Apple Silicon (M-series chips)
- macOS Tahoe 26.2 or later (released December 12th 2025)
  - Older macOS versions may work without RDMA, but only 26.2+ is officially supported
- For RDMA over Thunderbolt: a high quality Thunderbolt 5 cable

We intend to add support for other hardware platforms [like the DGX Spark](https://x.com/exolabs/status/1978525767739883736) in the future, but they are not currently supported. If you'd like support for a new hardware platform, please search for an existing feature request and add a thumbs up so we know what hardware is important to the community.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to EXO.
