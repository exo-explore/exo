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

## Features

- **Automatic Device Discovery**: Devices running EXO automatically discover each other on your local network - no manual configuration.
- **RDMA over Thunderbolt**: EXO ships with Day-0 support for RDMA over Thunderbolt 5, enabling 99% reduction in latency between devices.
- **Auto Parallel**: EXO automatically splits up models to run distributed across devices.
- **Tensor Parallelism**: EXO supports sharding models, for up to 1.8x speedup on 2 devices and 3.2x speedup on 4 devices.
- **MLX Support**: EXO uses [ml-explore/mlx](https://github.com/ml-explore/mlx) as an inference backend and [MLX distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) for distributed communication.

---

## Quick Start

You need at least one Mac device running macOS Tahoe 26.2 (released December 12th 2025).

You can download the latest build here: [EXO-latest.dmg](https://assets.exolabs.net/EXO-latest.dmg). It will ask for permission to modify system settings and install a new Network profile. We hope to make this smoother in the future!

To run from source, clone the repo, build the dashboard with `cd dashboard && npm install && npm run build` and run `uv run exo`.

After starting with either of these methods go to `http://localhost:52415` in your browser, and you'll have EXO.

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
