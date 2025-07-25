# `macmon` – Mac Monitor

<div align="center">

Sudoless performance monitoring CLI tool for Apple Silicon processors.

[<img src="https://badges.ws/github/assets-dl/vladkens/macmon" />](https://github.com/vladkens/macmon/releases)
[<img src="https://badges.ws/github/release/vladkens/macmon" />](https://github.com/vladkens/macmon/releases)
[<img src="https://badges.ws/github/license/vladkens/macmon" />](https://github.com/vladkens/macmon/blob/main/LICENSE)
[<img src="https://badges.ws/badge/-/buy%20me%20a%20coffee/ff813f?icon=buymeacoffee&label" alt="donate" />](https://buymeacoffee.com/vladkens)

</div>

<div align="center">
  <img src="https://github.com/vladkens/macmon/blob/assets/macmon.png?raw=true" alt="preview" />
</div>

## Motivation

Apple Silicon processors don't provide an easy way to see live power consumption. I was interested in this information while testing local LLM models. `asitop` is a nice and simple TUI to quickly see current metrics, but it reads data from `powermetrics` and requires root privileges. `macmon` uses a private macOS API to gather metrics (essentially the same as `powermetrics`) but runs without sudo. 🎉

## 🌟 Features

- 🚫 Works without sudo
- ⚡ Real-time CPU / GPU / ANE power usage
- 📊 CPU utilization per cluster
- 💾 RAM / Swap usage
- 📈 Historical charts + avg / max values
- 🌡️ Average CPU / GPU temperature
- 🎨 Switchable colors (6 variants)
- 🪟 Can be rendered in a small window
- 🦀 Written in Rust

## 🍺 Install via Homebrew

You can install [`macmon`](https://formulae.brew.sh/formula/macmon) using [brew](https://brew.sh/):

```sh
$ brew install macmon
```

## 🖥️ Install via MacPorts

You can also install [`macmon`](https://ports.macports.org/port/macmon/) using [MacPorts](https://macports.org/):

```sh
$ sudo port install macmon
```

## 📦 Install from source

1. Install [Rust toolchain](https://www.rust-lang.org/tools/install)

2. Clone the repo:

```sh
git clone https://github.com/vladkens/macmon.git && cd macmon
```

3. Build and run:

```sh
cargo run -r
```

4. (Optionally) Binary can be moved to bin folder:

```sh
sudo cp target/release/macmon /usr/local/bin
```

## 🚀 Usage

```sh
Usage: macmon [OPTIONS] [COMMAND]

Commands:
  pipe   Output metrics in JSON format
  debug  Print debug information
  help   Print this message or the help of the given subcommand(s)

Options:
  -i, --interval <INTERVAL>  Update interval in milliseconds [default: 1000]
  -h, --help                 Print help
  -V, --version              Print version

Controls:
  c - change color
  v - switch charts view: gauge / sparkline
  q - quit
```

## 🚰 Piping

You can use the pipe subcommand to output metrics in JSON format, which is suitable for piping into other tools or scripts. For example:

```sh
macmon pipe | jq
```

This command runs `macmon` in "pipe" mode and navigate output to `jq` for pretty-printing.

You can also specify the number of samples to run using `-s` or `--samples` parameter (default: `0`, which runs indefinitely), and set update interval in milliseconds using the `-i` or `--interval` parameter (default: `1000` ms). For example:

```sh
macmon pipe -s 10 -i 500 | jq
```

This will collect 10 samples with an update interval of 500 milliseconds.

### Output

```jsonc
{
  "timestamp": "2025-02-24T20:38:15.427569+00:00",
  "temp": {
    "cpu_temp_avg": 43.73614,         // Celsius
    "gpu_temp_avg": 36.95167          // Celsius
  },
  "memory": {
    "ram_total": 25769803776,         // Bytes
    "ram_usage": 20985479168,         // Bytes
    "swap_total": 4294967296,         // Bytes
    "swap_usage": 2602434560          // Bytes
  },
  "ecpu_usage": [1181, 0.082656614],  // (Frequency MHz, Usage %)
  "pcpu_usage": [1974, 0.015181795],  // (Frequency MHz, Usage %)
  "gpu_usage": [461, 0.021497859],    // (Frequency MHz, Usage %)
  "cpu_power": 0.20486385,            // Watts
  "gpu_power": 0.017451683,           // Watts
  "ane_power": 0.0,                   // Watts
  "all_power": 0.22231553,            // Watts
  "sys_power": 5.876533,              // Watts
  "ram_power": 0.11635789,            // Watts
  "gpu_ram_power": 0.0009615385       // Watts (not sure what it means)
}
```

## 🤝 Contributing
We love contributions! Whether you have ideas, suggestions, or bug reports, feel free to open an issue or submit a pull request. Your input is essential in helping us improve `macmon` 💪

## 📝 License
`macmon` is distributed under the MIT License. For more details, check out the LICENSE.

## 🔍 See also
- [tlkh/asitop](https://github.com/tlkh/asitop) – Original tool. Python, requires sudo.
- [dehydratedpotato/socpowerbud](https://github.com/dehydratedpotato/socpowerbud) – ObjectiveC, sudoless, no TUI.
- [op06072/NeoAsitop](https://github.com/op06072/NeoAsitop) – Swift, sudoless.
- [graelo/pumas](https://github.com/graelo/pumas) – Rust, requires sudo.
- [context-labs/mactop](https://github.com/context-labs/mactop) – Go, requires sudo.

---

*PS: One More Thing... Remember, monitoring your Mac's performance with `macmon` is like having a personal trainer for your processor — keeping those cores in shape! 💪*
