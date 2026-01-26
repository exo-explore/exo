# Windows 11 notes

This repo can run the control plane on Windows 11 (API, master, worker, and
networking). For GPU-backed inference on Windows, the easiest path today is the
**llama.cpp (GGUF) backend** using a prebuilt CUDA-enabled `llama-server.exe`.

## GPU discovery

Windows nodes report NVIDIA GPU memory when either:
- `pynvml` is installed, or
- `nvidia-smi` is available in PATH.

Placement uses free GPU memory when present; otherwise it falls back to RAM.

## llama.cpp (GGUF) backend

### 1) Install llama.cpp binaries (Windows x64 CUDA 12.4)

Download the Windows x64 CUDA 12.4 build from:

- https://github.com/ggml-org/llama.cpp/releases/tag/b7836

Extract `llama-server.exe` **and all the adjacent `.dll` files** somewhere on
each node.

If you want to keep things simple, you can place them in the repo root (e.g.
`C:\Users\peter\exo\`). These binaries are intentionally **not committed** to
git (the repo’s `.gitignore` ignores `/*.dll` and `/*.exe`).

### 2) Point exo at `llama-server.exe`

Set the path to `llama-server.exe` on each node:

```
EXO_LLAMA_SERVER_PATH=C:\path\to\llama-server.exe
```

Example (if you put the binaries in the repo root):

```
EXO_LLAMA_SERVER_PATH=C:\Users\peter\exo\llama-server.exe
```

Optional overrides for command templates:

```
EXO_LLAMA_SERVER_ARGS_TEMPLATE=--model {model_path} --host {host} --port {http_port} --rpc {rpc_list}
EXO_LLAMA_RPC_ARGS_TEMPLATE=--rpc --host {host} --port {rpc_port} --model {model_path}
```

Default ports:
- RPC: `EXO_LLAMA_RPC_PORT` (default `50052`)
- HTTP: `EXO_LLAMA_SERVER_PORT` (default `8081`)

Place GGUF files in `EXO_MODELS_DIR` and reference them by filename in requests.

Use the `LlamaRpc` instance meta when placing instances:

```
{"model_id": "my-model.gguf", "sharding": "Pipeline", "instance_meta": "LlamaRpc", "min_nodes": 2}
```

## Network sharding across PCs

For stable LAN discovery, consider fixing the libp2p listen port and opening
firewall rules:

```
EXO_LIBP2P_LISTEN_ADDR=0.0.0.0
EXO_LIBP2P_LISTEN_PORT=4001
```

The API listens on port `52415` by default. Ensure inbound TCP is allowed on:
- `52415` (API and peer checks)
- `EXO_LIBP2P_LISTEN_PORT` (libp2p gossip)

## Native CUDA backend TODO

Longer-term, a native CUDA-backed runner could be implemented (e.g. PyTorch +
distributed pipeline/tensor parallelism, or an alternate runtime that supports
multi-node sharding). For now, llama.cpp is the practical Windows GPU path.
