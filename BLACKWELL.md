# Blackwell GPU (GB10) + Mac Studio — Distributed Inference with exo

This documents the setup and fixes for running exo with prefill/decode disaggregation across an NVIDIA GB10 (Blackwell) and a Mac Studio.

## Architecture

```
┌─────────────────────────┐     10 GbE Cat 6a      ┌─────────────────────────┐
│     ASUS GX10           │◄───────────────────────►│     Mac Studio          │
│                         │   MTU 9000 (jumbo)      │                         │
│  NVIDIA GB10 (122 GB)   │   16 MB socket buffer   │  Apple Silicon          │
│  vLLM + FLASHINFER      │                         │  MLX (mxfp8 decode)     │
│  Qwen3.5-27B bf16       │                         │  Qwen3.5-27B mxfp8     │
│                         │                         │                         │
│  Role: PREFILL          │    KV cache transfer    │  Role: DECODE           │
│  1,400-2,170+ tok/s     │───────────────────────► │  18.7 tok/s             │
└─────────────────────────┘                         └─────────────────────────┘
```

## Performance (measured 2026-04-05)

### Prefill (GX10 vLLM + FLASHINFER, 10 GbE, MTU 9000, 16 MB send buffer)

| Prompt | Tokens | Cached | New | Time | Speed |
|--------|--------|--------|-----|------|-------|
| 1st (cold) | 2,467 | 0 | 2,467 | 7.0s | 354 tok/s |
| 2nd follow-up | 5,171 | 2,465 | 2,704 | 3.65s | 1,415 tok/s |
| 3rd follow-up | 7,846 | 5,169 | 2,675 | 3.61s | 2,171 tok/s |

### Timing breakdown (warm prompt, ~2,700 new tokens)

| Phase | Time | Notes |
|-------|------|-------|
| GPU compute | ~0.3s | Fast when prefix is cached |
| Network transfer | ~2.9s | 32 KV chunks over 10 GbE |
| Client inject | ~0.5s | MLX cache injection on Mac |
| **Total** | **~3.6s** | |

### Decode (Mac Studio MLX mxfp8)
- **18.7 tok/s** generation speed

### Time to First Token (TTFT)
- 1st prompt (cold): ~8,000 ms
- Follow-up prompts: ~2,300-3,700 ms (depends on new token count)

### Key behaviors
- Remote prefill only triggers for prompts **>1,000 uncached tokens**. Short prompts are handled locally by the Mac.
- Prefix caching means follow-up prompts in a conversation get progressively faster.
- Prefill and decode models are matched by `base_model` field, not `model_id` — so bf16 prefill + mxfp8 decode works automatically.

## Setup

### 1. Network (10 GbE direct link)

**Critical**: Without this, traffic routes over WiFi and KV transfer is 10x slower.

```bash
# GX10 (Linux):
sudo ip addr add 10.0.0.1/24 dev enP7s7
sudo ip link set enP7s7 mtu 9000

# Mac Studio (find ethernet interface with `ifconfig | grep -B2 "10.0.0"`):
sudo ifconfig en<X> 10.0.0.2 netmask 255.255.255.0 up
sudo ifconfig en<X> mtu 9000
```

Verify: `ping -c 2 10.0.0.2` from GX10 should show ~0.5ms latency.

### 2. Clone and run

```bash
git clone https://github.com/humanrouter/exo-thanh.git
cd exo-thanh
git checkout pr-1776
uv run exo
```

Run on both machines. They discover each other via libp2p.

### 3. Launch models

1. **GX10 dashboard**: Select `Qwen/Qwen3.5-27B` → **Vllm / Pipeline**
2. **Mac dashboard**: Select Qwen3.5-27B mxfp8 variant → **MlxRing / Pipeline / 1 node**
3. Send prompts to the Mac's model — prefill auto-routes to GX10

## Blackwell-Specific Fixes

These patches are required on top of exo PR #1776 to support Blackwell GPUs:

### 1. Force FLASHINFER on Blackwell (`vllm_generator.py`)
FlashAttention kernels (FLASH_ATTN, TRITON_ATTN) only support sm80-sm90. Blackwell is sm_121. Additionally, the `has_mamba` flag in PR 1776 excludes FLASHINFER for Qwen3.5 (which has hybrid linear/full attention layers). Fix: force FLASHINFER when GPU compute capability `major >= 10`.

### 2. Bump growable KV cache to 80% (`growable_cache.py`)
The growable cache patch overrides `gpu_memory_utilization` from EngineArgs with its own `INITIAL_FRACTION`. At the default 5%, only ~6 GB of the GB10's 122 GB VRAM was used for KV cache, severely limiting prefill throughput. Changed to 80% (~97 GB).

### 3. Disable vision encoder profiling (`vllm_generator.py`)
Qwen3.5-27B is a multimodal architecture (text + vision). The ViT encoder uses flash_attn internally which crashes on Blackwell during vLLM's memory profiling step. Since we only do text generation, `limit_mm_per_prompt={"image": 0, "video": 0}` skips the vision encoder entirely.

### 4. FlexAttention .view() → .reshape() (`growable_cache.py`)
The growable KV cache produces non-contiguous tensors after resizing. FlexAttention's forward pass calls `.view()` which requires contiguous memory, causing a RuntimeError. Patched to fall back to `.reshape()`.

### 5. Increase socket send buffer (`prefill_server.py`)
Increased from 4 MB to 16 MB. Reduces syscall overhead and allows the kernel to batch more KV cache data per TCP segment over 10 GbE with jumbo frames.

### 6. Model card for Qwen/Qwen3.5-27B
Added `resources/inference_model_cards/Qwen--Qwen3.5-27B.toml` for the native HuggingFace bf16 format used by vLLM.

## Hardware Notes

- The GB10 has 48 SMs — it's a desktop/mobile Blackwell chip, not datacenter. Prefill throughput scales with SM count.
- The GB10 has 122 GB unified memory shared with the Grace CPU. vLLM sees it all as VRAM.
- Cat 6a cable supports 10 GbE. Jumbo frames (MTU 9000) and larger socket buffers help with bulk KV cache transfer.
