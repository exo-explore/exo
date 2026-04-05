# Blackwell GPU (GB10) + Mac Studio — Distributed Inference with exo

This documents the setup and fixes for running exo with prefill/decode disaggregation across an NVIDIA GB10 (Blackwell) and a Mac Studio.

## Architecture

```
┌─────────────────────────┐     10 GbE Cat 6a      ┌─────────────────────────┐
│     ASUS GX10           │◄───────────────────────►│     Mac Studio          │
│                         │    MTU 9000 (jumbo)     │                         │
│  NVIDIA GB10 (122 GB)   │                         │  Apple Silicon          │
│  vLLM + FLASHINFER      │                         │  MLX (mxfp8 decode)     │
│  Qwen3.5-27B bf16       │                         │  Qwen3.5-27B mxfp8     │
│                         │                         │                         │
│  Role: PREFILL          │    KV cache transfer    │  Role: DECODE           │
│  354-2133 tok/s         │───────────────────────► │  18.7 tok/s             │
└─────────────────────────┘                         └─────────────────────────┘
```

## Performance

| Metric | Value |
|--------|-------|
| Prefill (cold, 2.5k tokens) | 354 tok/s, ~8s TTFT |
| Prefill (warm, cached prefix) | 1,400-2,100+ tok/s |
| Decode (Mac, mxfp8) | 18.7 tok/s |
| TTFT (follow-up prompts) | 3.5-4.5s |

Prefix caching means follow-up prompts in a conversation get progressively faster.

## Setup

### 1. Network (10 GbE direct link)

**Critical**: Without this, traffic routes over WiFi and KV transfer is 10x slower.

```bash
# GX10 (Linux):
sudo ip addr add 10.0.0.1/24 dev enP7s7
sudo ip link set enP7s7 mtu 9000

# Mac Studio:
sudo ifconfig en<X> 10.0.0.2 netmask 255.255.255.0 up
sudo ifconfig en<X> mtu 9000
```

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

These are the patches applied on top of exo PR 1776:

### FLASHINFER forced on Blackwell (`vllm_generator.py`)
FlashAttention kernels (FLASH_ATTN, TRITON_ATTN) only support sm80-sm90. Blackwell is sm_121. The `has_mamba` flag in PR 1776 was also excluding FLASHINFER for Qwen3.5 (which has hybrid linear/full attention). Fix: force FLASHINFER when `major >= 10`.

### Growable KV cache initial fraction bumped to 80% (`growable_cache.py`)
The growable cache patch overrides `gpu_memory_utilization` with `INITIAL_FRACTION`. At the default 5%, only ~6 GB of 122 GB VRAM was used for KV cache. Bumped to 80% (~97 GB).

### Vision encoder disabled (`vllm_generator.py`)
Qwen3.5-27B is a multimodal architecture. The ViT encoder uses flash_attn internally which crashes on Blackwell during memory profiling. Since we only do text generation, `limit_mm_per_prompt={"image": 0, "video": 0}` skips it entirely.

### FlexAttention .view() → .reshape() (`growable_cache.py`)
The growable cache produces non-contiguous KV tensors after resizing. FlexAttention's forward pass calls `.view()` which requires contiguous memory. Patched to fall back to `.reshape()`.

### Model card for Qwen/Qwen3.5-27B
Added `resources/inference_model_cards/Qwen--Qwen3.5-27B.toml` for the native HuggingFace bf16 format.

## Notes

- Remote prefill only triggers for prompts **>1000 uncached tokens**. Short prompts are handled locally.
- Prefill and decode models are matched by `base_model` field, not `model_id` — so bf16 prefill + mxfp8 decode works.
- The GX10's GB10 has 48 SMs — it's a desktop/mobile Blackwell chip, not datacenter. Performance scales with SM count.
