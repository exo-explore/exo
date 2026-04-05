# How to Submit Blackwell Fixes to exo-explore/exo

The GX10 machine's GitHub token doesn't have permission to create PRs on exo-explore/exo. Use the Mac Studio instead.

## One-time setup (Mac Studio)

```bash
# 1. Make sure gh is installed and authenticated
brew install gh
gh auth login

# 2. Fork exo-explore/exo to your account (if not already done)
gh repo fork exo-explore/exo --clone=false

# 3. In the exo-thanh repo, add the fork as a remote
cd ~/path/to/exo-thanh
git remote add fork https://github.com/humanrouter/exo.git

# 4. Push the branch to the fork
git push fork pr-1776:blackwell-support
```

## Create the PR

```bash
gh pr create \
  --repo exo-explore/exo \
  --head humanrouter:blackwell-support \
  --base leo/prefill-decode-really \
  --title "Add NVIDIA Blackwell (GB10) support for prefill/decode disaggregation" \
  --body "$(cat <<'PREOF'
## Summary

Adds support for NVIDIA Blackwell GPUs (GB10, sm_121) running vLLM prefill in exo's prefill/decode disaggregation architecture (PR #1776). Tested with an ASUS GX10 (GB10, 122 GB VRAM) + Mac Studio over 10 GbE.

### Changes
- **Force FLASHINFER on Blackwell** — FlashAttention kernels (FLASH_ATTN, TRITON_ATTN) only support sm80-sm90. The \`has_mamba\` flag was also excluding FLASHINFER for hybrid attention models like Qwen3.5. Fix: force FLASHINFER when \`major >= 10\`
- **Bump growable KV cache initial fraction from 5% to 80%** — \`INITIAL_FRACTION\` overrides \`gpu_memory_utilization\` from EngineArgs. At 5%, only ~6 GB of 122 GB VRAM was used for KV cache, crippling prefill throughput
- **Disable vision encoder profiling** — Qwen3.5-27B's ViT encoder uses flash_attn which crashes on Blackwell during memory profiling. \`limit_mm_per_prompt={"image": 0, "video": 0}\` skips it for text-only inference
- **FlexAttention .view() → .reshape()** — Growable cache produces non-contiguous KV tensors
- **Increase socket send buffer to 16 MB** — Better throughput over 10 GbE with jumbo frames
- **Add Qwen/Qwen3.5-27B model card** and **BLACKWELL.md** setup documentation

### Performance (ASUS GX10 + Mac Studio, 10 GbE, MTU 9000)

| Metric | Value |
|--------|-------|
| Prefill (cold, 2.5k tokens) | 354 tok/s, ~8s TTFT |
| Prefill (warm, cached prefix) | 1,400 – 2,170+ tok/s |
| Decode (Mac Studio, mxfp8) | 18.7 tok/s |
| TTFT (follow-up prompts) | 2.3 – 3.7s |

### Timing breakdown (warm prefill, ~2,700 new tokens)
- GPU compute: ~0.3s
- Network transfer (10 GbE): ~2.9s
- Client inject: ~0.5s

## Test plan

- [x] vLLM engine loads with FLASHINFER on GB10 (sm_121)
- [x] Prefill server starts and accepts connections over 10 GbE
- [x] KV cache transfer works between GX10 (vLLM) and Mac Studio (MLX)
- [x] Prefix caching accelerates follow-up prompts (354 → 2,171 tok/s)
- [x] Decode runs at 18.7 tok/s on Mac Studio with mxfp8
- [x] Model matching by \`base_model\` works across bf16 prefill + mxfp8 decode
- [x] Jumbo frames (MTU 9000) + 16 MB send buffer improve transfer times

🤖 Generated with [Claude Code](https://claude.com/claude-code)
PREOF
)"
```

## Updating the PR later

After making more changes on the GX10:

```bash
# On GX10: push to exo-thanh
git push thanh pr-1776

# On Mac: pull and push to the fork
cd ~/path/to/exo-thanh
git pull
git push fork pr-1776:blackwell-support --force
```

The PR will auto-update.

## Files changed (relative to PR #1776)

```
src/exo/worker/engines/vllm/vllm_generator.py   — FLASHINFER on Blackwell, disable vision profiling
src/exo/worker/engines/vllm/growable_cache.py    — 80% KV cache, FlexAttention reshape patch
src/exo/disaggregated/prefill_server.py          — 16 MB send buffer
resources/inference_model_cards/Qwen--Qwen3.5-27B.toml  — Model card
BLACKWELL.md                                     — Setup documentation
```
