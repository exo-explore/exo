# Qwen3.8 Flash Next on EXO

EXO supports the `qwen4_exp` architecture used by
`Qwen/Qwen3.8-Flash-Next` through the MLX-VLM language runtime. The built-in
model catalog recommends
`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`, a corrected group-size-64
conversion with 288 of the original 512 experts retained per MoE layer.

## Why a separate loader is required

Qwen3.8 Flash Next is not a Qwen3-Next or Qwen3.5 model. It combines Gated
DeltaNet layers, Qwen Sparse Attention (QSA), hyper-connection residual streams,
a sparse MoE, and a large n-gram PLE table. The architecture and its auxiliary
QSA cache live in MLX-VLM. EXO therefore uses MLX-VLM to sanitize and load
`qwen4_exp`, extracts its language model, and keeps EXO's existing vision and
generation orchestration around it. Other architectures continue to load via
MLX-LM.

EXO pins MLX-VLM to an immutable upstream commit containing QSA/APC disk-cache,
continuous-batching, FP8, MTP, and external-PLE fixes. EXO does not use
MLX-VLM's bundled web server, so the lockfile preserves EXO's tested
FastAPI/Starlette versions.

## Recommended checkpoint

| Property | Value |
| --- | --- |
| Model ID | `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit` |
| Download size | 73,504,903,388 bytes (about 68.5 GiB) |
| Quantization | 4-bit affine, group size 64 (PLE table group size 32) |
| Layers | 48 |
| Context | 262,144 tokens |
| Default mode | Thinking (`xhigh` when no effort is supplied by the client) |

The REAP checkpoint reports 91.5% HumanEval pass@1 versus 93.9% for its
unpruned conversion while reducing checkpoint and resident memory by about
31%. Its publisher also corrected two conversion defects: RMSNorm tensors are
stored using the zero-centered convention expected by MLX-VLM, and PLE tensors
and quantization overrides use the runtime's module names. An older
`Vontra/Qwen3.8-Flash-Next-MLX-4bit` conversion produced incoherent output in
both native MLX-VLM and EXO and is deliberately not cataloged.

The original BF16 repository occupies about 360 GB. It is not recommended for
EXO unless BF16 fidelity is specifically required. The REAP checkpoint retains
the vision tower, but its publisher has only evaluated text quality; EXO does
not advertise vision capability for this catalog entry yet.

## External model storage

Large checkpoints should live on a volume with sufficient free space. Add that
directory when starting EXO:

```bash
EXO_MODELS_DIRS="/Volumes/ExternalSSD/exo-models" uv run exo
```

EXO searches its default directory first and then additional writable model
directories. Existing complete models in any configured directory are reused.
The dashboard's model catalog can download the recommended checkpoint after the
directory is configured.

To validate a complete local checkpoint through EXO's own loader, chat template,
cache, and generation path:

```bash
uv run python scripts/smoke_qwen38_flash_next.py \
  "/Volumes/ExternalSSD/exo-models/sh0wie--Qwen3.8-Flash-Next-REAP-288-MLX-4bit"
```

## Parallelism and current boundary

- Single-node MLX execution is supported.
- Pipeline parallelism recognizes and repairs Qwen3.8's hybrid layer/cache
  indices after layer slicing.
- Tensor parallelism is intentionally not advertised. QSA, GDN, MoE, and PLE
  need a dedicated bit-exact tensor-sharding implementation before EXO may
  select that placement.
- The model's QSA cache carries auxiliary index keys and positions in addition
  to ordinary KV state. Prefix-cache rollback relies on that cache's specialized
  trim implementation.

## Validation

The focused suite covers:

- MLX-VLM routing for `qwen4_exp` while preserving MLX-LM routing for existing
  architectures;
- official model-card shape and vision metadata;
- Qwen3.8 EOS tokens and reasoning-effort normalization;
- a Metal-backed tiny Qwen4-Exp prefill followed by decode, including QSA
  auxiliary-cache growth;
- recurrent-state snapshot/restore and full-weight prefix reuse (18 of 20
  prompt tokens reused, with the repeated smoke request dropping from 2.8 to
  0.7 seconds on the validation host);
- existing EXO cache, pipeline, shared-state, and tool-parser regressions.

The recommended full checkpoint was also validated independently through
native MLX-VLM and through EXO. Both returned the requested coherent text.
Native MLX-VLM peaked at 41.7 GB active memory; EXO peaked at 44.5 GB while
using the external PLE store on an external SSD. A thinking-mode run produced
the correct result and a properly delimited reasoning section.
