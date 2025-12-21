# Android Model & Inference Issues

> **Tracking document for fixing model download and inference on Android/Termux**

## Current Status

| Issue | Status | Priority |
|-------|--------|----------|
| `llama_cpp` module not found | ✅ Fixed | Critical |
| MLX models not compatible with Android | ✅ Fixed | Critical |
| GGUF download fails - missing index.json | ✅ Fixed | Critical |
| Connection instability during inference | 🟡 Open | High |

---

## ✅ FIXED: GGUF Download & MLX Model Filtering

**Fixed in commit:** December 2024

### Changes Made

1. **`src/exo/worker/download/download_utils.py`**
   - Added `is_gguf_model()` function to detect GGUF format models
   - Added `resolve_allow_patterns_for_gguf()` to handle GGUF repos
   - Modified `resolve_allow_patterns()` to skip `model.safetensors.index.json` lookup for GGUF models
   - GGUF models now download the `.gguf` file directly (prefers Q4_K_M quantization)

2. **`src/exo/shared/models/model_cards.py`**
   - MLX `MODEL_CARDS` fully commented out (kept for future re-enablement)
   - Empty `MODEL_CARDS` dict for Android-only builds
   - `GGUF_MODEL_CARDS` is now the primary model source
   - `ALL_MODEL_CARDS` only includes GGUF models

3. **`src/exo/master/api.py`**
   - Updated `resolve_model_meta()` to check `GGUF_MODEL_CARDS` first
   - Dashboard now only shows GGUF-compatible models

### Result
- Dashboard model list only shows GGUF models (compatible with Android)
- Model downloads work correctly for GGUF format
- No more `FileNotFoundError: model.safetensors.index.json`

---

## ✅ FIXED: `llama_cpp` Module Not Found

### Solution

Install `llama-cpp-python` and ensure environment variables are set:

```bash
# Ensure environment is set
source ~/.bashrc

# Verify llama.cpp libraries exist
ls -la ~/llama.cpp/build/bin/*.so

# Install llama-cpp-python
pip install llama-cpp-python

# Verify
python -c "from llama_cpp import Llama; print('OK')"
```

Ensure `.bashrc` contains:
```bash
export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH
export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
```

---

## 🟡 OPEN: Connection Instability

### Symptoms
```
ConnectionClosed { peer_id: ... cause: None }
Manually removing node ... due to inactivity
Waiting for other campaign to finish
```

### Likely Causes
1. Runner crashes causing node state issues
2. Election re-triggering on every runner failure
3. Mobile network switching (WiFi handoff)

### Mitigation
- Ensure `termux-wake-lock` is active
- Use stable WiFi connection
- Keep both devices on same network subnet

---

## Available GGUF Models

These models are now available in the dashboard for Android:

| Model | Size | RAM Needed | Model ID |
|-------|------|------------|----------|
| TinyLlama 1.1B | ~700MB | 2GB | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` |
| Qwen 2.5 0.5B | ~400MB | 1.5GB | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` |
| Qwen 2.5 1.5B | ~1GB | 3GB | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` |
| Qwen 2.5 3B | ~2GB | 4GB | `Qwen/Qwen2.5-3B-Instruct-GGUF` |
| Llama 3.2 1B | ~750MB | 2GB | `bartowski/Llama-3.2-1B-Instruct-GGUF` |
| Llama 3.2 3B | ~2GB | 4GB | `bartowski/Llama-3.2-3B-Instruct-GGUF` |
| Phi 3.5 Mini | ~2.3GB | 5GB | `bartowski/Phi-3.5-mini-instruct-GGUF` |

---

## Re-enabling MLX Models (macOS)

To re-enable MLX models for Apple Silicon:

1. Edit `src/exo/shared/models/model_cards.py`
2. Uncomment the `MODEL_CARDS` dict
3. The `ALL_MODEL_CARDS` line already includes both:
   ```python
   ALL_MODEL_CARDS: dict[str, ModelCard] = {**MODEL_CARDS, **GGUF_MODEL_CARDS}
   ```

---

## Testing Checklist

- [x] llama-cpp-python imports on Device 1
- [x] llama-cpp-python imports on Device 2
- [x] Dashboard shows only GGUF models
- [x] GGUF model downloaded successfully
- [ ] Inference runs without crash
- [ ] Chat response generated
- [ ] Multi-node inference works

---

## Related Files

| File | Purpose |
|------|---------|
| `src/exo/master/api.py` | API endpoints including `/models` |
| `src/exo/worker/download/download_utils.py` | Model download logic with GGUF support |
| `src/exo/shared/models/model_cards.py` | Model definitions (GGUF + commented MLX) |
| `src/exo/worker/runner/runner.py` | Runner subprocess logic |
| `scripts/download_model.sh` | Manual model download script |

---

*Created: December 2024*
*Last Updated: December 2024*
