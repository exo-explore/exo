# Mobile & Cluster LLM Models Guide

This guide contains curated lists of LLM models optimized for two scenarios:
1. **Mobile/Edge Devices** - Small, quantized models that can run on phones and low-resource devices
2. **Distributed Clusters** - Larger, more powerful models that benefit from running across multiple devices

---

## 📱 Models for Mobile/Phone Devices

These models are small enough to run on smartphones (especially Android via Termux with llama.cpp). Focus on GGUF quantized versions (Q4_K_M or similar) for best performance.

### Ultra-Tiny Models (< 1B parameters)

| Model Name | Parameters | Use Case | Hugging Face Link |
|------------|------------|----------|-------------------|
| **Qwen 2.5 0.5B Instruct** | 0.5B | General chat, code assistance | [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) |
| **SmolLM2 135M** | 135M | Ultra-lightweight chat | [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) |
| **SmolLM2 360M** | 360M | Lightweight chat, edge devices | [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| **Danube 3 500M** | 500M | General purpose, efficient | [h2oai/h2o-danube3-500m-chat](https://huggingface.co/h2oai/h2o-danube3-500m-chat) |

### Small Models (1B - 2B parameters)

| Model Name | Parameters | Use Case | Hugging Face Link |
|------------|------------|----------|-------------------|
| **Llama 3.2 1B Instruct** | 1B | General chat, mobile-optimized | [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) |
| **TinyLlama 1.1B Chat** | 1.1B | Lightweight chat, edge devices | [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) |
| **Qwen 2.5 1.5B Instruct** | 1.5B | General chat, code, reasoning | [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) |
| **DeepSeek R1 Distill Qwen 1.5B** | 1.5B | Reasoning, chain-of-thought | [bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) |
| **SmolLM2 1.7B Instruct** | 1.7B | General chat, code | [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) |
| **StableLM 2 1.6B Chat** | 1.6B | General conversation | [bartowski/stablelm-2-1_6b-chat-GGUF](https://huggingface.co/bartowski/stablelm-2-1_6b-chat-GGUF) |
| **Gemma 2 2B Instruct** | 2B | General chat, Google's efficient model | [bartowski/gemma-2-2b-it-GGUF](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF) |

### Medium-Small Models (3B - 4B parameters)

| Model Name | Parameters | Use Case | Hugging Face Link |
|------------|------------|----------|-------------------|
| **Llama 3.2 3B Instruct** | 3B | General chat, good quality | [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| **Qwen 2.5 3B Instruct** | 3B | General chat, code, math | [Qwen/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) |
| **Phi-3 Mini 4K Instruct** | 3.8B | General chat, reasoning | [bartowski/Phi-3-mini-4k-instruct-GGUF](https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF) |
| **Phi-3.5 Mini Instruct** | 3.8B | Improved Phi-3, multilingual | [bartowski/Phi-3.5-mini-instruct-GGUF](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF) |
| **Danube 3 4B Chat** | 4B | General purpose | [h2oai/h2o-danube3-4b-chat](https://huggingface.co/h2oai/h2o-danube3-4b-chat) |

### Capable Mobile Models (7B - 9B parameters)

These require high-end phones (8GB+ RAM) or tablets:

| Model Name | Parameters | Use Case | Hugging Face Link |
|------------|------------|----------|-------------------|
| **Llama 3.1 8B Instruct** | 8B | High-quality general chat | [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) |
| **Qwen 2.5 7B Instruct** | 7B | General chat, code, math | [Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) |
| **DeepSeek R1 Distill Llama 8B** | 8B | Advanced reasoning | [bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF) |
| **DeepSeek R1 Distill Qwen 7B** | 7B | Reasoning, chain-of-thought | [bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF) |
| **Gemma 2 9B Instruct** | 9B | High-quality Google model | [bartowski/gemma-2-9b-it-GGUF](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF) |
| **Mistral 7B Instruct v0.3** | 7B | General chat, efficient | [bartowski/Mistral-7B-Instruct-v0.3-GGUF](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF) |

### Specialized Mobile Models

| Model Name | Parameters | Use Case | Hugging Face Link |
|------------|------------|----------|-------------------|
| **Qwen 2.5 Coder 1.5B** | 1.5B | Code generation/completion | [Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF) |
| **Qwen 2.5 Coder 3B** | 3B | Code generation | [Qwen/Qwen2.5-Coder-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF) |
| **Qwen 2.5 Coder 7B** | 7B | Advanced code assistance | [Qwen/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) |
| **CodeGemma 2B** | 2B | Code completion | [bartowski/codegemma-2b-GGUF](https://huggingface.co/bartowski/codegemma-2b-GGUF) |
| **CodeGemma 7B Instruct** | 7B | Code generation | [bartowski/codegemma-7b-it-GGUF](https://huggingface.co/bartowski/codegemma-7b-it-GGUF) |

---

## 🖥️ Models for Distributed Cluster Inference

These larger models benefit significantly from distributed inference across multiple devices using frameworks like exo. They're too large for single consumer devices but can be split across a cluster.

### Medium-Large Models (14B - 32B parameters)

Good starting point for small clusters (2-4 devices):

| Model Name | Parameters | Use Case | Hugging Face Link |
|------------|------------|----------|-------------------|
| **Qwen 2.5 14B Instruct** | 14B | High-quality general purpose | [Qwen/Qwen2.5-14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF) |
| **DeepSeek R1 Distill Qwen 14B** | 14B | Advanced reasoning | [bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF) |
| **Qwen 2.5 32B Instruct** | 32B | Excellent reasoning/code | [Qwen/Qwen2.5-32B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF) |
| **DeepSeek R1 Distill Qwen 32B** | 32B | State-of-art reasoning | [bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF) |
| **Gemma 2 27B Instruct** | 27B | High-quality Google model | [bartowski/gemma-2-27b-it-GGUF](https://huggingface.co/bartowski/gemma-2-27b-it-GGUF) |
| **Mistral Small 24B Instruct** | 24B | Efficient, high-quality | [bartowski/Mistral-Small-24B-Instruct-2501-GGUF](https://huggingface.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF) |

### Large Models (70B - 72B parameters)

Excellent for medium clusters (4-8 devices):

| Model Name | Parameters | Use Case | Hugging Face Link |
|------------|------------|----------|-------------------|
| **Llama 3.1 70B Instruct** | 70B | Excellent general purpose | [bartowski/Meta-Llama-3.1-70B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF) |
| **Llama 3.3 70B Instruct** | 70B | Latest Llama, improved | [bartowski/Llama-3.3-70B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF) |
| **Qwen 2.5 72B Instruct** | 72B | Top-tier reasoning/code | [Qwen/Qwen2.5-72B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF) |
| **DeepSeek R1 Distill Llama 70B** | 70B | Excellent reasoning | [bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF) |
| **Qwen 2.5 Coder 32B Instruct** | 32B | Best open-source code model | [Qwen/Qwen2.5-Coder-32B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF) |

### Mixture-of-Experts (MoE) Models

Efficient for their capability, good for clusters:

| Model Name | Parameters (Active) | Use Case | Hugging Face Link |
|------------|---------------------|----------|-------------------|
| **Mixtral 8x7B Instruct** | 47B (12B active) | Efficient MoE, great quality | [bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF) |
| **Mixtral 8x22B Instruct** | 176B (39B active) | High-performance MoE | [bartowski/Mixtral-8x22B-Instruct-v0.1-GGUF](https://huggingface.co/bartowski/Mixtral-8x22B-Instruct-v0.1-GGUF) |
| **DeepSeek V2 Lite** | 16B (2.4B active) | Efficient MoE | [bartowski/DeepSeek-V2-Lite-Chat-GGUF](https://huggingface.co/bartowski/DeepSeek-V2-Lite-Chat-GGUF) |
| **DeepSeek V2** | 236B | Large MoE model | [bartowski/DeepSeek-V2-Chat-GGUF](https://huggingface.co/bartowski/DeepSeek-V2-Chat-GGUF) |
| **DBRX Instruct** | 132B (36B active) | Databricks MoE | [bartowski/dbrx-instruct-GGUF](https://huggingface.co/bartowski/dbrx-instruct-GGUF) |

### Frontier-Class Models (100B+ parameters)

For large clusters (8+ devices) with significant resources:

| Model Name | Parameters | Use Case | Hugging Face Link |
|------------|------------|----------|-------------------|
| **Llama 3.1 405B Instruct** | 405B | Top-tier open model | [bartowski/Meta-Llama-3.1-405B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-405B-Instruct-GGUF) |
| **DeepSeek V3** | 671B (37B active) | State-of-art MoE | [mlx-community/DeepSeek-V3.1-4bit](https://huggingface.co/mlx-community/DeepSeek-V3.1-4bit) |
| **DeepSeek R1** | 671B | Best reasoning model | [mlx-community/DeepSeek-R1-4bit](https://huggingface.co/mlx-community/DeepSeek-R1-4bit) |
| **Qwen 2.5 Max** | ~1T (unconfirmed) | Alibaba's flagship | [Coming soon - check Qwen HF page](https://huggingface.co/Qwen) |
| **Command R+** | 104B | Enterprise chat, RAG | [bartowski/c4ai-command-r-plus-GGUF](https://huggingface.co/bartowski/c4ai-command-r-plus-GGUF) |

### Vision-Language Models (Multimodal)

For clusters that need image understanding:

| Model Name | Parameters | Use Case | Hugging Face Link |
|------------|------------|----------|-------------------|
| **LLaVA 1.6 Mistral 7B** | 7B | Image + text understanding | [bartowski/llava-v1.6-mistral-7b-hf-GGUF](https://huggingface.co/bartowski/llava-v1.6-mistral-7b-hf-GGUF) |
| **Qwen2-VL 7B Instruct** | 7B | Vision-language tasks | [Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4) |
| **Qwen2-VL 72B Instruct** | 72B | High-quality vision-language | [Qwen/Qwen2-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct) |
| **Pixtral 12B** | 12B | Mistral's vision model | [bartowski/Pixtral-12B-2409-GGUF](https://huggingface.co/bartowski/Pixtral-12B-2409-GGUF) |

---

## 📊 Recommended Quantization Levels

When downloading GGUF models, choose quantization based on your RAM:

| Quantization | Memory Savings | Quality Impact | Recommended For |
|--------------|----------------|----------------|-----------------|
| **Q8_0** | ~50% | Minimal | Desktop/high-end devices |
| **Q6_K** | ~55% | Very slight | Desktop with good RAM |
| **Q5_K_M** | ~60% | Slight | Balanced quality/size |
| **Q4_K_M** | ~70% | Noticeable but good | **Mobile recommended** |
| **Q4_K_S** | ~72% | Moderate | Lower RAM devices |
| **Q3_K_M** | ~78% | Significant | Very limited RAM |
| **Q2_K** | ~85% | Heavy | Emergency use only |

### Memory Requirements (Approximate)

For Q4_K_M quantization:

| Model Size | VRAM/RAM Required |
|------------|-------------------|
| 0.5B | ~400MB |
| 1B | ~750MB |
| 1.5B | ~1GB |
| 3B | ~2GB |
| 7B | ~4.5GB |
| 8B | ~5GB |
| 13-14B | ~8GB |
| 32B | ~20GB |
| 70B | ~42GB |
| 405B | ~250GB |

---

## 🔗 Useful Resources

- **Hugging Face GGUF Models**: [huggingface.co/models?library=gguf](https://huggingface.co/models?library=gguf)
- **bartowski's GGUF Collection**: [huggingface.co/bartowski](https://huggingface.co/bartowski) - Excellent quantized models
- **TheBloke's Models** (Legacy): [huggingface.co/TheBloke](https://huggingface.co/TheBloke) - Classic GGUF provider
- **llama.cpp**: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) - Inference engine for GGUF
- **Ollama**: [ollama.ai](https://ollama.ai) - Easy local LLM deployment
- **exo**: [github.com/exo-explore/exo](https://github.com/exo-explore/exo) - Distributed LLM inference

---

## 💡 Recommendations by Use Case

### For Android Phones (4-6GB RAM)
1. **Qwen 2.5 0.5B** - Best ultra-light option
2. **Llama 3.2 1B** - Good balance
3. **TinyLlama 1.1B** - Reliable fallback

### For High-End Phones (8-12GB RAM)
1. **Llama 3.2 3B** - Great quality
2. **Phi-3.5 Mini** - Excellent reasoning
3. **Qwen 2.5 3B** - Strong all-around

### For Small Clusters (2-4 phones/devices)
1. **Qwen 2.5 14B** - High quality
2. **Mistral Small 24B** - Efficient
3. **Gemma 2 27B** - Google quality

### For Medium Clusters (4-8 devices)
1. **Llama 3.3 70B** - Best general purpose
2. **Qwen 2.5 72B** - Excellent reasoning
3. **DeepSeek R1 Distill 70B** - Best reasoning

### For Large Clusters (8+ devices)
1. **DeepSeek V3** - State-of-art MoE
2. **Llama 3.1 405B** - Massive capability
3. **DeepSeek R1** - Best reasoning available

